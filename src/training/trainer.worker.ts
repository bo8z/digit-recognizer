/// <reference lib="webworker" />

import {
  BATCH_SIZE,
  LEARNING_RATE,
  TOTAL_EPOCHS,
  buildParameterDescriptors,
  parameterIndex,
  type LandscapeData,
  type TrainerCommand,
  type TrainerErrorMessage,
  type TrainerLandscapeMessage,
  type TrainerSnapshotMessage,
  type TrainerStatusMessage,
} from "./types";
import {
  UspsNetwork,
  computeExactLandscape,
  decodeUspsDataset,
  samplePixels,
  type NetworkGradients,
  type NetworkScratch,
  type UspsDataset,
} from "./usps-network";

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const RUN_INTERVAL_MS = 18;
const DATA_BASE_URL = import.meta.env.BASE_URL;
const descriptors = buildParameterDescriptors();
const descriptorById = new Map(
  descriptors.map((descriptor) => [descriptor.id, descriptor]),
);

let trainDataset: UspsDataset | null = null;
let testDataset: UspsDataset | null = null;
let network = new UspsNetwork();
let visualizationNetwork = new UspsNetwork(1);
let gradients: NetworkGradients = network.createGradients();
let scratch: NetworkScratch = network.createScratch();
let order = new Uint32Array(0);
let epochIndex = 0;
let batchStart = 0;
let sampleOffset = 0;
let optimizerStep = 0;
let processedSamples = 0;
let accumulatedLoss = 0;
let accumulatedCount = 0;
let currentSampleIndex = 0;
let currentPrediction = 0;
let currentSampleLoss = 0;
let snapshotVersion = 0;
let running = false;
let initialized = false;
let runTimer: number | null = null;
let awaitingSnapshotAck = false;
let testAccuracy: number | null = null;
let selectedParameterId = "w-0-6-93";
let visualGradients: Float32Array<ArrayBufferLike> = new Float32Array(
  descriptors.length,
);

workerScope.addEventListener("message", (event: MessageEvent<TrainerCommand>) => {
  void handleCommand(event.data).catch(reportError);
});

async function handleCommand(command: TrainerCommand) {
  switch (command.type) {
    case "initialize":
      await initialize();
      break;
    case "start":
      ensureInitialized();
      start();
      break;
    case "pause":
      pause();
      break;
    case "step-sample":
      ensureInitialized();
      pause();
      processOneSample();
      if (accumulatedCount > 0) postSnapshot();
      break;
    case "step-batch":
      ensureInitialized();
      pause();
      processCurrentBatch();
      break;
    case "reset":
      ensureInitialized();
      pause();
      resetTraining();
      postInitialSnapshot();
      break;
    case "ack-snapshot":
      if (command.snapshotVersion >= snapshotVersion) {
        awaitingSnapshotAck = false;
        scheduleRun();
      }
      break;
    case "select-parameter":
      selectedParameterId = command.parameterId;
      postSelectedLandscape();
      break;
  }
}

async function initialize() {
  if (initialized) {
    postStatus("ready");
    return;
  }

  postStatus("loading");
  const [trainResponse, testResponse] = await Promise.all([
    fetch(`${DATA_BASE_URL}usps/train.bin`),
    fetch(`${DATA_BASE_URL}usps/test.bin`),
  ]);
  if (!trainResponse.ok || !testResponse.ok) {
    throw new Error(
      `Unable to load USPS resources (train ${trainResponse.status}, test ${testResponse.status})`,
    );
  }

  const [trainBuffer, testBuffer] = await Promise.all([
    trainResponse.arrayBuffer(),
    testResponse.arrayBuffer(),
  ]);
  trainDataset = decodeUspsDataset(trainBuffer);
  testDataset = decodeUspsDataset(testBuffer);
  initialized = true;
  resetTraining();
  postStatus("ready");
  postInitialSnapshot();
}

function resetTraining() {
  const dataset = requireTrainDataset();
  network = new UspsNetwork(20260730);
  visualizationNetwork = new UspsNetwork(1);
  gradients = network.createGradients();
  scratch = network.createScratch();
  order = createShuffledOrder(dataset.count, 20260730);
  epochIndex = 0;
  batchStart = 0;
  sampleOffset = 0;
  optimizerStep = 0;
  processedSamples = 0;
  accumulatedLoss = 0;
  accumulatedCount = 0;
  currentSampleIndex = order[0] ?? 0;
  currentPrediction = 0;
  currentSampleLoss = 0;
  snapshotVersion = 0;
  testAccuracy = null;
  visualGradients = new Float32Array(descriptors.length);
  awaitingSnapshotAck = false;
}

function start() {
  if (running || epochIndex >= TOTAL_EPOCHS) return;
  running = true;
  postStatus("ready");
  scheduleRun();
}

function pause() {
  running = false;
  awaitingSnapshotAck = false;
  if (runTimer !== null) {
    workerScope.clearTimeout(runTimer);
    runTimer = null;
  }
  if (initialized) postStatus("ready");
}

function scheduleRun() {
  if (!running || awaitingSnapshotAck || runTimer !== null) return;
  runTimer = workerScope.setTimeout(() => {
    runTimer = null;
    try {
      processCurrentBatch();
      if (running) scheduleRun();
    } catch (error) {
      running = false;
      reportError(error);
    }
  }, RUN_INTERVAL_MS);
}

function processCurrentBatch() {
  if (epochIndex >= TOTAL_EPOCHS) return;
  const targetCount = currentBatchSize();
  const startingEpoch = epochIndex;
  const startingBatch = batchStart;
  while (
    epochIndex === startingEpoch &&
    batchStart === startingBatch &&
    accumulatedCount < targetCount
  ) {
    processOneSample();
  }
}

function processOneSample() {
  const dataset = requireTrainDataset();
  if (epochIndex >= TOTAL_EPOCHS) return;

  const position = batchStart + sampleOffset;
  const sampleIndex = order[position];
  const pixels = samplePixels(dataset, sampleIndex);
  const result = network.accumulateSample(
    pixels,
    dataset.labels[sampleIndex],
    gradients,
    scratch,
  );

  currentSampleIndex = sampleIndex;
  currentPrediction = result.prediction;
  currentSampleLoss = result.loss;
  accumulatedLoss += result.loss;
  accumulatedCount++;
  sampleOffset++;
  processedSamples++;

  if (accumulatedCount >= currentBatchSize()) {
    finishBatch();
  }
}

function finishBatch() {
  const dataset = requireTrainDataset();
  const completedBatchSize = accumulatedCount;
  const completedIndices = Array.from(
    order.subarray(batchStart, batchStart + completedBatchSize),
  );

  network.applyGradients(gradients, LEARNING_RATE, completedBatchSize);
  optimizerStep++;
  refreshCurrentSampleView();
  postSnapshot();
  batchStart += completedBatchSize;
  sampleOffset = 0;
  accumulatedCount = 0;
  accumulatedLoss = 0;
  network.clearGradients(gradients);

  if (batchStart >= dataset.count) {
    const evaluation = network.evaluate(requireTestDataset());
    testAccuracy = evaluation.accuracy;
    epochIndex++;

    if (epochIndex >= TOTAL_EPOCHS) {
      running = false;
      postFinalSnapshot(completedIndices);
      postStatus("ready");
      return;
    }

    order = createShuffledOrder(dataset.count, 20260730 + epochIndex * 7919);
    batchStart = 0;
  }
}

function postInitialSnapshot() {
  const dataset = requireTrainDataset();
  const sampleIndex = order[0] ?? 0;
  const probeGradients = network.createGradients();
  const probeScratch = network.createScratch();
  const result = network.accumulateSample(
    samplePixels(dataset, sampleIndex),
    dataset.labels[sampleIndex],
    probeGradients,
    probeScratch,
  );

  currentSampleIndex = sampleIndex;
  currentPrediction = result.prediction;
  currentSampleLoss = result.loss;
  visualizationNetwork.copyFrom(network);
  visualGradients = network.flattenGradients(probeGradients, 1);
  scratch = probeScratch;
  postSnapshotMessage({
    gradientsForView: visualGradients,
    lossForView: result.loss,
    optimizerStepForView: 0,
    sampleForView: 0,
    samplesInBatchForView: currentBatchSize(),
  });
}

function postSnapshot() {
  visualizationNetwork.copyFrom(network);
  visualGradients = network.flattenGradients(gradients, accumulatedCount);

  postSnapshotMessage({
    gradientsForView: visualGradients,
    lossForView:
      accumulatedCount > 0 ? accumulatedLoss / accumulatedCount : null,
    optimizerStepForView: optimizerStep,
    sampleForView: accumulatedCount,
    samplesInBatchForView: currentBatchSize(),
  });
}

function refreshCurrentSampleView() {
  const dataset = requireTrainDataset();
  network.forward(samplePixels(dataset, currentSampleIndex), scratch);
  const output = scratch.activations[3];
  let prediction = 0;
  let loss = 0;

  for (let outputIndex = 0; outputIndex < output.length; outputIndex++) {
    const target = outputIndex === dataset.labels[currentSampleIndex] ? 1 : 0;
    const difference = output[outputIndex] - target;
    loss += 0.5 * difference * difference;
    if (output[outputIndex] > output[prediction]) prediction = outputIndex;
  }

  currentPrediction = prediction;
  currentSampleLoss = loss;
}

function postFinalSnapshot(sampleIndices: number[]) {
  const dataset = requireTrainDataset();
  const probeGradients = network.createGradients();
  const probeScratch = network.createScratch();
  let loss = 0;
  let latestResult = { loss: 0, prediction: 0 };

  for (const sampleIndex of sampleIndices) {
    latestResult = network.accumulateSample(
      samplePixels(dataset, sampleIndex),
      dataset.labels[sampleIndex],
      probeGradients,
      probeScratch,
    );
    loss += latestResult.loss;
    currentSampleIndex = sampleIndex;
  }

  currentPrediction = latestResult.prediction;
  currentSampleLoss = latestResult.loss;
  scratch = probeScratch;
  visualizationNetwork.copyFrom(network);
  visualGradients = network.flattenGradients(
    probeGradients,
    sampleIndices.length,
  );

  postSnapshotMessage({
    gradientsForView: visualGradients,
    lossForView: loss / Math.max(1, sampleIndices.length),
    optimizerStepForView: optimizerStep,
    sampleForView: sampleIndices.length,
    samplesInBatchForView: sampleIndices.length,
  });
}

function postSnapshotMessage({
  gradientsForView,
  lossForView,
  optimizerStepForView,
  sampleForView,
  samplesInBatchForView,
}: {
  gradientsForView: Float32Array;
  lossForView: number | null;
  optimizerStepForView: number;
  sampleForView: number;
  samplesInBatchForView: number;
}) {
  const dataset = requireTrainDataset();
  const pixels = samplePixels(dataset, currentSampleIndex).slice();
  const activations = scratch.activations.map((layer) => layer.slice()) as [
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
  ];
  const parameterValues = visualizationNetwork.flattenParameters();
  const parameterGradients = gradientsForView.slice();
  const landscape = createSelectedLandscape();
  snapshotVersion++;
  if (landscape) landscape.snapshotVersion = snapshotVersion;

  const message: TrainerSnapshotMessage = {
    type: "snapshot",
    snapshotVersion,
    running,
    batchLoss: lossForView,
    testAccuracy,
    currentLabel: dataset.labels[currentSampleIndex],
    prediction: currentPrediction,
    sampleLoss: currentSampleLoss,
    pixels,
    activations,
    parameterValues,
    parameterGradients,
    landscape,
    ...progress(
      optimizerStepForView,
      sampleForView,
      samplesInBatchForView,
    ),
  };

  const transfers: Transferable[] = [
    pixels.buffer,
    parameterValues.buffer,
    parameterGradients.buffer,
    ...activations.map((layer) => layer.buffer),
  ];
  if (landscape) {
    transfers.push(landscape.xValues.buffer, landscape.losses.buffer);
  }
  awaitingSnapshotAck = running;
  workerScope.postMessage(message, transfers);
}

function postSelectedLandscape() {
  if (!initialized) return;
  const landscape = createSelectedLandscape();
  if (!landscape) return;
  landscape.snapshotVersion = snapshotVersion;
  const message: TrainerLandscapeMessage = { type: "landscape", landscape };
  workerScope.postMessage(message, [
    landscape.xValues.buffer,
    landscape.losses.buffer,
  ]);
}

function createSelectedLandscape(): LandscapeData | null {
  const descriptor = descriptorById.get(selectedParameterId);
  if (!descriptor || parameterIndex(selectedParameterId) < 0) return null;

  return computeExactLandscape(
    visualizationNetwork,
    requireTrainDataset(),
    descriptor,
    snapshotVersion,
    optimizerStep,
  );
}

function progress(
  optimizerStepForView: number,
  sampleForView: number,
  samplesInBatchForView: number,
) {
  const dataset = requireTrainDataset();
  const batchesPerEpoch = Math.ceil(dataset.count / BATCH_SIZE);
  const completedEpoch = Math.min(epochIndex, TOTAL_EPOCHS - 1);
  const batchIndex =
    epochIndex >= TOTAL_EPOCHS
      ? batchesPerEpoch - 1
      : Math.min(Math.floor(batchStart / BATCH_SIZE), batchesPerEpoch - 1);

  return {
    epoch: Math.min(epochIndex + 1, TOTAL_EPOCHS),
    totalEpochs: TOTAL_EPOCHS,
    batch: batchIndex + 1,
    batchesPerEpoch,
    sample: sampleForView,
    samplesInBatch: samplesInBatchForView,
    optimizerStep: optimizerStepForView,
    processedSamples: Math.min(
      processedSamples,
      (completedEpoch + 1) * dataset.count,
    ),
    totalTrainingSamples: dataset.count * TOTAL_EPOCHS,
  };
}

function currentBatchSize() {
  const dataset = requireTrainDataset();
  return Math.min(BATCH_SIZE, dataset.count - batchStart);
}

function postStatus(phase: "loading" | "ready") {
  const message: TrainerStatusMessage = {
    type: "status",
    phase,
    running,
    trainCount: trainDataset?.count,
    testCount: testDataset?.count,
  };
  workerScope.postMessage(message);
}

function reportError(error: unknown) {
  const message: TrainerErrorMessage = {
    type: "error",
    message: error instanceof Error ? error.message : String(error),
  };
  workerScope.postMessage(message);
}

function ensureInitialized() {
  if (!initialized) throw new Error("The USPS trainer has not finished loading.");
}

function requireTrainDataset() {
  if (!trainDataset) throw new Error("USPS training data is unavailable.");
  return trainDataset;
}

function requireTestDataset() {
  if (!testDataset) throw new Error("USPS test data is unavailable.");
  return testDataset;
}

function createShuffledOrder(count: number, seed: number) {
  const values = Uint32Array.from({ length: count }, (_, index) => index);
  const random = createRandom(seed);
  for (let index = values.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    const current = values[index];
    values[index] = values[other];
    values[other] = current;
  }
  return values;
}

function createRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
