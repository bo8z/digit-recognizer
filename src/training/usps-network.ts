import {
  ARCHITECTURE,
  PARAMETER_COUNT,
  type LandscapeData,
  type ParameterDescriptor,
} from "./types";

const INPUT_SIZE = ARCHITECTURE[0];
const HIDDEN_1_SIZE = ARCHITECTURE[1];
const HIDDEN_2_SIZE = ARCHITECTURE[2];
const OUTPUT_SIZE = ARCHITECTURE[3];

export interface UspsDataset {
  count: number;
  width: number;
  height: number;
  labels: Uint8Array;
  pixels: Uint8Array;
}

export interface NetworkGradients {
  weights: [Float32Array, Float32Array, Float32Array];
  biases: [Float32Array, Float32Array, Float32Array];
}

export interface NetworkScratch {
  activations: [
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
  ];
  weightedInputs: [Float32Array, Float32Array, Float32Array];
  deltas: [Float32Array, Float32Array, Float32Array];
}

export interface SampleResult {
  loss: number;
  prediction: number;
}

export interface EvaluationResult {
  loss: number;
  accuracy: number;
}

export class UspsNetwork {
  readonly weights: [Float32Array, Float32Array, Float32Array];
  readonly biases: [Float32Array, Float32Array, Float32Array];

  constructor(seed = 20260730) {
    this.weights = [
      new Float32Array(HIDDEN_1_SIZE * INPUT_SIZE),
      new Float32Array(HIDDEN_2_SIZE * HIDDEN_1_SIZE),
      new Float32Array(OUTPUT_SIZE * HIDDEN_2_SIZE),
    ];
    this.biases = [
      new Float32Array(HIDDEN_1_SIZE),
      new Float32Array(HIDDEN_2_SIZE),
      new Float32Array(OUTPUT_SIZE),
    ];
    this.initialize(seed);
  }

  initialize(seed: number) {
    const random = createNormalRandom(seed);
    const fanIns = [INPUT_SIZE, HIDDEN_1_SIZE, HIDDEN_2_SIZE];

    this.weights.forEach((layer, layerIndex) => {
      const scale =
        layerIndex < this.weights.length - 1
          ? Math.sqrt(2 / fanIns[layerIndex])
          : Math.sqrt(1 / fanIns[layerIndex]);
      for (let index = 0; index < layer.length; index++) {
        layer[index] = random() * scale;
      }
    });
    this.biases.forEach((layer) => layer.fill(0));
  }

  copyFrom(source: UspsNetwork) {
    this.weights.forEach((layer, index) => layer.set(source.weights[index]));
    this.biases.forEach((layer, index) => layer.set(source.biases[index]));
  }

  createScratch(): NetworkScratch {
    return {
      activations: [
        new Float32Array(INPUT_SIZE),
        new Float32Array(HIDDEN_1_SIZE),
        new Float32Array(HIDDEN_2_SIZE),
        new Float32Array(OUTPUT_SIZE),
      ],
      weightedInputs: [
        new Float32Array(HIDDEN_1_SIZE),
        new Float32Array(HIDDEN_2_SIZE),
        new Float32Array(OUTPUT_SIZE),
      ],
      deltas: [
        new Float32Array(HIDDEN_1_SIZE),
        new Float32Array(HIDDEN_2_SIZE),
        new Float32Array(OUTPUT_SIZE),
      ],
    };
  }

  createGradients(): NetworkGradients {
    return {
      weights: [
        new Float32Array(this.weights[0].length),
        new Float32Array(this.weights[1].length),
        new Float32Array(this.weights[2].length),
      ],
      biases: [
        new Float32Array(this.biases[0].length),
        new Float32Array(this.biases[1].length),
        new Float32Array(this.biases[2].length),
      ],
    };
  }

  clearGradients(gradients: NetworkGradients) {
    gradients.weights.forEach((layer) => layer.fill(0));
    gradients.biases.forEach((layer) => layer.fill(0));
  }

  forward(pixels: Uint8Array, scratch: NetworkScratch) {
    const [input, hidden1, hidden2, output] = scratch.activations;
    const [z1, z2, z3] = scratch.weightedInputs;

    for (let source = 0; source < INPUT_SIZE; source++) {
      input[source] = pixels[source] / 255;
    }

    denseRelu(
      input,
      hidden1,
      z1,
      this.weights[0],
      this.biases[0],
      INPUT_SIZE,
    );
    denseRelu(
      hidden1,
      hidden2,
      z2,
      this.weights[1],
      this.biases[1],
      HIDDEN_1_SIZE,
    );
    denseSigmoid(
      hidden2,
      output,
      z3,
      this.weights[2],
      this.biases[2],
      HIDDEN_2_SIZE,
    );
  }

  accumulateSample(
    pixels: Uint8Array,
    label: number,
    gradients: NetworkGradients,
    scratch: NetworkScratch,
  ): SampleResult {
    this.forward(pixels, scratch);

    const [input, hidden1, hidden2, output] = scratch.activations;
    const [z1, z2] = scratch.weightedInputs;
    const [delta1, delta2, delta3] = scratch.deltas;
    let loss = 0;
    let prediction = 0;
    let largestOutput = -Infinity;

    for (let destination = 0; destination < OUTPUT_SIZE; destination++) {
      const target = destination === label ? 1 : 0;
      const activation = output[destination];
      const difference = activation - target;
      loss += 0.5 * difference * difference;
      delta3[destination] = difference * activation * (1 - activation);
      if (activation > largestOutput) {
        largestOutput = activation;
        prediction = destination;
      }
    }

    for (let destination = 0; destination < HIDDEN_2_SIZE; destination++) {
      let propagated = 0;
      for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {
        propagated +=
          this.weights[2][outputIndex * HIDDEN_2_SIZE + destination] *
          delta3[outputIndex];
      }
      delta2[destination] = z2[destination] > 0 ? propagated : 0;
    }

    for (let destination = 0; destination < HIDDEN_1_SIZE; destination++) {
      let propagated = 0;
      for (let hidden2Index = 0; hidden2Index < HIDDEN_2_SIZE; hidden2Index++) {
        propagated +=
          this.weights[1][hidden2Index * HIDDEN_1_SIZE + destination] *
          delta2[hidden2Index];
      }
      delta1[destination] = z1[destination] > 0 ? propagated : 0;
    }

    accumulateDenseGradients(
      input,
      delta1,
      gradients.weights[0],
      gradients.biases[0],
      INPUT_SIZE,
    );
    accumulateDenseGradients(
      hidden1,
      delta2,
      gradients.weights[1],
      gradients.biases[1],
      HIDDEN_1_SIZE,
    );
    accumulateDenseGradients(
      hidden2,
      delta3,
      gradients.weights[2],
      gradients.biases[2],
      HIDDEN_2_SIZE,
    );

    return { loss, prediction };
  }

  applyGradients(gradients: NetworkGradients, learningRate: number, count: number) {
    if (count <= 0) return;
    const scale = learningRate / count;
    this.weights.forEach((layer, layerIndex) => {
      const gradient = gradients.weights[layerIndex];
      for (let index = 0; index < layer.length; index++) {
        layer[index] -= scale * gradient[index];
      }
    });
    this.biases.forEach((layer, layerIndex) => {
      const gradient = gradients.biases[layerIndex];
      for (let index = 0; index < layer.length; index++) {
        layer[index] -= scale * gradient[index];
      }
    });
  }

  flattenParameters(): Float32Array {
    const values = new Float32Array(PARAMETER_COUNT);
    let offset = 0;
    for (let layer = 0; layer < this.weights.length; layer++) {
      values.set(this.weights[layer], offset);
      offset += this.weights[layer].length;
      values.set(this.biases[layer], offset);
      offset += this.biases[layer].length;
    }
    return values;
  }

  flattenGradients(
    gradients: NetworkGradients,
    sampleCount: number,
  ): Float32Array {
    const values = new Float32Array(PARAMETER_COUNT);
    const divisor = Math.max(1, sampleCount);
    let offset = 0;
    for (let layer = 0; layer < gradients.weights.length; layer++) {
      const weightGradients = gradients.weights[layer];
      for (let index = 0; index < weightGradients.length; index++) {
        values[offset++] = weightGradients[index] / divisor;
      }
      const biasGradients = gradients.biases[layer];
      for (let index = 0; index < biasGradients.length; index++) {
        values[offset++] = biasGradients[index] / divisor;
      }
    }
    return values;
  }

  evaluate(dataset: UspsDataset): EvaluationResult {
    const scratch = this.createScratch();
    let totalLoss = 0;
    let correct = 0;

    for (let sample = 0; sample < dataset.count; sample++) {
      const pixels = samplePixels(dataset, sample);
      this.forward(pixels, scratch);
      const output = scratch.activations[3];
      const label = dataset.labels[sample];
      let prediction = 0;

      for (let destination = 0; destination < OUTPUT_SIZE; destination++) {
        const target = destination === label ? 1 : 0;
        const difference = output[destination] - target;
        totalLoss += 0.5 * difference * difference;
        if (output[destination] > output[prediction]) prediction = destination;
      }
      if (prediction === label) correct++;
    }

    return {
      loss: totalLoss / dataset.count,
      accuracy: correct / dataset.count,
    };
  }
}

export function decodeUspsDataset(buffer: ArrayBuffer): UspsDataset {
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "USPSBIN1") {
    throw new Error(`Invalid USPS resource header: ${magic}`);
  }

  const view = new DataView(buffer);
  const count = view.getUint32(8, true);
  const width = view.getUint16(12, true);
  const height = view.getUint16(14, true);
  const labelOffset = 16;
  const pixelOffset = labelOffset + count;
  const expectedSize = pixelOffset + count * width * height;

  if (width !== 16 || height !== 16 || buffer.byteLength !== expectedSize) {
    throw new Error(
      `Invalid USPS resource dimensions or size (${width}×${height}, ${buffer.byteLength} bytes)`,
    );
  }

  return {
    count,
    width,
    height,
    labels: new Uint8Array(buffer, labelOffset, count),
    pixels: new Uint8Array(buffer, pixelOffset, count * width * height),
  };
}

export function samplePixels(dataset: UspsDataset, sample: number): Uint8Array {
  const offset = sample * INPUT_SIZE;
  return dataset.pixels.subarray(offset, offset + INPUT_SIZE);
}

export function computeExactLandscape(
  network: UspsNetwork,
  dataset: UspsDataset,
  sampleIndices: number[],
  parameter: ParameterDescriptor,
  snapshotVersion: number,
  gradient: number,
  learningRate: number,
): LandscapeData {
  const pointCount = 81;
  const center = readParameter(network, parameter);
  const radius = Math.max(
    0.45,
    Math.min(1.5, Math.abs(center) * 1.5 + Math.abs(gradient) * learningRate * 8),
  );
  const xValues = new Float32Array(pointCount);
  const losses = new Float32Array(pointCount);
  const scratch = network.createScratch();
  const samples = sampleIndices.length > 0 ? sampleIndices : [0];
  const caches = samples.map((sampleIndex) => {
    const pixels = samplePixels(dataset, sampleIndex);
    network.forward(pixels, scratch);
    return {
      label: dataset.labels[sampleIndex],
      input: scratch.activations[0].slice(),
      hidden1: scratch.activations[1].slice(),
      hidden2: scratch.activations[2].slice(),
      output: scratch.activations[3].slice(),
      z1: scratch.weightedInputs[0].slice(),
      z2: scratch.weightedInputs[1].slice(),
      z3: scratch.weightedInputs[2].slice(),
    };
  });

  for (let point = 0; point < pointCount; point++) {
    const candidate =
      center - radius + (point / (pointCount - 1)) * radius * 2;
    xValues[point] = candidate;
    let totalLoss = 0;

    for (const cache of caches) {
      totalLoss += evaluateChangedParameterLoss(
        network,
        cache,
        parameter,
        candidate - center,
      );
    }
    losses[point] = totalLoss / caches.length;
  }

  const nextValue = center - learningRate * gradient;
  let nextLoss = 0;
  for (const cache of caches) {
    nextLoss += evaluateChangedParameterLoss(
      network,
      cache,
      parameter,
      nextValue - center,
    );
  }

  return {
    parameterId: parameter.id,
    snapshotVersion,
    center,
    gradient,
    nextValue,
    nextLoss: nextLoss / caches.length,
    batchSize: samples.length,
    xValues,
    losses,
  };
}

function evaluateChangedParameterLoss(
  network: UspsNetwork,
  cache: {
    label: number;
    input: Float32Array;
    hidden1: Float32Array;
    hidden2: Float32Array;
    output: Float32Array;
    z1: Float32Array;
    z2: Float32Array;
    z3: Float32Array;
  },
  parameter: ParameterDescriptor,
  difference: number,
) {
  if (parameter.layer === 0) {
    const inputFactor =
      parameter.kind === "weight" ? cache.input[parameter.source ?? 0] : 1;
    const changedZ1 =
      cache.z1[parameter.destination] + difference * inputFactor;
    const changedHidden1 = Math.max(0, changedZ1);
    const hidden1Difference =
      changedHidden1 - cache.hidden1[parameter.destination];
    const changedHidden2 = new Float32Array(HIDDEN_2_SIZE);

    for (let destination = 0; destination < HIDDEN_2_SIZE; destination++) {
      const changedZ2 =
        cache.z2[destination] +
        network.weights[1][
          destination * HIDDEN_1_SIZE + parameter.destination
        ] *
          hidden1Difference;
      changedHidden2[destination] = Math.max(0, changedZ2);
    }

    let loss = 0;
    for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {
      let changedZ3 = cache.z3[outputIndex];
      const rowOffset = outputIndex * HIDDEN_2_SIZE;
      for (let hidden2Index = 0; hidden2Index < HIDDEN_2_SIZE; hidden2Index++) {
        changedZ3 +=
          network.weights[2][rowOffset + hidden2Index] *
          (changedHidden2[hidden2Index] - cache.hidden2[hidden2Index]);
      }
      const activation = sigmoid(changedZ3);
      const target = outputIndex === cache.label ? 1 : 0;
      const outputDifference = activation - target;
      loss += 0.5 * outputDifference * outputDifference;
    }
    return loss;
  }

  if (parameter.layer === 1) {
    const inputFactor =
      parameter.kind === "weight" ? cache.hidden1[parameter.source ?? 0] : 1;
    const changedZ2 =
      cache.z2[parameter.destination] + difference * inputFactor;
    const changedHidden2 = Math.max(0, changedZ2);
    const hidden2Difference =
      changedHidden2 - cache.hidden2[parameter.destination];
    let loss = 0;

    for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {
      const changedZ3 =
        cache.z3[outputIndex] +
        network.weights[2][
          outputIndex * HIDDEN_2_SIZE + parameter.destination
        ] *
          hidden2Difference;
      const activation = sigmoid(changedZ3);
      const target = outputIndex === cache.label ? 1 : 0;
      const outputDifference = activation - target;
      loss += 0.5 * outputDifference * outputDifference;
    }
    return loss;
  }

  const inputFactor =
    parameter.kind === "weight" ? cache.hidden2[parameter.source ?? 0] : 1;
  let loss = 0;
  for (let outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {
    const activation =
      outputIndex === parameter.destination
        ? sigmoid(cache.z3[outputIndex] + difference * inputFactor)
        : cache.output[outputIndex];
    const target = outputIndex === cache.label ? 1 : 0;
    const outputDifference = activation - target;
    loss += 0.5 * outputDifference * outputDifference;
  }
  return loss;
}

function readParameter(network: UspsNetwork, parameter: ParameterDescriptor) {
  if (parameter.kind === "bias") {
    return network.biases[parameter.layer][parameter.destination];
  }
  const sourceCount = ARCHITECTURE[parameter.layer];
  return network.weights[parameter.layer][
    parameter.destination * sourceCount + (parameter.source ?? 0)
  ];
}

function denseRelu(
  input: Float32Array,
  output: Float32Array,
  weightedInput: Float32Array,
  weights: Float32Array,
  biases: Float32Array,
  sourceCount: number,
) {
  for (let destination = 0; destination < output.length; destination++) {
    let sum = biases[destination];
    const rowOffset = destination * sourceCount;
    for (let source = 0; source < sourceCount; source++) {
      sum += weights[rowOffset + source] * input[source];
    }
    weightedInput[destination] = sum;
    output[destination] = Math.max(0, sum);
  }
}

function denseSigmoid(
  input: Float32Array,
  output: Float32Array,
  weightedInput: Float32Array,
  weights: Float32Array,
  biases: Float32Array,
  sourceCount: number,
) {
  for (let destination = 0; destination < output.length; destination++) {
    let sum = biases[destination];
    const rowOffset = destination * sourceCount;
    for (let source = 0; source < sourceCount; source++) {
      sum += weights[rowOffset + source] * input[source];
    }
    weightedInput[destination] = sum;
    output[destination] = sigmoid(sum);
  }
}

function accumulateDenseGradients(
  input: Float32Array,
  delta: Float32Array,
  weightGradients: Float32Array,
  biasGradients: Float32Array,
  sourceCount: number,
) {
  for (let destination = 0; destination < delta.length; destination++) {
    const currentDelta = delta[destination];
    biasGradients[destination] += currentDelta;
    const rowOffset = destination * sourceCount;
    for (let source = 0; source < sourceCount; source++) {
      weightGradients[rowOffset + source] += currentDelta * input[source];
    }
  }
}

function sigmoid(value: number) {
  if (value >= 0) {
    const exponential = Math.exp(-value);
    return 1 / (1 + exponential);
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function createNormalRandom(seed: number) {
  let state = seed >>> 0;
  let spare: number | null = null;

  const uniform = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };

  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const first = Math.max(uniform(), Number.EPSILON);
    const second = uniform();
    const radius = Math.sqrt(-2 * Math.log(first));
    const angle = 2 * Math.PI * second;
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}
