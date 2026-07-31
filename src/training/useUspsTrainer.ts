import { useCallback, useEffect, useRef, useState } from "react";
import {
  ARCHITECTURE,
  BATCH_SIZE,
  PARAMETER_COUNT,
  TOTAL_EPOCHS,
  type LandscapeData,
  type TrainerCommand,
  type TrainerProgress,
  type TrainerWorkerMessage,
} from "./types";

export interface LossHistoryPoint {
  step: number;
  loss: number;
}

export interface UspsTrainerState extends TrainerProgress {
  phase: "loading" | "ready" | "error";
  error: string | null;
  running: boolean;
  trainCount: number;
  testCount: number;
  snapshotVersion: number;
  batchLoss: number | null;
  testAccuracy: number | null;
  currentLabel: number;
  prediction: number;
  sampleLoss: number;
  pixels: Uint8Array;
  activations: Float32Array[];
  parameterValues: Float32Array;
  parameterGradients: Float32Array;
  landscape: LandscapeData | null;
  lossHistory: LossHistoryPoint[];
}

const initialProgress: TrainerProgress = {
  epoch: 1,
  totalEpochs: TOTAL_EPOCHS,
  batch: 1,
  batchesPerEpoch: 228,
  sample: 0,
  samplesInBatch: BATCH_SIZE,
  optimizerStep: 0,
  processedSamples: 0,
  totalTrainingSamples: 7291 * TOTAL_EPOCHS,
};

const initialState: UspsTrainerState = {
  ...initialProgress,
  phase: "loading",
  error: null,
  running: false,
  trainCount: 0,
  testCount: 0,
  snapshotVersion: 0,
  batchLoss: null,
  testAccuracy: null,
  currentLabel: 0,
  prediction: 0,
  sampleLoss: 0,
  pixels: new Uint8Array(ARCHITECTURE[0]),
  activations: ARCHITECTURE.map((size) => new Float32Array(size)),
  parameterValues: new Float32Array(PARAMETER_COUNT),
  parameterGradients: new Float32Array(PARAMETER_COUNT),
  landscape: null,
  lossHistory: [],
};

export function useUspsTrainer() {
  const [state, setState] = useState<UspsTrainerState>(initialState);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("./trainer.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    worker.addEventListener(
      "message",
      (event: MessageEvent<TrainerWorkerMessage>) => {
        const message = event.data;

        if (message.type === "status") {
          setState((current) => ({
            ...current,
            phase: message.phase,
            running: message.running,
            trainCount: message.trainCount ?? current.trainCount,
            testCount: message.testCount ?? current.testCount,
          }));
          return;
        }

        if (message.type === "error") {
          setState((current) => ({
            ...current,
            phase: "error",
            error: message.message,
            running: false,
          }));
          return;
        }

        if (message.type === "landscape") {
          setState((current) => {
            if (
              message.landscape.snapshotVersion !== current.snapshotVersion
            ) {
              return current;
            }
            return { ...current, landscape: message.landscape };
          });
          return;
        }

        if (message.type === "sample") {
          setState((current) => ({
            ...current,
            ...pickProgress(message),
            running: message.running,
            currentLabel: message.currentLabel,
            prediction: message.prediction,
            sampleLoss: message.sampleLoss,
            pixels: message.pixels,
            activations: message.activations,
          }));
          return;
        }

        setState((current) => {
          const completedBatch =
            message.batchLoss !== null &&
            message.sample === message.samplesInBatch &&
            message.optimizerStep > 0;
          const lastHistoryPoint = current.lossHistory.at(-1);
          const lossHistory =
            completedBatch &&
            lastHistoryPoint?.step !== message.optimizerStep
              ? [
                  ...current.lossHistory,
                  {
                    step: message.optimizerStep,
                    loss: message.batchLoss ?? 0,
                  },
                ]
              : current.lossHistory;

          return {
            ...current,
            ...pickProgress(message),
            phase: "ready",
            error: null,
            running: message.running,
            snapshotVersion: message.snapshotVersion,
            batchLoss: message.batchLoss,
            testAccuracy: message.testAccuracy,
            currentLabel: message.currentLabel,
            prediction: message.prediction,
            sampleLoss: message.sampleLoss,
            pixels: message.pixels,
            activations: message.activations,
            parameterValues: message.parameterValues,
            parameterGradients: message.parameterGradients,
            landscape: message.landscape,
            lossHistory,
          };
        });
      },
    );

    post(worker, { type: "initialize" });
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const send = useCallback((command: TrainerCommand) => {
    const worker = workerRef.current;
    if (worker) post(worker, command);
  }, []);

  useEffect(() => {
    if (state.snapshotVersion > 0) {
      send({
        type: "ack-snapshot",
        snapshotVersion: state.snapshotVersion,
      });
    }
  }, [send, state.snapshotVersion]);

  const start = useCallback(() => send({ type: "start" }), [send]);
  const pause = useCallback(() => send({ type: "pause" }), [send]);
  const stepSample = useCallback(
    () => send({ type: "step-sample" }),
    [send],
  );
  const stepBatch = useCallback(() => send({ type: "step-batch" }), [send]);
  const reset = useCallback(() => {
    setState((current) => ({
      ...current,
      lossHistory: [],
      landscape: null,
      running: false,
    }));
    send({ type: "reset" });
  }, [send]);
  const selectParameter = useCallback(
    (parameterId: string) =>
      send({ type: "select-parameter", parameterId }),
    [send],
  );

  return {
    state,
    start,
    pause,
    stepSample,
    stepBatch,
    reset,
    selectParameter,
  };
}

function post(worker: Worker, command: TrainerCommand) {
  worker.postMessage(command);
}

function pickProgress(progress: TrainerProgress): TrainerProgress {
  return {
    epoch: progress.epoch,
    totalEpochs: progress.totalEpochs,
    batch: progress.batch,
    batchesPerEpoch: progress.batchesPerEpoch,
    sample: progress.sample,
    samplesInBatch: progress.samplesInBatch,
    optimizerStep: progress.optimizerStep,
    processedSamples: progress.processedSamples,
    totalTrainingSamples: progress.totalTrainingSamples,
  };
}
