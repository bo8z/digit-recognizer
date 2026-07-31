export const ARCHITECTURE = [256, 16, 16, 10] as const;
export const BATCH_SIZE = 32;
export const TOTAL_EPOCHS = 20;
export const LEARNING_RATE = 0.08;
export const PARAMETER_COUNT = 4554;

export interface ParameterDescriptor {
  id: string;
  kind: "weight" | "bias";
  layer: number;
  source: number | null;
  destination: number;
  index: number;
}

export interface LandscapeData {
  parameterId: string;
  snapshotVersion: number;
  center: number;
  gradient: number;
  nextValue: number;
  nextLoss: number;
  batchSize: number;
  xValues: Float32Array;
  losses: Float32Array;
}

export interface TrainerProgress {
  epoch: number;
  totalEpochs: number;
  batch: number;
  batchesPerEpoch: number;
  sample: number;
  samplesInBatch: number;
  optimizerStep: number;
  processedSamples: number;
  totalTrainingSamples: number;
}

export interface TrainerSnapshotMessage extends TrainerProgress {
  type: "snapshot";
  snapshotVersion: number;
  running: boolean;
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
}

export interface TrainerSampleMessage extends TrainerProgress {
  type: "sample";
  running: boolean;
  currentLabel: number;
  prediction: number;
  sampleLoss: number;
  pixels: Uint8Array;
  activations: Float32Array[];
}

export interface TrainerStatusMessage {
  type: "status";
  phase: "loading" | "ready";
  running: boolean;
  trainCount?: number;
  testCount?: number;
}

export interface TrainerLandscapeMessage {
  type: "landscape";
  landscape: LandscapeData;
}

export interface TrainerErrorMessage {
  type: "error";
  message: string;
}

export type TrainerWorkerMessage =
  | TrainerSnapshotMessage
  | TrainerSampleMessage
  | TrainerStatusMessage
  | TrainerLandscapeMessage
  | TrainerErrorMessage;

export type TrainerCommand =
  | { type: "initialize" }
  | { type: "start" }
  | { type: "pause" }
  | { type: "step-sample" }
  | { type: "step-batch" }
  | { type: "reset" }
  | { type: "ack-snapshot"; snapshotVersion: number }
  | { type: "select-parameter"; parameterId: string };

export function buildParameterDescriptors(): ParameterDescriptor[] {
  const descriptors: ParameterDescriptor[] = [];

  for (let layer = 0; layer < ARCHITECTURE.length - 1; layer++) {
    const sourceCount = ARCHITECTURE[layer];
    const destinationCount = ARCHITECTURE[layer + 1];

    for (let destination = 0; destination < destinationCount; destination++) {
      for (let source = 0; source < sourceCount; source++) {
        descriptors.push({
          id: `w-${layer}-${destination}-${source}`,
          kind: "weight",
          layer,
          source,
          destination,
          index: descriptors.length,
        });
      }
    }

    for (let destination = 0; destination < destinationCount; destination++) {
      descriptors.push({
        id: `b-${layer}-${destination}`,
        kind: "bias",
        layer,
        source: null,
        destination,
        index: descriptors.length,
      });
    }
  }

  return descriptors;
}

export function parseParameterId(
  parameterId: string,
): Omit<ParameterDescriptor, "index"> | null {
  const weight = parameterId.match(/^w-(\d+)-(\d+)-(\d+)$/);
  if (weight) {
    return {
      id: parameterId,
      kind: "weight",
      layer: Number(weight[1]),
      destination: Number(weight[2]),
      source: Number(weight[3]),
    };
  }

  const bias = parameterId.match(/^b-(\d+)-(\d+)$/);
  if (bias) {
    return {
      id: parameterId,
      kind: "bias",
      layer: Number(bias[1]),
      destination: Number(bias[2]),
      source: null,
    };
  }

  return null;
}

export function parameterIndex(parameterId: string): number {
  const parameter = parseParameterId(parameterId);
  if (!parameter) return -1;

  let offset = 0;
  for (let layer = 0; layer < parameter.layer; layer++) {
    offset +=
      ARCHITECTURE[layer] * ARCHITECTURE[layer + 1] +
      ARCHITECTURE[layer + 1];
  }

  const sourceCount = ARCHITECTURE[parameter.layer];
  const destinationCount = ARCHITECTURE[parameter.layer + 1];
  if (
    parameter.destination < 0 ||
    parameter.destination >= destinationCount ||
    (parameter.kind === "weight" &&
      (parameter.source === null ||
        parameter.source < 0 ||
        parameter.source >= sourceCount))
  ) {
    return -1;
  }

  if (parameter.kind === "weight") {
    return offset + parameter.destination * sourceCount + (parameter.source ?? 0);
  }
  return offset + sourceCount * destinationCount + parameter.destination;
}
