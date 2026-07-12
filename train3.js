import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { gunzipSync } from "zlib";

// This version trains the same kind of network as train.mjs, but uses a tiny
// scalar autograd engine instead of hand-written backpropagation equations.
// It is intentionally educational and much slower than train.mjs.

// --- MNIST Loading ---
function loadImages(path) {
  const gz = readFileSync(path);
  const buf = gunzipSync(gz);
  const count = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const images = [];
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * rows * cols;
    const pixels = new Float64Array(rows * cols);
    for (let j = 0; j < rows * cols; j++) {
      pixels[j] = buf[offset + j] / 255;
    }
    images.push(pixels);
  }
  return images;
}

function loadLabels(path) {
  const gz = readFileSync(path);
  const buf = gunzipSync(gz);
  const count = buf.readUInt32BE(4);
  const labels = [];
  for (let i = 0; i < count; i++) {
    labels.push(buf[8 + i]);
  }
  return labels;
}

// --- Tiny scalar autograd ---
class Value {
  constructor(data, children = [], localGradients = []) {
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.localGradients = localGradients;
  }

  add(other) {
    other = asValue(other);
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }

  mul(other) {
    other = asValue(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }

  pow(exponent) {
    return new Value(this.data ** exponent, [this], [exponent * this.data ** (exponent - 1)]);
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    return this.add(asValue(other).neg());
  }

  relu() {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  sigmoid() {
    const activation = 1 / (1 + Math.exp(-this.data));
    return new Value(activation, [this], [activation * (1 - activation)]);
  }

  backward() {
    const topo = [];
    const visited = new Set();

    function buildTopo(value) {
      if (visited.has(value)) return;
      visited.add(value);
      for (const child of value.children) buildTopo(child);
      topo.push(value);
    }

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const value = topo[i];
      for (let j = 0; j < value.children.length; j++) {
        const child = value.children[j];
        child.grad += value.localGradients[j] * value.grad;
      }
    }
  }
}

function asValue(value) {
  return value instanceof Value ? value : new Value(value);
}

// --- Math helpers ---
function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// --- Neural Network ---
class Network {
  constructor(sizes) {
    this.sizes = sizes;
    this.numLayers = sizes.length;
    this.weights = [];
    this.biases = [];

    for (let layer = 0; layer < sizes.length - 1; layer++) {
      const fanIn = sizes[layer];
      const fanOut = sizes[layer + 1];
      const scale = Math.sqrt(2 / fanIn);

      const layerWeights = [];
      const layerBiases = [];
      for (let neuron = 0; neuron < fanOut; neuron++) {
        const row = [];
        for (let input = 0; input < fanIn; input++) {
          row.push(new Value(randn() * scale));
        }
        layerWeights.push(row);
        layerBiases.push(new Value(0));
      }
      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }

    this.params = [
      ...this.weights.flat(2),
      ...this.biases.flat(),
    ];
  }

  forward(input) {
    let activations = Array.from(input);

    for (let layer = 0; layer < this.numLayers - 1; layer++) {
      const nextActivations = [];
      const isOutputLayer = layer === this.numLayers - 2;

      for (let neuron = 0; neuron < this.sizes[layer + 1]; neuron++) {
        let weightedSum = this.biases[layer][neuron];
        for (let source = 0; source < activations.length; source++) {
          weightedSum = weightedSum.add(
            this.weights[layer][neuron][source].mul(activations[source])
          );
        }
        nextActivations.push(isOutputLayer ? weightedSum.sigmoid() : weightedSum.relu());
      }

      activations = nextActivations;
    }

    return activations;
  }

  loss(input, label) {
    const output = this.forward(input);
    let loss = new Value(0);

    for (let digit = 0; digit < output.length; digit++) {
      const target = digit === label ? 1 : 0;
      const error = output[digit].sub(target);
      loss = loss.add(error.pow(2).mul(0.5));
    }

    return loss;
  }

  train(images, labels, testImages, testLabels, epochs, learningRate, batchSize) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      this.trainEpoch(images, labels, epoch, epochs, learningRate, batchSize);

      const acc = this.evaluate(testImages, testLabels);
      console.log(
        `Epoch ${epoch + 1}/${epochs} — accuracy: ${(acc * 100).toFixed(1)}%`
      );
    }
  }

  trainEpoch(images, labels, epoch, epochs, learningRate, batchSize) {
    const indices = shuffledIndices(images.length);
    const totalBatches = Math.ceil(images.length / batchSize);

    for (let batchStart = 0; batchStart < images.length; batchStart += batchSize) {
      const batchLoss = this.trainBatch(
        images,
        labels,
        indices,
        batchStart,
        batchSize,
        learningRate
      );

      if (batchStart % (batchSize * 50) === 0) {
        const batchNumber = Math.floor(batchStart / batchSize) + 1;
        console.log(
          `epoch ${epoch + 1}/${epochs} batch ${batchNumber}/${totalBatches} loss ${batchLoss.toFixed(4)}`
        );
      }
    }
  }

  trainBatch(images, labels, indices, batchStart, batchSize, learningRate) {
    const batchEnd = Math.min(batchStart + batchSize, images.length);
    const actualBatchSize = batchEnd - batchStart;
    let batchLoss = 0;

    this.zeroGradients();

    for (let batchIndex = batchStart; batchIndex < batchEnd; batchIndex++) {
      const exampleIndex = indices[batchIndex];
      batchLoss += this.trainExample(images[exampleIndex], labels[exampleIndex]);
    }

    this.applyGradients(learningRate, actualBatchSize);

    return batchLoss / actualBatchSize;
  }

  trainExample(image, label) {
    const loss = this.loss(image, label);
    loss.backward();
    return loss.data;
  }

  zeroGradients() {
    for (const param of this.params) param.grad = 0;
  }

  applyGradients(learningRate, batchSize) {
    const scaledLearningRate = learningRate / batchSize;
    for (const param of this.params) {
      param.data -= scaledLearningRate * param.grad;
    }
  }

  evaluate(images, labels) {
    let correct = 0;
    for (let i = 0; i < images.length; i++) {
      const output = this.forward(images[i]);
      const scores = output.map((value) => value.data);
      const prediction = scores.indexOf(Math.max(...scores));
      if (prediction === labels[i]) correct++;
    }
    return correct / images.length;
  }

  export() {
    return {
      sizes: this.sizes,
      weights: this.weights.map((layer) =>
        layer.map((row) => row.map((weight) => weight.data))
      ),
      biases: this.biases.map((layer) => layer.map((bias) => bias.data)),
    };
  }
}

function shuffledIndices(length) {
  const indices = Array.from({ length }, (_, i) => i);
  for (let i = length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

// --- Main ---
console.log("Loading MNIST data...");
const allTrainImages = loadImages("data/train-images-idx3-ubyte.gz");
const allTrainLabels = loadLabels("data/train-labels-idx1-ubyte.gz");
const allTestImages = loadImages("data/t10k-images-idx3-ubyte.gz");
const allTestLabels = loadLabels("data/t10k-labels-idx1-ubyte.gz");

const trainLimit = Number(process.env.TRAIN_LIMIT ?? allTrainImages.length);
const testLimit = Number(process.env.TEST_LIMIT ?? allTestImages.length);
const trainImages = allTrainImages.slice(0, trainLimit);
const trainLabels = allTrainLabels.slice(0, trainLimit);
const testImages = allTestImages.slice(0, testLimit);
const testLabels = allTestLabels.slice(0, testLimit);

console.log(
  `Loaded ${trainImages.length} training images, ${testImages.length} test images`
);

const sizes = (process.env.SIZES ?? "784,16,16,10")
  .split(",")
  .map((value) => Number(value.trim()));
const epochs = Number(process.env.EPOCHS ?? 20);
const learningRate = Number(process.env.LEARNING_RATE ?? 0.5);
const batchSize = Number(process.env.BATCH_SIZE ?? 32);
const outputPath = process.env.OUTPUT_PATH ?? "public/weights-autograd.json";

const net = new Network(sizes);

console.log("Training with scalar autograd...");
net.train(trainImages, trainLabels, testImages, testLabels, epochs, learningRate, batchSize);

mkdirSync("public", { recursive: true });
writeFileSync(outputPath, JSON.stringify(net.export()));
console.log(`Weights saved to ${outputPath}`);
