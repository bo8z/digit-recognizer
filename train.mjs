import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { gunzipSync } from "zlib";

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

// --- Math helpers ---
function randn() {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

function relu(x) {
  return x > 0 ? x : 0;
}

function reluPrime(x) {
  return x > 0 ? 1 : 0;
}

// --- Neural Network ---
class Network {
  constructor(sizes) {
    this.sizes = sizes; // e.g. [784, 16, 16, 10]
    this.numLayers = sizes.length;

    // Initialize weights and biases (He initialization for hidden, Xavier for output)
    this.weights = []; // weights[l][j][k] = weight from neuron k in layer l to neuron j in layer l+1
    this.biases = []; // biases[l][j] = bias for neuron j in layer l+1

    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l];
      const fanOut = sizes[l + 1];
      const scale = Math.sqrt(2 / fanIn);

      const w = [];
      const b = [];
      for (let j = 0; j < fanOut; j++) {
        const row = [];
        for (let k = 0; k < fanIn; k++) {
          row.push(randn() * scale);
        }
        w.push(row);
        b.push(0);
      }
      this.weights.push(w);
      this.biases.push(b);
    }
  }

  forward(input) {
    const activations = [Array.from(input)];
    const zs = [];

    for (let l = 0; l < this.numLayers - 1; l++) {
      const prev = activations[l];
      const size = this.sizes[l + 1];
      const z = new Array(size);
      const a = new Array(size);

      for (let j = 0; j < size; j++) {
        let sum = this.biases[l][j];
        for (let k = 0; k < prev.length; k++) {
          sum += this.weights[l][j][k] * prev[k];
        }
        z[j] = sum;
        // ReLU for hidden layers, sigmoid for output
        if (l < this.numLayers - 2) {
          a[j] = relu(sum);
        } else {
          a[j] = sigmoid(sum);
        }
      }
      zs.push(z);
      activations.push(a);
    }

    return { activations, zs };
  }

  backprop(input, target) {
    const { activations, zs } = this.forward(input);
    const L = this.numLayers - 2; // index of last weight layer

    // Initialize gradient arrays
    const dw = [];
    const db = [];
    for (let l = 0; l <= L; l++) {
      dw.push(
        this.weights[l].map((row) => row.map(() => 0))
      );
      db.push(this.biases[l].map(() => 0));
    }

    // Output layer error: (a - y) * sigmoid'(z)
    const deltas = new Array(L + 1);
    const outputSize = this.sizes[this.numLayers - 1];
    deltas[L] = new Array(outputSize);
    for (let j = 0; j < outputSize; j++) {
      deltas[L][j] =
        (activations[L + 1][j] - target[j]) * sigmoidPrime(zs[L][j]);
    }

    // Hidden layer errors
    for (let l = L - 1; l >= 0; l--) {
      const size = this.sizes[l + 1];
      deltas[l] = new Array(size);
      for (let j = 0; j < size; j++) {
        let sum = 0;
        for (let k = 0; k < this.sizes[l + 2]; k++) {
          sum += this.weights[l + 1][k][j] * deltas[l + 1][k];
        }
        deltas[l][j] = sum * reluPrime(zs[l][j]);
      }
    }

    // Compute gradients
    for (let l = 0; l <= L; l++) {
      for (let j = 0; j < this.sizes[l + 1]; j++) {
        db[l][j] = deltas[l][j];
        for (let k = 0; k < this.sizes[l]; k++) {
          dw[l][j][k] = deltas[l][j] * activations[l][k];
        }
      }
    }

    return { dw, db };
  }

  train(images, labels, epochs, learningRate, batchSize) {
    const n = images.length;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle training data
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Mini-batch SGD
      for (let batch = 0; batch < n; batch += batchSize) {
        const end = Math.min(batch + batchSize, n);
        const m = end - batch;

        // Accumulate gradients
        const dwAccum = this.weights.map((layer) =>
          layer.map((row) => row.map(() => 0))
        );
        const dbAccum = this.biases.map((layer) => layer.map(() => 0));

        for (let i = batch; i < end; i++) {
          const idx = indices[i];
          const target = new Array(10).fill(0);
          target[labels[idx]] = 1;

          const { dw, db } = this.backprop(images[idx], target);

          for (let l = 0; l < dw.length; l++) {
            for (let j = 0; j < dw[l].length; j++) {
              dbAccum[l][j] += db[l][j];
              for (let k = 0; k < dw[l][j].length; k++) {
                dwAccum[l][j][k] += dw[l][j][k];
              }
            }
          }
        }

        // Update weights and biases
        const lr = learningRate / m;
        for (let l = 0; l < this.weights.length; l++) {
          for (let j = 0; j < this.weights[l].length; j++) {
            this.biases[l][j] -= lr * dbAccum[l][j];
            for (let k = 0; k < this.weights[l][j].length; k++) {
              this.weights[l][j][k] -= lr * dwAccum[l][j][k];
            }
          }
        }
      }

      // Evaluate accuracy on test data
      const acc = this.evaluate(testImages, testLabels);
      console.log(
        `Epoch ${epoch + 1}/${epochs} — accuracy: ${(acc * 100).toFixed(1)}%`
      );
    }
  }

  evaluate(images, labels) {
    let correct = 0;
    for (let i = 0; i < images.length; i++) {
      const { activations } = this.forward(images[i]);
      const output = activations[activations.length - 1];
      const prediction = output.indexOf(Math.max(...output));
      if (prediction === labels[i]) correct++;
    }
    return correct / images.length;
  }

  export() {
    return {
      sizes: this.sizes,
      weights: this.weights,
      biases: this.biases,
    };
  }
}

// --- Main ---
console.log("Loading MNIST data...");
const trainImages = loadImages("data/train-images-idx3-ubyte.gz");
const trainLabels = loadLabels("data/train-labels-idx1-ubyte.gz");
const testImages = loadImages("data/t10k-images-idx3-ubyte.gz");
const testLabels = loadLabels("data/t10k-labels-idx1-ubyte.gz");
console.log(
  `Loaded ${trainImages.length} training images, ${testImages.length} test images`
);

const net = new Network([784, 16, 16, 10]);

console.log("Training...");
net.train(trainImages, trainLabels, 20, 0.5, 32);

mkdirSync("public", { recursive: true });
writeFileSync("public/weights.json", JSON.stringify(net.export()));
console.log("Weights saved to public/weights.json");
