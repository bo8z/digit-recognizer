import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { gunzipSync } from "zlib";

// This version is organized like a tiny neural-network framework:
// data -> model.forward(input) -> loss -> loss.backward() -> optimizer.step()
// The model is a Sequential stack of modules, so Dense/ReLU/Sigmoid could be
// swapped for other differentiable modules later.

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

function randn() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// --- Framework pieces ---
class Module {
    forward() {
        throw new Error("Module.forward(input) must be implemented by subclasses.");
    }

    parameters() {
        return [];
    }
}

class Sequential extends Module {
    constructor(modules) {
        super();
        this.modules = modules;
    }

    forward(input) {
        let output = input;
        for (const module of this.modules) {
            output = module.forward(output);
        }
        return output;
    }

    parameters() {
        return this.modules.flatMap((module) => module.parameters());
    }

    denseLayers() {
        return this.modules.filter((module) => module instanceof Dense);
    }
}

class Dense extends Module {
    constructor(inputSize, outputSize) {
        super();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = [];
        this.biases = [];

        const scale = Math.sqrt(2 / inputSize);
        for (let neuron = 0; neuron < outputSize; neuron++) {
            const neuronWeights = [];
            for (let sourceNeuron = 0; sourceNeuron < inputSize; sourceNeuron++) {
                neuronWeights.push(new Value(randn() * scale));
            }
            this.weights.push(neuronWeights);
            this.biases.push(new Value(0));
        }
    }

    forward(input) {
        const activations = [];

        for (let neuron = 0; neuron < this.outputSize; neuron++) {
            let weightedSum = this.biases[neuron];

            for (let sourceNeuron = 0; sourceNeuron < this.inputSize; sourceNeuron++) {
                weightedSum = weightedSum.add(
                    this.weights[neuron][sourceNeuron].mul(input[sourceNeuron])
                );
            }

            activations.push(weightedSum);
        }

        return activations;
    }

    parameters() {
        return [
            ...this.weights.flat(),
            ...this.biases,
        ];
    }
}

class ReLU extends Module {
    forward(input) {
        return input.map((value) => value.relu());
    }
}

class Sigmoid extends Module {
    forward(input) {
        return input.map((value) => value.sigmoid());
    }
}

class MeanSquaredErrorLoss {
    forward(prediction, target) {
        let loss = new Value(0);

        for (let i = 0; i < prediction.length; i++) {
            const error = prediction[i].sub(target[i]);
            loss = loss.add(error.pow(2).mul(0.5));
        }

        return loss;
    }
}

class SGD {
    constructor(parameters, learningRate) {
        this.parameters = parameters;
        this.learningRate = learningRate;
    }

    zeroGradients() {
        for (const parameter of this.parameters) parameter.grad = 0;
    }

    step(batchSize) {
        const scaledLearningRate = this.learningRate / batchSize;
        for (const parameter of this.parameters) {
            parameter.data -= scaledLearningRate * parameter.grad;
        }
    }
}

class Trainer {
    constructor(model, lossFunction, optimizer) {
        this.model = model;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
    }

    train(trainingExamples, validationExamples, epochs, batchSize) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            this.trainEpoch(trainingExamples, epoch, epochs, batchSize);

            const acc = this.evaluate(validationExamples);
            console.log(
                `Epoch ${epoch + 1}/${epochs} — accuracy: ${(acc * 100).toFixed(1)}%`
            );
        }
    }

    trainEpoch(trainingExamples, epoch, epochs, batchSize) {
        const indices = shuffledIndices(trainingExamples.length);
        const totalBatches = Math.ceil(trainingExamples.length / batchSize);

        for (let batchStart = 0; batchStart < trainingExamples.length; batchStart += batchSize) {
            const batchLoss = this.trainBatch(trainingExamples, indices, batchStart, batchSize);

            /* For logging */
            if (batchStart % (batchSize * 50) === 0) {
                const batchNumber = Math.floor(batchStart / batchSize) + 1;
                console.log(
                    `epoch ${epoch + 1}/${epochs} batch ${batchNumber}/${totalBatches} loss ${batchLoss.toFixed(4)}`
                );
            }
        }
    }

    trainBatch(trainingExamples, indices, batchStart, batchSize) {
        const batchEnd = Math.min(batchStart + batchSize, trainingExamples.length);
        const actualBatchSize = batchEnd - batchStart;
        let batchLoss = 0;

        this.optimizer.zeroGradients();

        for (let batchIndex = batchStart; batchIndex < batchEnd; batchIndex++) {
            const exampleIndex = indices[batchIndex];
            batchLoss += this.trainExample(trainingExamples[exampleIndex]);
        }

        this.optimizer.step(actualBatchSize);

        return batchLoss / actualBatchSize;
    }

    trainExample(example) {
        const prediction = this.model.forward(example.input);
        const loss = this.lossFunction.forward(prediction, example.target);
        loss.backward();
        return loss.data;
    }

    evaluate(validationExamples) {
        let correct = 0;

        for (const example of validationExamples) {
            const prediction = this.model.forward(example.input);
            const scores = prediction.map((value) => value.data);
            const predictedLabel = scores.indexOf(Math.max(...scores));
            const expectedLabel = example.target.indexOf(Math.max(...example.target));
            if (predictedLabel === expectedLabel) correct++;
        }

        return correct / validationExamples.length;
    }
}

function buildDenseClassifier(sizes) {
    const modules = [];

    for (let i = 0; i < sizes.length - 1; i++) {
        modules.push(new Dense(sizes[i], sizes[i + 1]));
        modules.push(i === sizes.length - 2 ? new Sigmoid() : new ReLU());
    }

    return new Sequential(modules);
}

function exportDenseClassifier(model, sizes) {
    const denseLayers = model.denseLayers();

    return {
        sizes,
        weights: denseLayers.map((layer) =>
            layer.weights.map((row) => row.map((weight) => weight.data))
        ),
        biases: denseLayers.map((layer) => layer.biases.map((bias) => bias.data)),
    };
}

function oneHot(label, size) {
    const target = new Array(size).fill(0);
    target[label] = 1;
    return target;
}

function makeClassificationExamples(inputs, labels, classCount) {
    return inputs.map((input, index) => ({
        input,
        target: oneHot(labels[index], classCount),
    }));
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
const classCount = 10;
const trainingExamples = makeClassificationExamples(trainImages, trainLabels, classCount);
const validationExamples = makeClassificationExamples(testImages, testLabels, classCount);

console.log(
    `Loaded ${trainImages.length} training images, ${testImages.length} test images`
);

const sizes = (process.env.SIZES ?? "784,16,16,10")
    .split(",")
    .map((value) => Number(value.trim()));
const epochs = Number(process.env.EPOCHS ?? 20);
const learningRate = Number(process.env.LEARNING_RATE ?? 0.5);
const batchSize = Number(process.env.BATCH_SIZE ?? 32);
const outputPath = process.env.OUTPUT_PATH ?? "public/weights-framework-style.json";

const model = buildDenseClassifier(sizes);
const lossFunction = new MeanSquaredErrorLoss();
const optimizer = new SGD(model.parameters(), learningRate);
const trainer = new Trainer(model, lossFunction, optimizer);

console.log("Training with framework-style scalar autograd...");
trainer.train(trainingExamples, validationExamples, epochs, batchSize);

mkdirSync("public", { recursive: true });
writeFileSync(outputPath, JSON.stringify(exportDenseClassifier(model, sizes)));
console.log(`Weights saved to ${outputPath}`);
