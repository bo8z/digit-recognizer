function relu(x) {
  return x > 0 ? x : 0;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export class NeuralNetwork {
  constructor(data) {
    this.neuronsPerLayer = data.sizes;
    this.weights = data.weights;
    this.biases = data.biases;
  }

  forward(input) {
    const activations = [input];

    for (let edgeLayer = 0; edgeLayer < this.neuronsPerLayer.length - 1; edgeLayer++) {
      const prevActivations = activations[edgeLayer];
      const edgeLayerActivations = new Array(this.neuronsPerLayer[edgeLayer + 1]);

      for (let neuron = 0; neuron < this.neuronsPerLayer[edgeLayer + 1]; neuron++) {
        let weightedSum = this.biases[edgeLayer][neuron];
        for (let input = 0; input < prevActivations.length; input++) {
          weightedSum += this.weights[edgeLayer][neuron][input] * prevActivations[input];
        }
        edgeLayerActivations[neuron] = edgeLayer < this.neuronsPerLayer.length - 2 ? relu(weightedSum) : sigmoid(weightedSum);
      }

      activations.push(edgeLayerActivations);
    }

    return { activations };
  }
}
