export interface NetworkData {
  sizes: number[];
  weights: number[][][];
  biases: number[][];
}

export interface ForwardResult {
  activations: number[][];
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class NeuralNetwork {
  sizes: number[];
  weights: number[][][];
  biases: number[][];

  constructor(data: NetworkData) {
    this.sizes = data.sizes;
    this.weights = data.weights;
    this.biases = data.biases;
  }

  forward(input: number[]): ForwardResult {
    const activations: number[][] = [input];
    const numLayers = this.sizes.length;

    for (let l = 0; l < numLayers - 1; l++) {
      const prev = activations[l];
      const size = this.sizes[l + 1];
      const a: number[] = new Array(size);

      for (let j = 0; j < size; j++) {
        let sum = this.biases[l][j];
        for (let k = 0; k < prev.length; k++) {
          sum += this.weights[l][j][k] * prev[k];
        }
        a[j] = l < numLayers - 2 ? relu(sum) : sigmoid(sum);
      }
      activations.push(a);
    }

    return { activations };
  }
}
