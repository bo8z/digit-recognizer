
export class NeuralNetwork {


    constructor(data) {

        this.neuronsPerLayer = data.sizes;
        this.weights = data.weights;
        this.biases = data.biases;
    }


    foward(input) {

        const activations = [input];

        // This iterates over the actual 
        for(let layer = 0; layer < this.neuronsPerLayer - 1; layer++) {


        }
    }

}