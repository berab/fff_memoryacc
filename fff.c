#include "Datasets/mnist.h"
#include "Models/mnist.h"

#define ReLU(x) ((x) > 0 ? (x) : 0)

float neuron(float weights[], float bias, float input[], int dim) {
		float accumulator = 0.0;

		for (int i = 0; i < dim; i++) {
				accumulator += weights[i] * input[i];
		}
		return accumulator + bias;
}

void fff(void) {
    int n = 0;
    float n_out = 0.0;
    for (int i = 0; i < DEPTH; i++) {
        n_out = neuron(nw[n], nb[n], input, IN_FEATURES);
        n = (n_out > 0) ? (n*2) : (n*2 + 1); // TODO: Fix!
    }
    // Fetch leaf id
    n = idx[n];
    // Layer 1
    for (int i = 0; i < LEAF_WIDTH; i++) {
        hidden[i] = ReLU(neuron(w1[n][i], b1[n][i], input, IN_FEATURES));
    }
    // Layer 2
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(w2[n][i], b2[n][i], hidden, 
                LEAF_WIDTH);
    }
}

int main(void) {
    for (int i = 0; i < 1000; i++) {
        fff();
    }
}
