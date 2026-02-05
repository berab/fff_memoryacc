#include <stdlib.h>
#include <stdio.h>

#include "bins/mnist.h"
#include "Models/mnist_d4_l16.h"

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define ReLU(x) ((x) > 0 ? (x) : 0)

int argmax(void) {
    int max_index = 0;
    float max_value = output[0];
    for (int i = 1; i < OUT_FEATURES; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }
    return max_index;
}

float neuron(float weights[], float bias, float input[], int dim) {
		float accumulator = 0.0;

		for (int i = 0; i < dim; i++) {
				accumulator += weights[i] * input[i];
		}
		return accumulator + bias;
}

int fff(void) {
    // ROUTING
    int n = 0;
    for (int i = 0; i < DEPTH; i++) {
        n = ROUTE(n, neuron(nw[n], nb[n], input, IN_FEATURES));
    }
    n -= N_NODES; // Convert node id to leaf id
                
    // FF
    // n = idx[n]; // Fetch leaf id from memory order
    for (int i = 0; i < LEAF_WIDTH; i++) {
        hidden[i] = ReLU(neuron(w1[n][i], b1[n][i], input, IN_FEATURES));
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(w2[n][i], b2[n][i], hidden, LEAF_WIDTH);
    }
    return argmax();
}

int main(void) {

    FILE *f = fopen(FILENAME, "rb");
    int target;
    for (int r = 0; r < N_SAMPLES; ++r) {
        fread(input, sizeof(float), IN_FEATURES, f);

        target = fff();
        printf("%d ", target);
    }
    fclose(f);
}
