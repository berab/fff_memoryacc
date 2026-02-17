#include "fff.h"

// Weights
static float nw[N_NODES * IN_FEATURES] = NW;
static float nb[N_NODES] = NB;
static float lw1[N_LEAVES * LEAF_WIDTH * IN_FEATURES] = LW1;
static float lb1[N_LEAVES * LEAF_WIDTH] = LB1;
static float lw2[N_LEAVES * OUT_FEATURES * LEAF_WIDTH] = LW2;
static float lb2[N_LEAVES * OUT_FEATURES] = LB2;
#ifdef SORTED
// Sorted leaf indices for memory access optimization based on leaf stats
static uint8_t li[N_LEAVES] = LI;
#endif


float input[IN_FEATURES];
float output[OUT_FEATURES];
float hidden[LEAF_WIDTH];

int argmax(float output[]) {
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

void fff(float input[], float output[]) {
    // ROUTING
    int n = 0;
    for (int i = 0; i < DEPTH; i++) {
        n = ROUTE(n, neuron(&nw[n * IN_FEATURES], nb[n], input, IN_FEATURES));
    }
    n -= N_NODES; // Convert node id to leaf id
                
    // FF
#ifdef SORTED
    n = li[n]; // Fetch leaf id from memory order
#endif
    float h;
    for (int i = 0; i < LEAF_WIDTH; i++) {
        h = neuron(&lw1[(n * LEAF_WIDTH + i) * IN_FEATURES], lb1[n * LEAF_WIDTH + i], input, IN_FEATURES);
        hidden[i] = RELU(h);
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(&lw2[(n * OUT_FEATURES + i) * LEAF_WIDTH], lb2[n * OUT_FEATURES + i], hidden, LEAF_WIDTH);
    }
    argmax(output);
}
