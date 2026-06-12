// #include <string.h>
#include "mem.h"
#include "fff.h"

// Routers and some leaves in SRAM
SRAM float nw[N_NODES * IN_FEATURES] = NW;
SRAM float nb[N_NODES] = NB;
SRAM float lw1_1[N_LEAVES_SRAM * LEAF_WIDTH * IN_FEATURES] = LW1_1;
SRAM float lb1_1[N_LEAVES_SRAM * LEAF_WIDTH] = LB1_1;
SRAM float lw2_1[N_LEAVES_SRAM * OUT_FEATURES * LEAF_WIDTH] = LW2_1;
SRAM float lb2_1[N_LEAVES_SRAM * OUT_FEATURES] = LB2_1;

// All leaves in FLASH
const float lw1_2[N_LEAVES * LEAF_WIDTH * IN_FEATURES] = LW1_2;
const float lb1_2[N_LEAVES * LEAF_WIDTH] = LB1_2;
const float lw2_2[N_LEAVES * OUT_FEATURES * LEAF_WIDTH] = LW2_2;
const float lb2_2[N_LEAVES * OUT_FEATURES] = LB2_2;

#ifdef SORTED
// Sorted leaf indices for memory access optimization based on leaf stats
const uint8_t li[N_LEAVES] = LI;
#endif
#ifdef MEMCHECK
volatile uint32_t g_TCMCount = 0;
volatile uint32_t g_RAMCount = 0;
#endif

// To simulate the all test set samples, we use precomputed leaf target indices
const uint8_t lt[N_SAMPLES] = LT;

float input[IN_FEATURES] = INPUT;
float output[OUT_FEATURES];
float hidden[LEAF_WIDTH];

int g_sample_index = 0;

int argmax() {
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

void fff() {
    // ROUTING
    uint8_t n = 0;
    for (int i = 0; i < DEPTH; i++) {
        n = ROUTE(n, neuron(&nw[n * IN_FEATURES], nb[n], input, IN_FEATURES)); //TOOD: Check if MACRO hurts. idk thsi apollo is weird somtimes
    }
    // n -= N_NODES; // Convert node id to leaf id
    n = lt[g_sample_index]; // Fetch leaf id from precomputed target indices
    // n = li[n]; // Fetch leaf id from memory order
    g_sample_index++;
    // FF
#ifdef SORTED
#endif

    // float *lw1, *lb1, *lw2, *lb2;
    // if (n >= N_LEAVES_SRAM) {
    //     lw1 = lw1_1;
    //     lb1 = lb1_1;
    //     lw2 = lw2_1;
    //     lb2 = lb2_1;
    //     n -= N_LEAVES_SRAM; // Convert leaf id to SRAM index
    // } else {
    //     lw1 = lw1_2;
    //     lb1 = lb1_2;
    //     lw2 = lw2_2;
    //     lb2 = lb2_2;
    // }
    // lw1 = lw1_2;
    // lb1 = lb1_2;
    // lw2 = lw2_2;
    // lb2 = lb2_2;

    float h;
    for (int i = 0; i < LEAF_WIDTH; i++) {
        h = neuron(&lw1_2[(n * LEAF_WIDTH + i) * IN_FEATURES], lb1_2[n * LEAF_WIDTH + i], input, IN_FEATURES);
        hidden[i] = RELU(h);
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(&lw2_2[(n * OUT_FEATURES + i) * LEAF_WIDTH], lb2_2[n * OUT_FEATURES + i], hidden, LEAF_WIDTH);
    }
    argmax();
}
