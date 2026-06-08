#include <string.h>
#include "mem.h"
#include "fff.h"

// Routers and some leaves in SRAM
SRAM float nw[N_NODES * IN_FEATURES] = NW;
SRAM float nb[N_NODES] = NB;
SRAM float lw1_r[N_LEAVES_SRAM * LEAF_WIDTH * IN_FEATURES] = LW1_R;
SRAM float lb1_r[N_LEAVES_SRAM * LEAF_WIDTH] = LB1_R;
SRAM float lw2_r[N_LEAVES_SRAM * OUT_FEATURES * LEAF_WIDTH] = LW2_R;
SRAM float lb2_r[N_LEAVES_SRAM * OUT_FEATURES] = LB2_R;

// All leaves in FLASH
const float lw1_f[N_LEAVES * LEAF_WIDTH * IN_FEATURES] = LW1_F;
const float lb1_f[N_LEAVES * LEAF_WIDTH] = LB1_F;
const float lw2_f[N_LEAVES * OUT_FEATURES * LEAF_WIDTH] = LW2_F;
const float lb2_f[N_LEAVES * OUT_FEATURES] = LB2_F;

#ifdef SORTED
// Sorted leaf indices for memory access optimization based on leaf stats
static uint8_t li[N_LEAVES] = LI;
#endif
#ifdef MEMCHECK
volatile uint32_t g_TCMCount = 0;
volatile uint32_t g_RAMCount = 0;
#endif

// To simulate the all test set samples, we use precomputed leaf target indices
static uint8_t lt[N_SAMPLES] = LT;

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
    n = li[n]; // Fetch leaf id from memory order
    g_sample_index++;
    // FF
#ifdef SORTED
#endif

    // load from flash 
    // memcpy(lw1, lw1_2, 
    //         4 * sizeof(float));
    
    float *lw1, *lb1, *lw2, *lb2;
    if (n >= N_LEAVES_SRAM) {
        lw1 = lw1_r;
        lb1 = lb1_r;
        lw2 = lw2_r;
        lb2 = lb2_r;
        n -= N_LEAVES_SRAM; // Convert leaf id to SRAM index
    } else {
        lw1 = lw1_f;
        lb1 = lb1_f;
        lw2 = lw2_f;
        lb2 = lb2_f;
    }

    float h;
    for (int i = 0; i < LEAF_WIDTH; i++) {
        h = neuron(&lw1[(n * LEAF_WIDTH + i) * IN_FEATURES], lb1[n * LEAF_WIDTH + i], input, IN_FEATURES);
        hidden[i] = RELU(h);
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(&lw2[(n * OUT_FEATURES + i) * LEAF_WIDTH], lb2[n * OUT_FEATURES + i], hidden, LEAF_WIDTH);
    }
    argmax();
}
