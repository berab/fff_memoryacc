#include "fff.h"
#include "arm_nnfunctions.h"

// Weights in TCM
static int8_t nw[N_NODES * IN_FEATURES] = NW;
static int32_t nb[N_NODES] = NB;
static int8_t lw1_1[N_LEAVES_TCM * LEAF_WIDTH * IN_FEATURES] = LW1_1;
static int32_t lb1_1[N_LEAVES_TCM * LEAF_WIDTH] = LB1_1;
static int8_t lw2_1[N_LEAVES_TCM * OUT_FEATURES * LEAF_WIDTH] = LW2_1;
static int32_t lb2_1[N_LEAVES_TCM * OUT_FEATURES] = LB2_1;

// Weights in SRAM
AM_SHARED_RW static int8_t lw1_2[N_LEAVES * LEAF_WIDTH * IN_FEATURES] = LW1_2;
AM_SHARED_RW static int32_t lb1_2[N_LEAVES * LEAF_WIDTH] = LB1_2;
AM_SHARED_RW static int8_t lw2_2[N_LEAVES * OUT_FEATURES * LEAF_WIDTH] = LW2_2;
AM_SHARED_RW static int32_t lb2_2[N_LEAVES * OUT_FEATURES] = LB2_2;

#ifdef SORTED
// Sorted leaf indices for memory access optimization based on leaf stats
static uint8_t li[N_LEAVES] = LI;
#endif

// To simulate the all test set samples, we use precomputed leaf target indices
static uint8_t lt[N_SAMPLES] = LT;

int8_t input[IN_FEATURES] = INPUT;
int8_t output[OUT_FEATURES];
int8_t hidden[LEAF_WIDTH];

int g_sample_index = 0;

int argmax() {
    int max_index = 0;
    int8_t max_value = output[0];
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
    cmsis_nn_context ctx = { .buf = NULL, .size = 0 };
    cmsis_nn_fc_params fc_params = FC_PARAMS;
    cmsis_nn_per_tensor_quant_params quant_params = QUANT_PARAMS; // TODO: Use actual quantization parameters
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    // ROUTING
    input_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = IN_FEATURES };
    filter_dims = (cmsis_nn_dims){ .n = IN_FEATURES, .h = 1, .w = 1, .c = 1 };
    bias_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = 1 };
    output_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = 1 };

    uint8_t n = 0;
    int8_t router_out;
    for (int i = 0; i < DEPTH; i++) {
        arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, input, 
                               &filter_dims, &nw[n * IN_FEATURES], 
                               &bias_dims, &nb[n], 
                               &output_dims, &router_out);
        n = ROUTE(n, router_out);
    }
    n = lt[g_sample_index]; // Fetch leaf id from precomputed target indices
    g_sample_index++;
    // FF
#ifdef SORTED
    n = li[n]; // Fetch leaf id from memory order
#endif
    const int8_t *lw1, *lw2;
    const int32_t *lb1, *lb2;
#ifndef SRAMMEM
    if (n <= N_LEAVES_TCM) {
        lw1 = lw1_1;
        lb1 = lb1_1;
        lw2 = lw2_1;
        lb2 = lb2_1;
    } else {
        lw1 = lw1_2;
        lb1 = lb1_2;
        lw2 = lw2_2;
        lb2 = lb2_2;
    }
#else
    lw1 = lw1_2;
    lb1 = lb1_2;
    lw2 = lw2_2;
    lb2 = lb2_2;
#endif
    input_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = IN_FEATURES };
    filter_dims = (cmsis_nn_dims){ .n = IN_FEATURES, .h = 1, .w = 1, .c = LEAF_WIDTH };
    bias_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = LEAF_WIDTH };
    output_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = LEAF_WIDTH };

    arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, input, 
                           &filter_dims, &lw1[n * LEAF_WIDTH * IN_FEATURES], 
                           &bias_dims, &lb1[n * LEAF_WIDTH], 
                           &output_dims, hidden);
    
    arm_relu_q7(hidden, LEAF_WIDTH);

    input_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = LEAF_WIDTH };
    filter_dims = (cmsis_nn_dims){ .n = LEAF_WIDTH, .h = 1, .w = 1, .c = OUT_FEATURES };
    bias_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = OUT_FEATURES };
    output_dims = (cmsis_nn_dims){ .n = 1, .h = 1, .w = 1, .c = OUT_FEATURES };
    
    arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, hidden, 
                           &filter_dims, &lw2[n * OUT_FEATURES * LEAF_WIDTH], 
                           &bias_dims, &lb2[n * OUT_FEATURES], 
                           &output_dims, output);

    arm_softmax_s8(output, 1, OUT_FEATURES, SOFTMAX_PARAMS, output);

    argmax();
}
