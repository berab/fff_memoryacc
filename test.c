#include <stdlib.h>
#include <stdio.h>

#include "datasets/mnist.h"
#include "models/mnist_d4_l16_test.h"

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

// Weights
static float nw[N_NODES * IN_FEATURES] = NW;
static float nb[N_NODES] = NB;
static float lw1[N_LEAVES * LEAF_WIDTH * IN_FEATURES] = LW1;
static float lb1[N_LEAVES * LEAF_WIDTH] = LB1;
static float lw2[N_LEAVES * OUT_FEATURES * LEAF_WIDTH] = LW2;
static float lb2[N_LEAVES * OUT_FEATURES] = LB2;

float input[IN_FEATURES];
float hidden[LEAF_WIDTH];
float output[OUT_FEATURES];
int targets[N_SAMPLES];
int preds[N_SAMPLES];


int argmax(void) {
    int max_idx = 0;
    float max_val = output[0];
    for (int i = 1; i < OUT_FEATURES; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    return max_idx;
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
        n = ROUTE(n, neuron(&nw[n * IN_FEATURES], nb[n], input, IN_FEATURES));
    }
    n -= N_NODES; // Convert node id to leaf id
                
    // FF
    // n = idx[n]; // TODO: Fetch leaf id from memory order
    float h;
    for (int i = 0; i < LEAF_WIDTH; i++) {
        h = neuron(&lw1[(n * LEAF_WIDTH + i) * IN_FEATURES], 
                lb1[n * LEAF_WIDTH + i], input, IN_FEATURES);
        hidden[i] = RELU(h);
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(&lw2[(n * OUT_FEATURES + i) * LEAF_WIDTH], 
                lb2[n * OUT_FEATURES + i], hidden, LEAF_WIDTH);
    }
    return argmax();
}

float calculate_acc(int preds[], int targets[]) {
    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        if (preds[i] == targets[i]) {
            correct++;
        }
    }
    return (float)correct / N_SAMPLES;
}

void print_array(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%.4f ", arr[i]);
    }
    printf("\n");
}

float mean(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

int main(void) {

    FILE *f_t = fopen(TARGETS_FILENAME, "rb");
    fread(targets, sizeof(float), N_SAMPLES, f_t);
    fclose(f_t);

    FILE *f = fopen(FILENAME, "rb");
    for (int i = 0; i < N_SAMPLES; ++i) {
        fread(input, sizeof(float), IN_FEATURES, f);
        fff();
        preds[i] = fff();
    }
    fclose(f);

    float c_acc = calculate_acc(preds, targets);
    float torch_acc = ACC;
    float diff = c_acc - torch_acc;  // Difference as decimal

    printf("C accuracy:     %.4f%%\n", c_acc * 100.0);  
    printf("Torch accuracy: %.4f%%\n", torch_acc * 100.0);
    printf("Difference:     %.4f%%\n", diff * 100.0);

    // Test assertion with threshold (using absolute value)
    float abs_diff = diff < 0 ? -diff : diff;
    float threshold = 0.01;  // 0.01 as decimal (1%)

    if (abs_diff > threshold) {
        printf("\nTEST FAILED: Accuracy difference %.4f exceeds threshold of 0.01\n", abs_diff);
        return 1;  // Return error code
    }

    printf("\nTEST PASSED: Accuracy within 0.01 threshold\n");
    return 0;  // Success
}
