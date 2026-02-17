#include <stdint.h>
#include "datasets/mnist.h"
#ifdef SORTED
#include "models/mnist_d4_l16_sorted.h"
#else
#include "models/mnist_d4_l16.h"
#endif

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

// Weights
static float nw[N_NODES * IN_FEATURES];
static float nb[N_NODES];
static float lw1[N_LEAVES * LEAF_WIDTH * IN_FEATURES];
static float lb1[N_LEAVES * LEAF_WIDTH];
static float lw2[N_LEAVES * OUT_FEATURES * LEAF_WIDTH];
static float lb2[N_LEAVES * OUT_FEATURES];

#ifdef SORTED
// Sorted leaf indices for memory access optimization based on leaf stats
static uint8_t li[N_LEAVES];
#endif

extern float input[IN_FEATURES];
extern float output[OUT_FEATURES];

int argmax(float output[]);
float neuron(float weights[], float bias, float input[], int dim);
int fff(float input[], float output[]);
