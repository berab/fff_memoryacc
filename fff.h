#include <stdint.h>
#include "datasets/mnist.h"
#ifdef SORTED
#include "models/mnist_d4_l16_sorted.h"
#else
#include "models/mnist_d4_l16.h"
#endif

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

extern float input[IN_FEATURES];
extern float output[OUT_FEATURES];

int argmax();
float neuron(float weights[], float bias, float input[], int dim);
void fff();
