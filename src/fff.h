#include <stdint.h>

#include "mnist.h"
#ifdef SORTED
#include "mnist_d4_l16_sorted.h"
#else
#include "mnist_d4_l16.h"
#endif

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

int argmax();
float neuron(float weights[], float bias, float input[], int dim);
void fff();
