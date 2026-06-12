#include <stdint.h>
// #include <am_hal_global.h> // Includes sram section

#include "mnist.h"
#include "mnist_a08_h8_d4_l4.h"
// #include "mnist_a08_h05_d4_l4.h"

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

int argmax();
float neuron(float weights[], float bias, float input[], int dim);
void fff();
