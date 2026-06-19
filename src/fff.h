#include <stdint.h>
#include <am_hal_global.h> // Includes sram section

#ifdef MNIST
#include "mnist.h"
#include "mnist_conf.h"
#endif
#ifdef SC
#include "sc.h"
#include "sc_conf.h"
#endif
#ifdef MS
#include "ms.h"
#include "ms_conf.h"
#endif

#define ROUTE(n, p) (p >= 0) ? (2 * n + 2) : (2 * n + 1)
#define RELU(x) ((x) > 0 ? (x) : 0)

int argmax();
float neuron(float weights[], float bias, float input[], int dim);
void fff();
