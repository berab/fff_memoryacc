
#ifndef FMNIST_CONF_H
#define FMNIST_CONF_H

#include "fmnist_weights.h"
#include "fmnist_leafstats.h"

#define DEPTH 4
#define LEAF_WIDTH 4
#define N_LEAVES (1 << DEPTH)
#define N_NODES (N_LEAVES - 1)
#define N_LEAVES_SRAM 4

#endif // FMNIST_CONF_H
            