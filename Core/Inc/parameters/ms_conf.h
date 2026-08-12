
#ifndef MS_CONF_H
#define MS_CONF_H

#include "ms_weights.h"
#include "ms_leafstats.h"

#define DEPTH 4
#define LEAF_WIDTH 4
#define N_LEAVES (1 << DEPTH)
#define N_NODES (N_LEAVES - 1)
#define N_LEAVES_SRAM 6

#endif // MS_CONF_H
            