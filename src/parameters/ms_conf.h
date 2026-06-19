#include "ms_weights.h"
#include "ms_leafstats.h"

#define DEPTH 4
#define LEAF_WIDTH 4
#define N_LEAVES (1 << DEPTH)
#define N_NODES (N_LEAVES - 1)
#define N_LEAVES_TCM 5
