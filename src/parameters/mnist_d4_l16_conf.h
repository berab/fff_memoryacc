#define DEPTH 4
#define LEAF_WIDTH 16
#define N_LEAVES (1 << DEPTH)
#define N_NODES (N_LEAVES - 1)
#define N_LEAVES_TCM 6 
#define N_LEAVES_SRAM (N_LEAVES - N_LEAVES_TCM)
