#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define IN_FEATURES 784
#define NUM_FILES 2
#define N_SAMPLES 10000
#define FILENAME "bins/mnist.bin"

int main(void) {

    FILE *f = fopen(FILENAME, "rb");
    float *row = malloc(IN_FEATURES * sizeof(float));
    
    for (int r = 0; r < N_SAMPLES; ++r) {
        // Read one sample at a time
        fread(row, sizeof(float), IN_FEATURES, f);
        MODEL(row);
    }
    
    printf("Finished reading %s\n", FILENAME);
    
    free(row);
    fclose(f);
}
