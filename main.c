#include <stdlib.h>
#include <stdio.h>

#include "fff.h"


int main(void) {
    FILE *f = fopen(FILENAME, "rb");
    int target;
    for (int r = 0; r < N_SAMPLES; ++r) {
        fread(input, sizeof(float), IN_FEATURES, f);

        target = fff(input, output);
        printf("%d ", target);
    }
    fclose(f);
}
