#include <stdlib.h>
#include <stdio.h>

#include "fff.h"

int main(void) {
    FILE *f = fopen(FILENAME, "rb");
    for (int r = 0; r < N_SAMPLES; ++r) {
        fread(input, sizeof(float), IN_FEATURES, f);

        fff(input, output);
    }
    fclose(f);
}
