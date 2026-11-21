/*
 * tests/parity_host.cpp — host-side parity test for siren_features.h
 *
 * Compile:  g++ -std=c++11 -I arduino/signal_controller tests/parity_host.cpp -o parity_host
 * Run:      ./parity_host adc_data.csv
 *
 * Input CSV: one line per test case, comma-separated ADC values (integers 0..1023).
 * Output: one line per extracted chunk: pmin,pmax,pmean,sweep,lvl
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "siren_features.h"

static void reset_state() {
    memset(&_ss, 0, sizeof(_ss));
    _ss.dc16  = 512 << DC_SHIFT_BITS;
    _ss.state = -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: parity_host adc_data.csv\n");
        return 1;
    }

    FILE* fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 2;
    }

    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        reset_state();
        char* tok = strtok(line, ",\n");
        while (tok) {
            int16_t sample = (int16_t)atoi(tok);
            if (siren_feed(sample)) {
                printf("%d,%d,%d,%d,%d\n",
                    (int)_ss.out.pmin, (int)_ss.out.pmax,
                    (int)_ss.out.pmean,(int)_ss.out.sweep,
                    (int)_ss.out.lvl);
            }
            tok = strtok(NULL, ",\n");
        }
    }
    fclose(fp);
    return 0;
}
