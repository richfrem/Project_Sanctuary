#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>

static uint64_t *N;
static uint64_t *prefix;
static uint64_t *hist;
static bool *sumset;
static int H, K, QMAX;

static inline uint64_t binomial_coefficient(int n, int k) {
    if (k > n || k < 0) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    uint64_t result = 1;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

static bool parse_args(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s H K QMAX\n", argv[0]);
        return false;
    }

    char *endptr;
    errno = 0;
    long temp_H = strtol(argv[1], &endptr, 10);
    if (errno || *endptr != '\0' || temp_H < 2 || temp_H > 16) {
        fprintf(stderr, "Invalid H: %s\n", argv[1]);
        return false;
    }
    H = (int)temp_H;

    errno = 0;
    long temp_K = strtol(argv[2], &endptr, 10);
    if (errno || *endptr != '\0' || (temp_K != 3 && temp_K != 4)) {
        fprintf(stderr, "Invalid K: %s\n", argv[2]);
        return false;
    }
    K = (int)temp_K;

    errno = 0;
    long temp_QMAX = strtol(argv[3], &endptr, 10);
    if (errno || *endptr != '\0' || temp_QMAX < K || temp_QMAX > 250) {
        fprintf(stderr, "Invalid QMAX: %s\n", argv[3]);
        return false;
    }
    QMAX = (int)temp_QMAX;

    return true;
}

static void enumerate_shapes() {
    int max_card = binomial_coefficient(H + K - 1, K - 1);
    int max_sum = H * (QMAX - 1);

    N = calloc((size_t)QMAX * (size_t)(max_card + 1), sizeof(uint64_t));
    prefix = calloc((size_t)(max_card + 1), sizeof(uint64_t));
    hist = calloc((size_t)(max_card + 1), sizeof(uint64_t));
    sumset = calloc((size_t)(max_sum + 1), sizeof(bool));

    if (!N || !prefix || !hist || !sumset) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    if (K == 3) {
        for (int v = 2; v < QMAX; ++v) {
            for (int u = 1; u < v; ++u) {
                memset(sumset, 0, (size_t)(max_sum + 1) * sizeof(bool));
                int count = 0;
                for (int a = 0; a <= H; ++a) {
                    for (int b = 0; b <= H - a; ++b) {
                        int c = H - a - b;
                        if (c < 0) continue;
                        int sum = a * 0 + b * u + c * v;
                        if (!sumset[sum]) {
                            sumset[sum] = true;
                            count++;
                        }
                    }
                }
                N[v * (max_card + 1) + count]++;
            }
        }
    } else if (K == 4) {
        for (int w = 3; w < QMAX; ++w) {
            for (int v = 2; v < w; ++v) {
                for (int u = 1; u < v; ++u) {
                    memset(sumset, 0, (size_t)(max_sum + 1) * sizeof(bool));
                    int count = 0;
                    for (int a = 0; a <= H; ++a) {
                        for (int b = 0; b <= H - a; ++b) {
                            for (int c = 0; c <= H - a - b; ++c) {
                                int d = H - a - b - c;
                                if (d < 0) continue;
                                int sum = a * 0 + b * u + c * v + d * w;
                                if (!sumset[sum]) {
                                    sumset[sum] = true;
                                    count++;
                                }
                            }
                        }
                    }
                    N[w * (max_card + 1) + count]++;
                }
            }
        }
    }

    printf("q,size,count\n");
    for (int q = K; q <= QMAX; ++q) {
        for (int size = 0; size <= max_card; ++size) {
            prefix[size] += N[(q-1) * (max_card + 1) + size];
        }
        for (int size = 0; size <= max_card; ++size) {
            hist[size] = prefix[size];
        }
        for (int size = 1; size <= max_card; ++size) {
            if (hist[size] > 0) {
                printf("%d,%d,%" PRIu64 "\n", q, size, hist[size]);
            }
        }
    }

    free(N);
    free(prefix);
    free(hist);
    free(sumset);
}

int main(int argc, char *argv[]) {
    if (!parse_args(argc, argv)) {
        return EXIT_FAILURE;
    }

    enumerate_shapes();
    return 0;
}
