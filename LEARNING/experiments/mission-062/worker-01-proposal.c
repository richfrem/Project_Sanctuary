#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdbool.h>

#define MAX_H 16
#define MAX_K 4
#define MAX_Q 250

static uint64_t *histogram;
static uint64_t *counts;
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

static void compute_histograms() {
    int max_sum = H * (QMAX - 1);
    int max_size = binomial_coefficient(H + K - 1, K - 1);
    histogram = calloc(max_size + 1, sizeof(uint64_t));
    counts = calloc(QMAX + 1, sizeof(uint64_t));
    sumset = calloc(max_sum + 1, sizeof(bool));

    if (!histogram || !counts || !sumset) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // For K=3
    if (K == 3) {
        for (int q = 3; q <= QMAX; ++q) {
            for (int v = 2; v < q; ++v) {
                for (int u = 1; u < v; ++u) {
                    int max_size = binomial_coefficient(H + 2, 2);
                    memset(sumset, 0, (max_sum + 1) * sizeof(bool));
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
                    histogram[count]++;
                }
            }
        }
    }

    // For K=4
    if (K == 4) {
        for (int q = 4; q <= QMAX; ++q) {
            for (int w = 3; w < q; ++w) {
                for (int v = 2; v < w; ++v) {
                    for (int u = 1; u < v; ++u) {
                        int max_size = binomial_coefficient(H + 3, 3);
                        memset(sumset, 0, (max_sum + 1) * sizeof(bool));
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
                        histogram[count]++;
                    }
                }
            }
        }
    }

    printf("q,size,count\n");
    for (int size = 1; size <= max_size; ++size) {
        if (histogram[size] > 0) {
            printf("%d,%d,%" PRIu64 "\n", QMAX, size, histogram[size]);
        }
    }

    free(histogram);
    free(counts);
    free(sumset);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s H K QMAX\n", argv[0]);
        return EXIT_FAILURE;
    }

    H = atoi(argv[1]);
    K = atoi(argv[2]);
    QMAX = atoi(argv[3]);

    if (H < 2 || H > MAX_H || K < 3 || K > MAX_K || QMAX < K || QMAX > MAX_Q) {
        fprintf(stderr, "Invalid input parameters\n");
        return EXIT_FAILURE;
    }

    compute_histograms();
    return 0;
}
