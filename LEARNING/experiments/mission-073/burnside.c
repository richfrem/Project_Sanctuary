/* burnside.c - MISSION-073 exact engine.
 *
 * For the n-cube Q_n (n=3,4) and every element g of its full symmetry group (signed
 * permutations, n!*2^n elements), enumerate every edge subset fixed by g (all unions of
 * g's edge cycles, Gray code) and tally, by axis-span mask A, the subsets that are
 * non-empty, not the complete edge set, and connected. One JSON line per group element.
 * Burnside aggregation (rotations only / all symmetries) is done in aggregate.py.
 *
 * Usage:  burnside N [THREADS]          -> JSON lines on stdout
 *         burnside N --props MASKFILE   -> "mask connected span" per hex mask (differential test)
 * Build:  cc -O2 -pthread -o burnside burnside.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

static int n, NV, NE;
static int ea[32], eb[32], edir[32];
static uint32_t FULL;

static void build_edges(void) {
    NV = 1 << n; NE = n * (NV / 2);
    int k = 0;
    for (int d = 0; d < n; d++)
        for (int v = 0; v < NV; v++)
            if (!((v >> d) & 1)) { ea[k] = v; eb[k] = v | (1 << d); edir[k] = d; k++; }
    FULL = (NE == 32) ? 0xFFFFFFFFu : ((1u << NE) - 1u);
}

/* returns 1 if non-empty and connected; span receives axis mask */
static inline int prop(uint32_t m, int *span) {
    if (!m) return 0;
    uint32_t touched = 0; int sp = 0;
    uint32_t t = m;
    while (t) { int e = __builtin_ctz(t); t &= t - 1; touched |= (1u << ea[e]) | (1u << eb[e]); sp |= 1 << edir[e]; }
    *span = sp;
    uint32_t R = touched & (~touched + 1u);
    int changed = 1;
    while (changed) {
        changed = 0;
        t = m;
        while (t) {
            int e = __builtin_ctz(t); t &= t - 1;
            uint32_t a = 1u << ea[e], b = 1u << eb[e];
            if (((R & a) != 0) != ((R & b) != 0)) { R |= a | b; changed = 1; }
        }
    }
    return R == touched;
}

typedef struct { int ncyc; uint32_t cyc[32]; } Perm;

static int fact(int k) { return k <= 1 ? 1 : k * fact(k - 1); }

static int vmap(int v, const int *pi, int s) {
    int w = 0;
    for (int d = 0; d < n; d++) if ((v >> d) & 1) w |= 1 << pi[d];
    return w ^ s;
}

static int edge_index(int a, int b) {
    for (int e = 0; e < NE; e++) if ((ea[e] == a && eb[e] == b) || (ea[e] == b && eb[e] == a)) return e;
    return -1;
}

typedef struct { Perm *p; int tbits; uint64_t pref; uint64_t cnt[16]; } Job;

static void *worker(void *arg) {
    Job *j = (Job *)arg; Perm *p = j->p;
    int free_c = p->ncyc - j->tbits;
    uint32_t base = 0;
    for (int i = 0; i < j->tbits; i++) if ((j->pref >> i) & 1) base |= p->cyc[free_c + i];
    uint32_t m = base; int sp;
    uint64_t total = 1ull << free_c;
    for (uint64_t g = 0;; ) {
        if (m != FULL && prop(m, &sp)) j->cnt[sp]++;
        g++;
        if (g >= total) break;
        m ^= p->cyc[__builtin_ctzll(g)];
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    n = atoi(argv[1]); if (n < 2 || n > 4) return 2;
    build_edges();
    if (argc >= 4 && !strcmp(argv[2], "--props")) {
        FILE *f = fopen(argv[3], "r"); char line[64];
        while (fgets(line, sizeof line, f)) {
            uint32_t m = (uint32_t)strtoul(line, NULL, 16); int sp = 0;
            int c = prop(m, &sp);
            printf("%08x %d %d\n", m, c, c ? sp : 0);
        }
        return 0;
    }
    int threads = (argc >= 3) ? atoi(argv[2]) : 8;
    int tb = 0; while ((1 << tb) < threads) tb++;
    int perm[4]; int nfact = fact(n);
    int idx[4];
    for (int pi_i = 0; pi_i < nfact; pi_i++) {
        /* decode permutation index via Lehmer code */
        int avail[4], na = n, r = pi_i;
        for (int i = 0; i < n; i++) avail[i] = i;
        for (int i = 0; i < n; i++) {
            int f = fact(n - 1 - i); int q = r / f; r %= f;
            perm[i] = avail[q];
            for (int k = q; k < na - 1; k++) avail[k] = avail[k + 1];
            na--;
        }
        /* parity of permutation */
        int inv = 0; for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (perm[a] > perm[b]) inv++;
        for (int s = 0; s < (1 << n); s++) {
            int det = ((inv & 1) ? -1 : 1) * ((__builtin_popcount(s) & 1) ? -1 : 1);
            int emap[32];
            for (int e = 0; e < NE; e++) emap[e] = edge_index(vmap(ea[e], perm, s), vmap(eb[e], perm, s));
            Perm P; P.ncyc = 0; int seen[32] = {0};
            for (int e = 0; e < NE; e++) if (!seen[e]) {
                uint32_t cm = 0; int x = e;
                while (!seen[x]) { seen[x] = 1; cm |= 1u << x; x = emap[x]; }
                P.cyc[P.ncyc++] = cm;
            }
            int t = (P.ncyc > 24) ? tb : 0;   /* split only the huge enumerations */
            int nj = 1 << t;
            Job jobs[16]; pthread_t th[16];
            for (int q = 0; q < nj; q++) { memset(&jobs[q], 0, sizeof(Job)); jobs[q].p = &P; jobs[q].tbits = t; jobs[q].pref = q; }
            if (nj == 1) worker(&jobs[0]);
            else { for (int q = 0; q < nj; q++) pthread_create(&th[q], NULL, worker, &jobs[q]);
                   for (int q = 0; q < nj; q++) pthread_join(th[q], NULL); }
            uint64_t cnt[16] = {0};
            for (int q = 0; q < nj; q++) for (int a = 0; a < 16; a++) cnt[a] += jobs[q].cnt[a];
            printf("{\"n\":%d,\"perm\":[", n);
            for (int i = 0; i < n; i++) printf("%d%s", perm[i], i + 1 < n ? "," : "");
            printf("],\"flip\":%d,\"det\":%d,\"cycles\":%d,\"cnt\":[", s, det, P.ncyc);
            for (int a = 0; a < 16; a++) printf("%llu%s", (unsigned long long)cnt[a], a < 15 ? "," : "");
            printf("]}\n"); fflush(stdout);
            (void)idx;
        }
    }
    return 0;
}
