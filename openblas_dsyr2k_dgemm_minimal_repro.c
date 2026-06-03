// openblas_dsyr2k_dgemm_repro.c
//
// Minimal reproducer for silent incorrect DGEMM results when a threaded
// DSYR2K call overlaps with a threaded DGEMM call in another application
// thread.
//
// Background thread:
//   repeated DSYR2K('U','N') with patterned inputs
//
// Foreground thread:
//   repeated DGEMM('N','N') with ones inputs
//   checks that ones(M,K) @ ones(K,N) gives every C entry equal to K
//
// Usage:
//   ./openblas_dsyr2k_dgemm_repro [use_mutex] [syr_n] [syr_k] [M] [N] [K] [calls]
//
// Defaults:
//   use_mutex = 0
//   syr_n     = 32
//   syr_k     = 32
//   M         = 64
//   N         = 64
//   K         = 128
//   calls     = 50
//
// Example failing case:
//   OMP_NUM_THREADS=8 ./openblas_dsyr2k_dgemm_repro 0
//
// Controls:
//   OMP_NUM_THREADS=8 ./openblas_dsyr2k_dgemm_repro 1
//   OMP_NUM_THREADS=1 ./openblas_dsyr2k_dgemm_repro 0
//
// Below-threshold controls:
//   OMP_NUM_THREADS=8 ./openblas_dsyr2k_dgemm_repro 0 31 32 64 64 128 500
//   OMP_NUM_THREADS=8 ./openblas_dsyr2k_dgemm_repro 0 32 32 64 64 127 500

#define _GNU_SOURCE

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern char *openblas_get_config(void);
extern int openblas_get_num_threads(void);

extern void dgemm_(
    const char *TRANSA, const char *TRANSB,
    const int *M, const int *N, const int *K,
    const double *ALPHA,
    const double *A, const int *LDA,
    const double *B, const int *LDB,
    const double *BETA,
    double *C, const int *LDC
);

extern void dsyr2k_(
    const char *UPLO, const char *TRANS,
    const int *N, const int *K,
    const double *ALPHA,
    const double *A, const int *LDA,
    const double *B, const int *LDB,
    const double *BETA,
    double *C, const int *LDC
);

static pthread_mutex_t blas_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_int bg_entered = 0;
static atomic_int bg_stop = 0;
static atomic_long bg_calls = 0;
static int use_mutex = 0;

static void *xalloc(size_t nbytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, nbytes) != 0 || p == NULL) {
        fprintf(stderr, "allocation failed for %zu bytes\n", nbytes);
        exit(1);
    }
    return p;
}

static void fill(double *x, size_t n, double value) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = value;
    }
}

static void fill_pattern(double *x, int rows, int cols, int ld, double base) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            x[i + (size_t)j * (size_t)ld] =
                base + 0.01 * (double)(i + 1) + 0.001 * (double)(j + 1);
        }
    }
}

typedef struct {
    int n;
    int k;
    int ld;
    double *A;
    double *B;
    double *C;
} Syr2kCtx;

static void call_dsyr2k(Syr2kCtx *s) {
    const char uplo = 'U';
    const char trans = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;

    memset(s->C, 0, sizeof(double) * (size_t)s->ld * (size_t)s->n);

    dsyr2k_(
        &uplo, &trans,
        &s->n, &s->k,
        &alpha,
        s->A, &s->ld,
        s->B, &s->ld,
        &beta,
        s->C, &s->ld
    );
}

static void *syr2k_thread(void *arg) {
    Syr2kCtx *s = (Syr2kCtx *)arg;

    atomic_store_explicit(&bg_entered, 1, memory_order_release);

    while (atomic_load_explicit(&bg_stop, memory_order_acquire) == 0) {
        if (use_mutex) {
            pthread_mutex_lock(&blas_mutex);
        }

        call_dsyr2k(s);

        if (use_mutex) {
            pthread_mutex_unlock(&blas_mutex);
        }

        atomic_fetch_add_explicit(&bg_calls, 1, memory_order_relaxed);
    }

    return NULL;
}

static void call_dgemm(double *A, double *B, double *C, int M, int N, int K) {
    const char trans = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;

    if (use_mutex) {
        pthread_mutex_lock(&blas_mutex);
    }

    dgemm_(
        &trans, &trans,
        &M, &N, &K,
        &alpha,
        A, &M,
        B, &K,
        &beta,
        C, &M
    );

    if (use_mutex) {
        pthread_mutex_unlock(&blas_mutex);
    }
}

static int check_gemm(double *C, int M, int N, int K, int iter) {
    const double expected = (double)K;
    const size_t total = (size_t)M * (size_t)N;

    double max_abs_err = 0.0;
    size_t bad = 0;
    size_t first_bad = 0;
    double first_value = 0.0;

    for (size_t i = 0; i < total; ++i) {
        double err = fabs(C[i] - expected);

        if (err > max_abs_err) {
            max_abs_err = err;
        }

        if (err > 1e-9) {
            if (bad == 0) {
                first_bad = i;
                first_value = C[i];
            }
            ++bad;
        }
    }

    if (bad) {
        fprintf(stderr,
                "FAIL iter=%d max_abs_err=%g bad=%zu "
                "first_bad_index=%zu first_bad_value=%g expected=%g\n",
                iter, max_abs_err, bad, first_bad, first_value, expected);
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    use_mutex = (argc > 1) ? atoi(argv[1]) : 0;

    const int syr_n = (argc > 2) ? atoi(argv[2]) : 32;
    const int syr_k = (argc > 3) ? atoi(argv[3]) : 32;
    const int M     = (argc > 4) ? atoi(argv[4]) : 64;
    const int N     = (argc > 5) ? atoi(argv[5]) : 64;
    const int K     = (argc > 6) ? atoi(argv[6]) : 128;
    const int calls = (argc > 7) ? atoi(argv[7]) : 50;

    if (syr_n < 1 || syr_k < 1 || M < 1 || N < 1 || K < 1 || calls < 1) {
        fprintf(stderr,
                "Usage: %s [use_mutex] [syr_n] [syr_k] [M] [N] [K] [calls]\n",
                argv[0]);
        return 2;
    }

    fprintf(stderr, "OpenBLAS config: %s\n", openblas_get_config());
    fprintf(stderr, "OpenBLAS num_threads: %d\n", openblas_get_num_threads());
    fprintf(stderr,
            "use_mutex=%d syr_n=%d syr_k=%d M=%d N=%d K=%d calls=%d "
            "SYR2K_work=%d GEMM_work=%zu\n",
            use_mutex, syr_n, syr_k, M, N, K, calls,
            syr_n * syr_k,
            (size_t)M * (size_t)N * (size_t)K);

    Syr2kCtx s;
    s.n = syr_n;
    s.k = syr_k;
    s.ld = syr_n;
    s.A = xalloc(sizeof(double) * (size_t)s.ld * (size_t)s.k);
    s.B = xalloc(sizeof(double) * (size_t)s.ld * (size_t)s.k);
    s.C = xalloc(sizeof(double) * (size_t)s.ld * (size_t)s.n);

    fill_pattern(s.A, s.n, s.k, s.ld, 0.75);
    fill_pattern(s.B, s.n, s.k, s.ld, 1.00);
    fill(s.C, (size_t)s.ld * (size_t)s.n, 0.0);

    double *GA = xalloc(sizeof(double) * (size_t)M * (size_t)K);
    double *GB = xalloc(sizeof(double) * (size_t)K * (size_t)N);
    double *GC = xalloc(sizeof(double) * (size_t)M * (size_t)N);

    fill(GA, (size_t)M * (size_t)K, 1.0);
    fill(GB, (size_t)K * (size_t)N, 1.0);
    fill(GC, (size_t)M * (size_t)N, 0.0);

    pthread_t th;

    atomic_store_explicit(&bg_entered, 0, memory_order_release);
    atomic_store_explicit(&bg_stop, 0, memory_order_release);
    atomic_store_explicit(&bg_calls, 0, memory_order_release);

    if (pthread_create(&th, NULL, syr2k_thread, &s) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        return 1;
    }

    while (atomic_load_explicit(&bg_entered, memory_order_acquire) == 0) {
    }

    int failed = 0;

    for (int iter = 0; iter < calls; ++iter) {
        memset(GC, 0, sizeof(double) * (size_t)M * (size_t)N);

        call_dgemm(GA, GB, GC, M, N, K);

        if (check_gemm(GC, M, N, K, iter)) {
            failed = 1;
            break;
        }
    }

    atomic_store_explicit(&bg_stop, 1, memory_order_release);
    pthread_join(th, NULL);

    const long b_calls = atomic_load_explicit(&bg_calls, memory_order_acquire);

    if (failed) {
        fprintf(stderr, "Observed failure after %ld background DSYR2K calls.\n", b_calls);
    } else {
        fprintf(stderr,
                "No incorrect result observed after %d foreground DGEMM calls "
                "while background performed %ld DSYR2K calls.\n",
                calls, b_calls);
    }

    free(s.A);
    free(s.B);
    free(s.C);
    free(GA);
    free(GB);
    free(GC);

    return failed ? 1 : 0;
}
