#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int Potoks = 20;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product_omp12(double *a, double *b, double *c, int m, int n){
    #pragma omp parallel num_threads(Potoks)
    {
        int nthreads = Potoks;
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_parallel(double *a, double *b, double *c, int m, int n){
    double t = cpuSecond();
    matrix_vector_product_omp12(a, b, c, m, n);
    t = cpuSecond() - t;
    printf("Elapsed time (parallel): %.6f sec.\n", t);
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n){
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(double *a, double *b, double *c, int m, int n){
    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t);
}

int main(int argc, char **argv){
    int m = 40000;
    int n = m;
    double *a, *b, *c;
    a = (double*) malloc(sizeof(*c) * m * n);
    b = (double*) malloc(sizeof(*c) * m);
    c = (double*) malloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = 1.0;
        a[i * n + i] = 2.0;
    } 
    for (int i = 0; i < n; i++) {
        b[n] = n+1;
    } 
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    run_parallel(a,b,c,m,n);
    //run_serial(a,b,c,m,n);
    free(a);
    free(b);
    free(c);
    return 0;
}