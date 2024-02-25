#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

int Potoks = 7;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x){
    return exp(-x * x);
}

double integrate(double a, double b, int n){
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double a, double b, int n){
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel num_threads(Potoks)
    {
        int nthreads = Potoks;
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
        #pragma omp atomic
            sum += sumloc;
    }
    sum *= h;
    return sum;
}

int main(int argc, char **argv){
    int n = 40000000;
    double b = -4.0;
    double a = 4.0;
    double t = cpuSecond();
   /*  integrate(a, b, n);
    t = cpuSecond() - t;
    printf("Elapsed time (serial): %.6f sec.\n", t); */
    t = cpuSecond();
    integrate_omp(a, b, n);
    t = cpuSecond() - t;
    printf("Elapsed time (paralel): %.6f sec.\n", t);
    return 0;
}