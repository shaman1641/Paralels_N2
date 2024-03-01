#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include  <memory>
#include <math.h>

int Potoks = 20;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}


void matrix_vector_product_omp12(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  c, int m, int n){
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
    //printf("End m X v");
}

void vector_chislo_product_omp12(std::shared_ptr<double []>  a, double b, std::shared_ptr<double []>  c, int m){
    #pragma omp parallel num_threads(Potoks)
    {
        int nthreads = Potoks;
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = a[i] * b;
        }
    }
    //printf("End v X c");
}

void vector_vector_minus_omp12(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  c, int m){
    #pragma omp parallel num_threads(Potoks)
    {
        int nthreads = Potoks;
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = a[i] - b[i];
        }
    }
    //printf("End v - v");
}

void iteration(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  x, std::shared_ptr<double []>  x2, double t, int n, int m){
    //double* ans;
    //ans = (double*) malloc(sizeof(*ans) * m);
    std::shared_ptr<double[]>    ans (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    matrix_vector_product_omp12(a,x,ans,m,n);
    vector_vector_minus_omp12(ans, b, x2, m);
    vector_chislo_product_omp12(x2, t, ans, m);
    vector_vector_minus_omp12(x, ans, x2, m);
    //free(ans);
}

double skobki(std::shared_ptr<double []>  a,int n){
    double sum = 0;
    for(int i =0; i<n; i++){
        sum += a[i];
    }
    return(sqrt(sum));
}

int test_con(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  x, int n, int m, double eps){
    //std::shared_ptr<double []> ans, *x2;
    //ans = (double*) malloc(sizeof(*ans) * m);
    //x2 = (double*) malloc(sizeof(*x2) * m);
    std::shared_ptr<double[]>    ans (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    std::shared_ptr<double[]>    x2 (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    matrix_vector_product_omp12(a,x,ans,m,n);
    vector_vector_minus_omp12(ans, b, x2, m);
    double test = skobki(x2, n)/skobki(b,n) < eps;
    //printf("test: %.10f eps %.10f\n", test, eps);
    if(test < eps){
        return 1;
    }
    else{
        return 0;
    }
    //free(x2);
    //free(ans);
}

void rehenie_omp(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  x, std::shared_ptr<double []>  x2, double t, int n, int m, double eps){
    double t2 = cpuSecond();
    do{
        iteration(a,b,x,x2,t,n,m);
    }while(test_con(a,b,x2,n,m,eps) != 1);
    t2 = cpuSecond() - t2;
    std::cout << "Elapsed time (paralel_1): " <<  t2 <<" sec.\n";
}

void rehenie_omp2(std::shared_ptr<double []>  a, std::shared_ptr<double []>  b, std::shared_ptr<double []>  x, std::shared_ptr<double []>  x2, double t, int n, int m, double eps){
    double t2 = cpuSecond();
    #pragma omp parallel num_threads(Potoks)
    {
        int nthreads = Potoks;
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        int test_con1 = 0;
        do{
            //double* ans;
            //ans = (double*) malloc(sizeof(*ans) * m);
            std::shared_ptr<double[]>    ans (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
            for (int i = lb; i <= ub; i++) {
                ans[i] = 0.0;
                for (int j = 0; j < n; j++)
                    ans[i] += a[i * n + j] * x[j];
            }

            for (int i = lb; i <= ub; i++) {
                x2[i] = ans[i] - b[i];
            }

            for (int i = lb; i <= ub; i++) {
                ans[i] = x2[i] * t;
            }
            
            for (int i = lb; i <= ub; i++) {
                x2[i] = x[i] - ans[i];
            }

            //free(ans);
            //std::shared_ptr<double []> ans12, *x212;
            //ans12 = (double*) malloc(sizeof(*ans12) * m);
            std::shared_ptr<double[]>    ans12 (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
            std::shared_ptr<double[]>    x212 (new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
            //x212 = (double*) malloc(sizeof(*x212) * m);
            for (int i = lb; i <= ub; i++) {
                ans12[i] = 0.0;
                for (int j = 0; j < n; j++)
                    ans12[i] += a[i * n + j] * x[j];
            }
            for (int i = lb; i <= ub; i++) {
                x212[i] = ans12[i] - b[i];
            }
            double test = skobki(x2, n)/skobki(b,n) < eps;
            //printf("test: %.10f eps %.10f\n", test, eps);
            if(test < eps){
                test_con1 = 1;
            }
            else{
                test_con1 = 0;
            }
            //free(x212);
            //free(ans12);
        }while(test_con1 != 1);
    }
    t2 = cpuSecond() - t2;
    std::cout << "Elapsed time (paralel_2): " <<  t2 <<" sec.\n";
}

int main(int argc, char **argv){
    int m = 46000;
    int n = m;
    //std::shared_ptr<double []> a, *b, *c, *x, t, eps;
    double eps, t;
    eps = 0.00001;
    t = -0.01;
    std::shared_ptr<double[]>    a(new double[sizeof(double) * m * n], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    std::shared_ptr<double[]>    b(new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    std::shared_ptr<double[]>    c(new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    std::shared_ptr<double[]>    x(new double[sizeof(double) * m], [] (double* i) { 
  delete[] i; // Кастомное удаление
});
    //a = (double*) malloc(sizeof(*a) * m * n);
    //b = (double*) malloc(sizeof(*b) * m);
    //c = (double*) malloc(sizeof(*c) * m);
    //x = (double*) malloc(sizeof(*x) * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = 1.0;
        a[i * n + i] = 2.0;
    } 
    for (int i = 0; i < n; i++) {
        b[n] = n+1;
    } 
    rehenie_omp(a,b,x,c,t,n,m,eps);
    rehenie_omp2(a,b,x,c,t,n,m,eps);
    //run_serial(a,b,c,m,n);
    //free(a);
    //free(b);
    //free(c);
    return 0;
}