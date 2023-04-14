// Page 292 Hello World  

#include <stdio.h>
int main() { printf("Hello World\n"); }


// Page 292 Example 13.1 Pi Integral Approximation  

#include <mpi.h>
#include <stdio.h>
float f(float x) { return 4.0/(1.0 + x*x); }
float partial_pi(int n, int start, int step) {
    float h = 1.0/n;
    float sum = 0.0;
    for (int i = start; i < n; i += step) {
        float x = h*(i + 0.5);
        sum += h*f(x); }
    return sum; }
int main() {
    MPI_Init(NULL,NULL);   
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n;
    if (rank == 0) n = 10; 
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    float pi_loc = partial_pi(n, rank, size);
    float pi;
    MPI_Reduce(&pi_loc, &pi, 1, MPI_FLOAT, 
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%f\n", pi); 
    MPI_Finalize(); }  
    
    
// Page 294 Example 13.2 Sequence Average 
    
#include <mpi.h>
#include <stdio.h>
int main() {
    MPI_Init(NULL,NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 16;
    float arr[n];
    if (rank == 0) for (int i = 0; i < n; i++) arr[i] = i; 
    int n_loc = n/size;
    float arr_loc[n_loc]; 
    MPI_Scatter(arr, n_loc, MPI_FLOAT, 
			    arr_loc, n_loc, MPI_FLOAT, 0, MPI_COMM_WORLD);   
    float sum_loc = 0.;
    for (int i = 0; i < n_loc ; i++) sum_loc += arr_loc[i];
    float avg_loc = sum_loc / n_loc;
    float avg_arr[size];
    MPI_Gather(&avg_loc, 1, MPI_FLOAT, 
               avg_arr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        float sum = 0.;
        for (int i = 0; i < size ; i++) 
            sum += avg_arr[i]; 
        float avg = sum/size;
        printf("%f\n", avg); }
    MPI_Finalize(); }
    
    
// Page 295 Example 13.3 Conjugate Gradient 

#include <stdio.h>
#include <mpi.h>
#include <math.h> 
float dot(int n, float v[], float w[]) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += v[i]*w[i];
    return sum; }
int main() {
    int n = 4;
    float A[] = {9., 3., -6., 12., 3., 26., -7., -11., 
                -6., -7., 9., 7., 12., -11., 7., 65.};
    float b[] = {18., 11., 3., 73.};
    float x[] = {0., 0., 0., 0.};
    float r[n], p[n];
    for (int i = 0; i < n; i++) { r[i] = b[i]; p[i] = r[i]; }
    float rs_old = dot(n,r,r); 
    MPI_Init(NULL,NULL);    
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n_loc = n/size;
    float A_loc[n_loc][n];
    MPI_Scatter(A, n_loc * n, MPI_FLOAT, 
                A_loc, n_loc*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int it = 0; it < n; it++) {  
        MPI_Bcast(p, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        float Ap_loc[n/size];
        for (int i = 0; i < n_loc; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                sum +=  A_loc[i][j]*p[j];}
            Ap_loc[i] = sum;}
        float Ap[n];
        MPI_Gather(Ap_loc, n_loc, MPI_FLOAT, 
                   Ap, n_loc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            float alpha = rs_old / dot(n, p, Ap);
            for (int i = 0; i < n; i++)  x[i] += alpha*p[i];
            for (int i = 0; i < n; i++)  r[i] -= alpha*Ap[i];
            float rs_new = dot(n, r, r);
            if (sqrt(rs_new) < 1e-6) break;
            for (int i = 0; i < n; i++) 
                p[i] = r[i] + (rs_new / rs_old)*p[i];
            rs_old = rs_new; } }   
    if (rank == 0)  {
        for (int i = 0; i < n; i++)  printf("%f ", x[i]);
        printf("\n"); }
    MPI_Finalize(); }


// Page 298 Example 13.4 OMP Hello World  

#include <stdio.h>
#include <omp.h>
int main() {
    printf("Number of processors: %d \n", omp_get_num_procs());
    #pragma omp parallel  
    { printf("Hello World from thread number %d of %d\n",
           omp_get_thread_num(), omp_get_num_threads()); } }
           
                     
// Page 299 Example 13.5 OMP Broadcast, Naive 

#include <stdio.h>
#include <omp.h>
int main() {
    int ans = 0;
    #pragma omp parallel 
    { if (omp_get_thread_num() == 0) ans = 42;
      printf("%d ", ans); }
    printf("\n"); }
    
       
// Page 299, 300 Example 13.5, 13.6 OMP Broadcast, "barrier" Directive 

#include <stdio.h>
#include <omp.h>
int main() {
    int ans = 0;
    #pragma omp parallel 
    { if (omp_get_thread_num() == 0) ans = 42;
    #pragma omp barrier
      printf("%d ", ans); }
    printf("\n"); }
    
    
// Page 301 Example 13.7 OMP Summation, Naive 

#include <stdio.h>
int main() {
    int sum = 0;
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 1; i <= 100; i++) sum += i;
    }
    printf("%d\n", sum); }
    
    
// Page 301, 302  OMP Summation, "critical", "atomic" Directives 

#include <stdio.h>
int main() {
    int sum = 0;
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 1; i <= 100; i++) {
        #pragma omp critical  // atomic pragma omp critical
        sum += i; }
    }
    printf("%d\n", sum); }
    

// Page 303 "reduction" Clause 

#include <stdio.h>
int main() {
    int sum = 0;
    #pragma omp parallel 
    {
        #pragma omp for reduction(+:sum)
        for (int i = 1; i <= 100; i++) sum += i;
    }
    printf("%d\n", sum); }
    
    
// Page 303 Example 13.8 Pi Integral Approximation 

#include <stdio.h>
float f(float x) { return 4.0/(1.0 + x*x); }
int main() {
    int n = 10; float h = 1.0/n;
    float x;  
    float sum = 0.0;
    # pragma omp parallel 
    {        
        #pragma omp for private(x) reduction(+:sum)    
            for (int i = 0; i < n; i++) {
        	x = h*(i + 0.5);
        	sum += h*f(x); } 
        #pragma omp single
            printf("%f\n", sum); } }  
            
            
// Page 304 Example 13.9 Vector Product 

#include <stdio.h>
float omp_dot(int n, float v[], float w[]) {
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < n; i++) sum += v[i]*w[i];
    return sum; }
int main() {
    int n = 10;
    float a[n], b[n];
    for (int i=0; i < n; i++){ 
        a[i] = i; b[i] = n-i; }
    float a_dot_b = omp_dot(n, a, b);   
    printf("%f\n", a_dot_b); }
    
    
// Page 305 Example 13.10 Fibonacci Function 

int fib(int n) {
    int i, j;
    if (n <= 2) return 1;
    #pragma omp task shared(i)
        i = fib(n-1);
    #pragma omp task shared(j)
        j = fib(n-2);
    #pragma omp taskwait
        return i + j; }
#include <stdio.h>
int main() {
    int n = 20;
    #pragma omp parallel
    {
        #pragma omp single
            printf("%d\n",fib(n)); } } 
            
            
// Page 305 Hybrid MPI-OMP Pi Approximation  

#include <stdio.h>
#include <mpi.h>
float f(float x) { return 4.0/(1.0 + x*x); }
float omp_partial_pi(int n, int start, int step) {   
    float h = 1.0/n;
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
        for (int i = start; i < n; i += step) 
            sum += h*f(h*(i + 0.5)); 
    return sum; }
int main() {
    MPI_Init(NULL,NULL);
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int n;
    if (rank == 0) n = 10; 
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    float pi_loc = omp_partial_pi(n, rank, size);  
    float pi;
    MPI_Reduce(&pi_loc, &pi, 1, MPI_FLOAT, 
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("%f\n", pi); }
    MPI_Finalize(); }