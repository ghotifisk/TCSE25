// Page 120 Example 6.1 Hello World 

#include <stdio.h>          
int main() {                
  printf("Hello World\n");    
  return 0; }               
  
  
// Page 121 Output

printf("%2d times %f is %.2f\n", 3, 4.2, 3*4.2);


// Page 123 Example 6.2  Collatz Problem

int n = 100;  
while (n > 1) {
    if (n % 2 == 0) n /= 2;  
    else n = 3*n + 1;
    printf ("%d\n", n); }
printf("arrived at 1\n");


// Page 123 Remark 6.3  Conditional Expression

n = (n % 2 == 0) ? n/2 : 3*n + 1;


// Page 123 Example 6.4 Rice and Chessboard Problem

int fieldno = 1;
unsigned long fieldval = 1;
unsigned long sum = 1;
for (fieldno = 2; fieldno <= 64; fieldno++) {
    fieldval *= 2;
    sum += fieldval; }
printf("%lu\n", sum);  


// Page 124 Example 6.5 Root Approximation

float r = 1;
float eps = 1.e-6;
for (; (r*r - 2 )*(r*r - 2) > eps*eps; ) 
       r = (r + 2/r) / 2; 
printf("%f\n", r);  


// Page 125 Example 6.6 Factorial Function

int factorial(int n) {  
    int res = 1;
    for (int i = 1; i <= n; i++) res *= i; 
    return res; }
    
int a = factorial(5);   
printf("%d\n", a);     


// Page 125 Example 6.7 Difference-Quotient Function

float diff_quot(float f (float), float x) {  
    float h = 1.e-6;
    return (f(x+h) - f(x)) / h; }
float f_ex(float x) { return 4*x*(1 - x); }  
printf("%f\n", diff_quot(f_ex, 0.5));        


// Page 127 Arrays

int arr[3];                              
for (int i = 0; i < 3; i++) arr[i] = i;  

int arr[3] = {0, 1, 2};
int arr2[] = {0, 1, 2};


// Page 127 Example 6.8 Euclidean Vector Norm

#include<math.h>

float norm(int n, float v[]) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i]*v[i];
    return sqrt(s); }   
float v[] = {0, 3, 4};
printf("%f\n", norm(3,v));


// Page 128 Matrices

int A[2][3];                 
for (int i = 0; i < 2; i++)  
    for (int j = 0; j < 3; j++) A[i][j] = 3*i + j; 
int B[2][3] = { {0, 1, 2}, {3, 4, 5} }; 
int C[][3]  = { {0, 1, 2}, {3, 4, 5} };  
int D[][3]  = {0, 1, 2, 3, 4, 5}; 


// Page 128 Example 6.9 Laplace Expansion

float det(int n, float A[][n]) {
    if (n==1) return A[0][0];             
    float sum = 0;
    for (int col = 0; col < n ; col++) {  
        float A_sub[n-1][n-1];            
        for (int i = 0; i < n-1; i++)     
            for (int j = 0; j < n-1; j++)
                A_sub[i][j] = (j < col) ? A[i+1][j] : A[i+1][j+1];
        float s = A[0][col] * det(n-1, A_sub);  
        sum += ( col % 2 == 0 ) ? s : -s; }     
    return sum; }    
float A[3][3] = { {1, 2, 3}, {1, 1, 1}, {3, 3, 1} }; 
printf("%f\n", det(3, A));


// Page 130 Pointers

int a;
printf("%p\n", &a);  
int* p; 


// Page 130 Example 6.10  Address-Content Function

void vpairf(int* p, int* q, int x) { *p = x; *q = -x; }

int a, b;
vpairf(&a, &b, 1);
printf("a = %d, b = %d\n", a, b);  


// Page 131 Arrays as Pointers

int a[] = {1, 2, 42};  
int* p = a + 2;       
printf("%d\n", *p);    


// Page 131 Example 6.11 Declare Array as Pointer

#include <stdlib.h>                  
int n =3;                           
int* arr = malloc(sizeof(int) * n);  


// Page 131 Example 6.12 Declare Matix as Pointer

int n =3;
int m =3; 
int** A = malloc(sizeof(int*) * n);  
for (int i = 0; i < n; i++)  A[i] = malloc(sizeof(int) * m); 
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) A[i][j] = i*m + j; 
for (int i = 0; i < n; i++) free(A[i]); 
free(A);


// Page 131 Structures

struct { int num; int den; } a;
a.num = 3; a.den = 4;
printf("%d, %d\n", a.num, a.den);  
struct { int num; int den; } a = {.num = 3, .den = 4};


// Page 131 Named Structures

struct frac { int num; int den; };
struct frac a;

typedef struct frac frac;

typedef struct { int num; int den; } frac;


// Page 131 Example 6.13 Fraction Addition

typedef struct { int num; int den; } frac;
frac add(frac a, frac b) {
    int nm = a.num*b.den + a.den*b.num;
    int dn = a.den*b.den;
    frac c = {.num = nm, .den = dn};
    return c; }
    
frac a = { .num = 3 , .den = 4 }; frac b = {.num = 1 , .den = 2};
frac c = add(a,b);
printf("%d, %d\n", c.num, c.den);  


// Page 131 Example 6.14 Sparse Vectors as Linked Lists

#include <stdio.h>
#include <stdlib.h>  
struct node { int index; int value; struct node* next; };
typedef struct node node;
node* create_node(int idx, int val) { 
    node* npt = malloc(sizeof(node));
    (*npt).index = idx;
    (*npt).value = val;
    (*npt).next = NULL;
    return npt; }
typedef struct { node* first; node* last; } vec;
vec* create_vector(void) {  
    vec* v = malloc(sizeof(vec));
    v->first = NULL;
    v->last = NULL;
    return v; }
void append(vec* v, node* npt) {
    if (v->first == NULL) {
        v->first = npt;
        v->last = npt;}
    else (v->last)->next = npt;
    v->last = npt; }
int arr[] = {0, 2, 0, 0, 8, 10, 0, 14, 16, 18};   
vec* v = create_vector();
for (int i = 0; i < 10; i++) {
    if (arr[i] != 0) {
        node* npt = create_node(i, arr[i]);
        append(v, npt); } }
for (node* npt = v->first; npt != NULL; npt = npt->next) 
        printf("%d %d\n", npt->index, npt->value); 
        
        
// Page 136 Example 6.15 Files, Input and Output

float A[2][3] = { {1., 2., 3.}, {4., 5, 6.} }; 
FILE* fp = fopen("test.txt", "w");
int n = 2, m = 3;
fprintf(fp, "%d %d ", n, m);
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
         fprintf(fp, "%f ", A[i][j]);
fclose(fp);

FILE* fp = fopen("test.txt", "r");
int n, m;
fscanf(fp, "%d %d ", &n, &m);
float B[n][m];
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
        fscanf(fp, "%f ", &B[i][j]);
fclose(fp);
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
        printf("%f ", B[i][j]);
    printf("\n"); }
