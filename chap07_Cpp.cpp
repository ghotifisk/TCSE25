// Page 139 Example 7.1

#include <iostream>  
int main() { printf("Hello World\n"); }


// Page 140 Example 7.2

#include <iostream>  
int main() { std::cout << "Hello World\n"; }


// Page 140

std::cout << 1./3 << " times " << 3.14 
          << " is " << 1./3*3.14 << std::endl;
          

// Page 140 Example 7.3

float p = 3.14159;
std::cout.precision(4);           // 4 digit precision ...
std::cout << p << "\n";           // Out: 3.142
std::cout.setf(std::cout.fixed);  // ... here for the decimals
std::cout << p << "\n";           // Out: 3.1416
std::cout.width(8);
std::cout << p << "\n";           // Out:   3.1416


// Page 141 Writing To Files

#include <fstream>
int main() {
	std::ofstream myfile;  // 'o f' for output file
	myfile.open("example.txt");
	myfile << "Text in file\n";
	myfile.close(); }


// Page 141 Reading From Files

#include <iostream>
#include <fstream>
#include <string>      
int main() {
	std::ifstream myfile;  // 'i f' for input file
	myfile.open ("example.txt");
	std::string line;      
	while (!myfile.eof()) {
		getline(myfile, line);
		std::cout << line << '\n'; }}
    
    
// Page 143 Differential Operator

#include <iostream>
using namespace std;
function<double(double)> ddx(double f (double)) {  
    function<double(double)> f_prime;
    f_prime = [f](double x){ double h = 1e-6;
                            return (f(x+h) - f(x)) / h; };
    return f_prime; };
double g(double x) { return x*x; };      

int main() {
    function<double(double)> dgdx = ddx(g);  
    cout << dgdx(2) << endl; } 


// Page 143 Vectors

#include <vector>
using namespace std;

vector<int> u(3);  // 0-initialized
vector<int> v(3,1);
vector<int> w = {1, 2, 3};

// Vector Operations

v[i]; 
v.size();
vector<int> v;
v.resize(3);
v.resize(3,1);


// Page 144 Sparse Vectors

#include <iostream>
#include <vector>
using namespace std;
    int main() {
    int arr[] = {0, 2, 0, 0, 8, 10, 0, 14, 16, 18};
    typedef struct { int index; int value; } node;
    vector<node> node_vec;
    for (int i = 0; i < 10; i++) {
        if (arr[i] == 0) continue;                  // proceed if != 0
        node nd = { .index = i, .value = arr[i] };  // produce node
        node_vec.push_back(nd); }                   // append to vec
    vector<node>::iterator it;
    for (it = node_vec.begin(); it != node_vec.end(); it++)
        cout << ' ' << '(' << it->index << "," << it->value << ")";
    cout << endl; }


// Page 145 Matrices 

vector<vector<int>> A(3, vector<int>(2));


// Page 146 Example 7.4

int a[] = {1, 2};
int& ref = a[0];


// Page 146 Example 7.5 Fraction Class

#include <iostream>
using namespace std;
class Fraction {   // in preamble or main
    public:
        int num, den;
        Fraction(int numerator, int denominator)  {
            num = numerator; den = denominator; }
        //Fraction add(Fraction b) {
        Fraction operator+(Fraction b) {
            return Fraction(num*b.den + b.num*den, den*b.den); }
        bool operator==(Fraction b) {
            return num*b.den == den*b.num; }};
int main() {
    Fraction a(3, 4);  // class name, identifier a, initial values
    cout << a.num << "," << a.den << endl;  // Out: 3, 4
    Fraction b(1, 2);
    cout << (a + b == b + a) << endl; }


// Page 147 - 149  Matrix Class

#include <iostream>
#include <vector>
using namespace std;

class Matrix {
        vector<vector<float>> mat;  // empty matrix
    public:
        int rows, cols;             // internal row/column numbers
        Matrix(int r, int c) {      // constructor
            rows = r; cols = c;
            mat.resize(rows, vector<float>(cols, 0)); }
    float& operator() (int i, int j) { return mat[i-1][j-1]; }
    void printm() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols;  j++) cout << mat[i][j] << "\t";
            cout << endl; }}
    Matrix operator+(Matrix B) {
        Matrix C(rows, cols);  // internal var for sum matrix
        for (int i = 1; i <= rows; i++)
            for (int j = 1; j <= cols; j++)
                C(i,j) = (*this)(i,j) + B(i,j);
        return C; }
};   // end of class def

float det(Matrix A) {
    int n = A.rows;
    if (n != A.cols) {          // if not square: Abort
        cerr << "Error\n";
        exit(41); }
    if (n == 1) return A(1,1);  // recursion's base case
    Matrix B(n-1,n-1);          // declaration of minor submatrix
    float d = 0;
    for (int c = 1; c <= n; c++) {
        for (int i = 1; i <= n-1; i++)
            for (int j = 1; j <= n-1; j++)
                B(i,j) = ( j < c ) ? A(i+1,j) : A(i+1,j+1);
        float s = A(1,c)*det(B);
        d += ( c % 2 == 0 ) ? -s : s; }
    return d; }

int main() {
    Matrix A(2,2);        // declare 2 x 2 matrix
    A(1,1) = A(2,2) = 1;  // assign entries
    cout << det(A) << endl;
    (A + A).printm(); }


// Page 150 Example 7.8 Declaration Order

// int g(int n);
int f(int n) { return n < 0 ? 22 : g(n); }
int g(int n) { return f(n-1); };


// Page  150 - 151  Header and Implementation Files

// Header File "matrix.hpp":

#include <vector>
using namespace std;

class Matrix {
    vector<vector<float>> mat;          // empty Matrix
public:
    int rows, cols;                     // internal vars
    Matrix(int r, int c);               // constructor
    void printm();                      // remaining methods
    float& operator() (int i, int j);   //     dito
    Matrix operator+(Matrix B); };      //     dito

float det(Matrix A);                    // function det, outside of class


// Implementation File "matrix.cpp":

#include "matrix.hpp"   // imports all declarations
#include <iostream>     // needed for the code in printm below

Matrix::Matrix(int r, int c) {       // constructor
    rows = r; cols = c;
    mat.resize(rows, vector<float>(cols, 0)); }

void Matrix::printm() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols;  j++) cout << mat[i][j] << "\t";
        cout << endl; }}

float& Matrix::operator() (int i, int j) { return mat[i-1][j-1]; }

Matrix Matrix::operator+(Matrix B) {
    Matrix C(rows,cols);     // internal var for sum matrix
    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= cols; j++)
            C(i,j) = (*this)(i,j) + B(i,j);
    return C; }

float det(Matrix A) {
    int n = A.rows;
    if (n != A.cols) {          //  if not square: Abort
        cerr << "Error\n";
        exit(41); }
    if (n == 1) return A(1,1);  // recursion's base case
    Matrix B(n-1,n-1);          // submatrices
    float d = 0;
    for (int c = 1; c <= n; c++) {
        for (int i = 1; i <= n-1; i++)
            for (int j = 1; j <= n-1; j++)
                B(i,j) = j < c ? A(i+1,j) : A(i+1,j+1);
        float s = A(1,c) * det(B);
        d += c % 2 == 0 ? -s : s; }
    return d; }

// matrixmain.cpp:

#include <iostream>
#include "matrix.hpp"   // imports all definitions and implementations
int main() {
    Matrix A(2,2);
    A(1,1) = 1;
    A(2,2) = 1;
    (A + A).printm();
    cout << det(A) << endl; }
    
    
// Page 152 - 153 Subclassing

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Polynomial { // in preamble or in main
    public:
        vector<float> coeff;
        Polynomial(vector<float> arr) { coeff = arr; };
        int operator()(float x) {
            float s = 0;
            for (int i = 0; i < coeff.size() ; i++ )
                { s += coeff[i] * pow(x, i); }
            return s; }
        Polynomial operator+(Polynomial q) {
            cout << "To be implemented. Dummy: ";
            Polynomial r({0}); return r; }
}; // end of Polynomial definition

class Parabola : public Polynomial {
    public:
        Parabola(vector<float> c) : Polynomial(c) {
            if (c.size() != 3) {cout << "No parabola\n"; exit(41);}};
    Parabola operator+(Parabola q) {
            Parabola r({0,0,0});
            for (int i = 0; i < 3 ; i++ )
                { r.coeff[i] = (*this).coeff[i] + q.coeff[i]; }
            return r; }
};  // end of Parabola definition

int main() {
    Polynomial p({1,2,3});
    cout << p(4) << endl;  // Out: 57
    cout << (p+p)(4) << endl;
    Parabola r({1,2,3});   // valid parabola
    cout << r(4) << endl;  // Out: 57
    cout << (r+r)(4) << endl; // Out: 114
    //Parabola q({1,2,3,4}); 
}
