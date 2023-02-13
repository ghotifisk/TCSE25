# Page 221 Arrow Operator

restart;
f := x -> a*x^2;
g := (x,h) -> (f(x+h) - f(x)) / h; 
delta := (x, y) -> if x = y then 1; else 0; end if;


# Page 221 "unapply" Command

term := x + x^2:
eval(term, x = 1);  
f := unapply(term, x);


# Page 222 Example 10.1 Factorial

fact := proc(n)
    local i, prod;
    if n < 0 then return "n must be positive"; end;	   
    prod := 1;
    for i from 1 to n do prod := i*prod; end;
    return(prod);
end proc;


# Page 223 Example 10.2 Collatz Problem

fcollatz := proc(n)
    local m, count;
    count := 0; m := n;
    while m > 1 do 
        if modp(m,2) = 0  then m := m/2;
        else  m := 3*m + 1; end;
        count := count + 1; end;
    return count;  
end proc;


# Page 223 Example 10.3 Recursive Factorial

fact_rec := proc(n)
    if n = 1 then return 1; end;	   
    return n*fact_rec(n-1);
end;


# Page 223 Example 10.4 Visualization, Runge Function

f := x -> 1/(1+ 25 * x^2):
p := 1250/377*x^4 - 3225/754*x^2 + 1:
plot({f(x), p}, x=-1..1, size=[default,golden]);


# Page 224 Symbolic Equation Solution

quad := x^2 + 2*x + c = 0:
solve(quad, x); 


# Page 224 Numeric Equation System

eq1 := (sin(x + y))^2 + exp(x)*y + cot(x - y) + cosh(z + x) = 0:
eq2 := x^5 - 8*y = 2:
eq3 := x + 3*y - 77*z = 55:
fsolve({eq1, eq2, eq3});  


# Page 225 Vectors

v := Vector([1, 0, -3]):  
w := <1, 0, -3>:

v := Vector[row]([1, 0, -3]):
v := <1 | 0 | -3>:


# Page 225 Matrices

A = Matrix([[1, 2], [3, 4]]);
A := <<1, 3> | <2, 4>>;
A.A
A.~A


# Page 225 Remark Elementwise Function Application

f := x -> 1/x:
v := <1 | 2 | 3>:  
f~(v); 


# Page 226 Function Values as Entries

f := i -> i^2:
v := Vector(10, f);
v := Vector(10, i -> i^2);

v := Vector(3);


# Page 226 Hilbert Matrix

H := Matrix(3, 3, (i,j) -> 1/(i+j-1), shape=symmetric);


# Page 226 Example 10.5 R3-Space Orthonormal Basis

v1 := <1, 1, 1>  / sqrt(3):
v2 := <1, 0, -1> / sqrt(2):
v3 := <1, -2, 1> / sqrt(6):
v1.v1;
v1.v2;
x := <a,b,c>:
y:=(v1.x)*v1 + (v2.x)*v2 + (v3.x)*v3;
simplify(y)


# Page 227 Example 10.6 Polynomial-Space Basis

restart;
p[1] := 1: p[2] := x - 1/2: p[3] := x^2 - x + 1/6: 
p_lin := add(c[i]*p[i], i = 1..3):
q := add(a[i]*x^i, i = 0..2):
r := q - p_lin:
solve({coeff(r, x, 0) = 0, coeff(r, x, 1) = 0, 
       coeff(r, x, 2) = 0}, {c[1], c[2], c[3]}); 
       
       
# Page 227 Example 10.7 Linear Independence

with(LinearAlgebra):
u1 := <1, 0, 2>:  u2 := <0, 1, 1>: u3 := <1, 2, -1>: 
A := <u1 | u2 | u3>; 
Determinant(A);
Rank(A);
NullSpace(A);
x := <8, 2, -4>:
c := LinearSolve(A,x); 
c[1]*u1 + c[2]*u2 + c[3]*u3 - x;


# Page 228 Example 10.8 Nonempty Kernel

with(LinearAlgebra):
C := Matrix([[1, 3, -1, 2], [0, 1, 4, 2], [2, 7, 2, 6], [1, 4, 3, 4]]):
NullSpace(C);


# Page 228 Eigenvectors and Eigenvalues

A := Matrix([[1, 2, -1], [4, 0, 1], [-7, -2, 3]]):
with(LinearAlgebra):
res := Eigenvectors(A);
e_vals :=res[1]:
V := res[2]:
A.Column(V,2) - e_vals[2]*Column(V,2):
simplify(%);


# Page 229 5x5 Hilbert Matrix

H5 := Matrix(5, 5, (i,j) -> 1/(i+j-1));
with(LinearAlgebra):
Eigenvalues(H5);

Eigenvalues(evalf(H5));

unassign('H5'):
H5 := Matrix(5, 5, (i,j) -> 1/(i+j-1), shape=symmetric):
Eigenvalues(evalf(H5));

# Page 230 Derivation

f := x -> a*x^2:
delta_f := (x,h) -> (f(x+h) - f(x))/h:           
f_prime := x -> limit(delta_f(x,h), h = 0):  

restart;
diff(x*sin(cos(x)),x)

w := (x,y) -> sin(y - a*x):  # example function
diff(w(x,y), x);          
diff(diff(w(x,y), x), x);
diff(w(x,y), x, x);
diff(w(x,y), x$2, y$3);


# Page 231 Differential Operator

w:=(x,y) -> x^4*y^4 + x*y^2;
D[1](w);


# Page 231 Integration

int(sin(x)^2, x); 
int(sin(x)^2, x = 0..Pi);
int(exp(x^3), x = 0..2); 
evalf(%);
int(1/s, s = 1..x) assuming x > 0; 


# Page 232 Example 10.9 Inner Product and Orthogonal Functions

int(sin(x)*cos(x), x = 0...Pi);  # Out: 0
sqrt(int(sin(x)*sin(x), x = 0...Pi)); 


# Page 232 Example 10.10 Cubic Spline Interpolation

restart;
n := 3: 
x_vec := Vector(n+1, i -> (i-1)/n):  
y_vec := Vector([0, 1/3, 1, 3/4]):
pts := <x_vec | y_vec>:              
with(CurveFitting):
spl3 := Spline(pts, x);
splplot := plot(spl3, x = 0..1):
with(plots):  # contains pointplot, display 
ptsplot := pointplot(pts, symbol=solidcircle, symbolsize=15, color=black):
display(splplot, ptsplot);


# Page 233 Example 10.11 Polygonal Interpolation 
# Ex. 10.10 cont'd 

spl1 := Spline(pts, x, degree=1);
spl1plot := plot(spl1, x = 0..1);
display(spl1plot, ptsplot);


# Page 233 Example 10.12 Hat Functions
# Ex. 10.11 cont'd  

plot_arr := Array(1..4):             
for i from 1 to 4 do 
    y_hat := Vector(n+1):            
    y_hat[i] := 1:                   
    phi[i] := Spline(x_vec, y_hat, x, degree=1):
    plot_arr[i] := plot(phi[i], x = 0..1):
end do:
display(plot_arr);


# Page 234 Example 10.13 Hat Function Interpolation
# Ex. 10.12 cont'd  

f_expr := add(c[i]*phi[i], i = 1..4):
f := unapply(f_expr, x):
sols := solve({f(0) = 0, f(1/3) = 1/3, f(2/3) = 1, f(1) = 3/4}, 
              {c[1], c[2], c[3], c[4]}):
assign(sols):
plot(f(x), x = 0..1);  


# Page 235 Example 10.14 BVP, Direct Integration  

restart;
int(-(1 + x), x) + c1;
int(%,x) + c2;
u := unapply(%,x);
sols := solve({u(0) = 0, u(1) = 0}, {c1, c2});
assign(sols):
u(x);


# Page 236 Example 10.15 BVP, "dsolve"

ode := diff(u(x), x, x) = -(1 + x):
bc := u(0) = 0, u(1) =0:
dsolve({ode, bc});


# Page 236 Euler Method

restart;
myeuler := proc(f, u0, n)
    local dx, pts, i:
    dx := 1/n:      
    pts := Matrix(n+1,2):  # to store (x,u)-pairs
    pts[1,1] := 0:         # start of x-intverval
    pts[1,2] := u0:        # initial u-value 
    for i from 1 to n do    
        pts[i+1,1] := pts[i,1] + dx:
        pts[i+1,2] := pts[i,2] + f(pts[i,1], pts[i,2])*dx end do:
    return pts
end proc:
f := (x,u) -> u / (1 + x^2):
u_eul := myeuler(f, 1, 10):
with(plots):  # contains the function pointplot
plot1 := pointplot(u_eul, symbolsize=15):
ode := diff(u(x), x) = u(x)/(1 + x^2):
ic := u(0) = 1:
sol := dsolve({ode, ic}):
u := rhs(sol);
plot2 := plot(u, x = 0..1):
display(plot1, plot2);


# Page 238 Galerkin Method

phi[1]:= x*(1 - x):
phi[2]:= x*(1/2 - x)*(1 - x):
phi[3]:= x*(1/3 - x)*(2/3 - x)*(1 - x):
phi[4]:= x*(1/4 - x)*(1/2 - x)*(3/4 - x)*(1 - x):
a := (u,v) -> -int(diff(u,x,x)*v, x = 0..1):
f := x*(x + 3)*exp(x):
L := v -> int(f*v, x = 0..1):
A := Matrix(4, 4, (i,j) -> a(phi[j], phi[i])):
b := Vector(4, j -> L(phi[j])):
with(LinearAlgebra):
c := LinearSolve(A,b):
u := add(c[i]*phi[i], i = 1..4):
plot(u, x = 0..1);


# Page 298 Exact Solution
# Last Example cont'd

int(f,x) + c1:
int(%,x) + c2:
u_e := unapply(%,x):
sols := solve({u_e(0) = 0, u_e(1) = 0}, {c1, c2}):
assign(sols):
u_e(x);


# Page 240 Finite Element Method
 
with(LinearAlgebra):
with(CurveFitting):
n := 10:
x_vec := Vector(n+1, i -> (i-1)/n):
for i from 1 to n+1 do
    y_hat := Vector(n+1):
    y_hat[i] := 1:
    phi[i] := Spline(x_vec, y_hat, x, degree=1) 
end do:
a := (u,v) -> int(diff(u,x)*diff(v,x), x = 0..1):
f := x * (x+3) * exp(x):
L := v -> int(f*v, x = 0..1):
A := Matrix(n-1, n-1, shape=symmetric, storage=band[1, 0]):
for i from 1 to n-1 do A[i,i] := a(phi[i+1], phi[i+1]) end do:
for i from 1 to n-2 do A[i+1,i] := a(phi[i+1], phi[i+2]) end do:
b := Vector(n-1, j -> L(phi[j+1])):
c := LinearSolve(A,b):
c := Vector(n+1, [0, c, 0]):
u := add(c[i]*phi[i], i = 1..11):
plot(u, x = 0..1);
