% Page 195 Vectors

v = [1 2 3 4]   % v = 1 2 3 4 
w = 1 : 4       % v = 1 2 3 4
x = 1: -0.3: 0  % x = 1.0000 0.7000 0.4000 0.1000 


% Page 195 Colon Operator

x = 1: ‐0.3: 0 % x = 1.0000 0.7000 0.4000 0.1000


% Page 196 linspace

linspace(-1, 1, 5)  % ans = -1.0000 -0.5000 0 0.5000 1.0000


% Page 196 Dynamic Vector Extension

v = [1 2]
v(5) = 1  % v = 1 2 0 0 1
clear w
w(3) = 1  % w = 0 0 1


% Page 196 Logical Arrays, Comparison Operators

x = [-1 2 3];
x > 0  % ans = 1×3 logical array  0  1  1

b = [false true true]  % 1×3 logical array 0  1  1

x = [true false true]; y = [true true false];
~x      % 1×3 logical array  0  1  0
x & y   % 1×3 logical array  1  0  0
x | y   % 1×3 logical array  1  1  1


% Page 196 Vector of Length 1  Identified with  Content 

x = 1; y = [1]; 
x == y  % ans = logical 1


% Page 197 Matrix

A = [1 2 3; 1 1 1; 3 3 1]


% Page 197 Matrix Defined and Extended Dynamically:

B(2,2) = 1
B(1,4) = 2


% Page 197 Colon Operator in Matrix

A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]; A(3:4,2:3)


% Page 198 Special Vectors

ones(1,3)  % row vector
ones(3,1)  % column vector


% Page 198 Hilbert Matrix Inverse

invhilb(3)


% Page 199 Matrix Division

A = [1 2; 3 4]; B = [5 6; 7 8]; 
C = A / B
C*B


% Page 199 Elementwise Operations

A = [1 2; 3 4]; B = [5 6; 7 8]; 
A .* B  % ans = [5 12; 21 32]    
A.^2    % ans = [1  4;  9 16]
A ./ B  % ans = [0.2000 0.3333; 0.4286 0.5000]
A .\ B  % ans = [5.0000 3.0000; 2.3333 2.0000]

v = [2 3];
1 ./ v  % ans = [0.5000 0.3333]


% Page 200 One-Line Conditional

if x > y, tmp = y; y = x; x = tmp; end


% Page 200 Example 9.1 Tridiagonal Matrix

n = 3;
for i = 1:3, for j = 1:3
    if i == j, D(i,j) = 4;
    elseif abs(i-j) == 1, D(i,j) = -1;
end, end, end


% Page 200 Example 9.2 Machine Epsilon

epsi = 1;      
while 1 + epsi > 1, epsi = epsi/2; end
format long
epsi = 2*epsi   


% Page 201 Loop vs Vector  

tic, i = 0;
for t = 0: 0.01: 10
  i = i + 1;
  y(i) = sin(t);
end, toc  
tic, t = 0: 0.01: 10; y = sin(t); toc 


% Page  202 Example 9.3 Plot Sigmoid Function

x = linspace(-6, 6); 
y = 1 ./ (1 + exp(-x)); 
plot(x, y, 'LineWidth', 2); 


% Page 203 Matrix Functions

A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
[n,m] = size(A)
sum(diag(A))   
sum(A')        
sum(A(:,3))    
sum(sum(A))    
diag(diag(A))  
A(:,2) = []    
reshape(A, 2, 8)
reshape(A', 8, 2)'


% Page 204 Example 9.4 Script scriptdemo.m 

s = 0;
for i = 0: 2: 100, s = s + i; end, s

type scriptdemo
scriptdemo 


% Page 205 Function File mv.m

%%% This function calculates the mean of a vector
function m  = mv(x)
n = length(x);
m = sum(x)/n;

x = [1 2 3 4 5 6];
mv(x) 


% Page 205 Function File stat.m

function [m,s] = stat(x)
n = length(x);
m = avg(x,n);  % subfunction defined below
s = sqrt(sum((x - m).^2)/n);

function m = avg(x,n)
m = sum(x)/n;

x = 1 : 100;
[mv, stddev] = stat(x)  


% Page 205  Function as Argument

function plotfunc(f,x)
y = feval(f,x);
plot(x,y);

x = -3: 0.01: 3;
plotfunc(@sin, x);  


% Page 206 Anonymous Functions

f = @(x) x.^2 + 2*x;  
f(.5) 

g = @(x,y) x.^2 - y.^2  
              

206 Save and Load

A = [1 2; 3 4];
save('a.txt', 'A', '-ascii')
B = load('a.txt');


% Page 207 Example 9.5 Linear Equation Systems

A = [1 2 3; 1 1 1; 3 3 1]; b = [2 2 0]';
x = A \ b


% Page 208 LU Decomposition

A = [1 2 3; 1 1 1; 3 3 1];  
[L,U,P] = lu(A)


% Page 208 Cholesky Decomposition

A = [1 2 1; 2 5 2; 1 2 10]; 
U = chol(A)


% Page 208 QR Decomposition

A = [12 -51 4; 6 167 -68; -4 24 -41]; 
[Q,R] = qr(A)


% Page 208 QR Decomposition Economic Form

A = [3 -6; 4 -8; 0 1]; b = [-1 7 2]';
[Q,R] = qr(A,0)
x = R\Q'*b  


% Page 209 Remark Least Squares Solver

A = [3 -6; 4 -8; 0 1]; b = [-1 7 2]';
x = A\b  


% Page 209 Euler Method

u(1) = 0;  
n = 100;
x = linspace(0, 1, n); 
for i = 1:(n-1) u(i+1) = u(i) + 2*x(i)/n; end
plot(x,u)


% Page 209 Simple IVP 

dudx = @(x,u) 2*x;  
[x,u] = ode45(dudx, [0, 1], 0);
plot(x,u)


% Page 210 IVP 

dudx = @(x,u) x - u; 
[x,u] = ode45(dudx, [0, 5], 1);
plot(x,u,'-o', 'LineWidth', 2)


% Page 211 2nd Order ODE

dydx = @(x,y) [y(2) -y(1)]';
y0 = [0 1];
xint = [0 10];
[x y] = ode45(dydx, xint, y0);
u = y(:,1);
plot(x,u)


% Page 211 BVP

dydx = @(x,y) [y(2) -2]';
bc = @(lb,rb) [lb(1) rb(1)-3]';  
solinit = bvpinit([0 5], [0 0]);
sol = bvp4c(dydx, bc, solinit);
x = linspace(0,5);
y = deval(sol, x, 1);
u = y(1,:);
plot(x,u)


% Page 212 Bratu Equation

dydx = @(x,y) [y(2) -exp(y(1))]';
bc = @(lb,rb) [lb(1) rb(1)]';
meshinit = linspace(0, 1, 3);
solinit = bvpinit(meshinit, [0 0]); 
sol = bvp4c(dydx, bc, solinit);
x = linspace(0,1);
y = deval(sol, x, 1);
u = y(1,:);
plot(x,u);


% Page 213  Poisson Equation

n = 100;  % n x n inner grid points
h = 1/(n+1);
u = zeros(n+2);  % recall: square matrix, unlike Python 
A = gallery('poisson', n);  
b = ones(n*n, 1);           
u_inner = (A/h^2) \ b;  
u(2:n+1, 2:n+1) = reshape(u_inner, n, n);  
lin = linspace(0, 1, n+2); 
[x y] = meshgrid(lin);  % grid 
mesh(x, y, u, 'LineWidth', 2);