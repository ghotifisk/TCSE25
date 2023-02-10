# Page 54 Example 4.1 Quadratic Equations

from numpy import sqrt, array
def solveq(a,b,c):
    d = b**2 - 4*a*c
    if d < 0: return 'No real-valued solutions'
    w = sqrt(d)  
    return 1.0/(2*a)*(-b + array([w, -w]))
print(solveq(4, -12, -40))  # Out: [5. -2.]


# Page 59 Example 4.2 Poisson Matrix

from numpy import ones, diag, eye, kron, identity
n = 3
w = ones(n-1); sD = diag(w, 1) + diag(w, -1)
I = identity(n)
D = 4*I - sD
A = kron(I, D) + kron(sD, -I)   # Poisson matrix


# Page 62 Example 4.3 Conjugate Gradient Method

from numpy import array, zeros, sqrt
A = array([[  9.,   3.,  -6.,  12.], [  3.,  26.,  -7., -11.],
           [ -6.,  -7.,   9.,   7.], [ 12., -11.,   7.,  65.]])
b = array([ 18.,  11.,   3.,  73.])
n = len(b) 
x = zeros(n)          
r = b.copy(); p = r.copy()           # copy to avoid side effects
rs_old = r @ r                       # for (4) and (6) above    
for i in range(n):                   # n steps to exact result
    Ap = A @ p                       # matrix-vector mult.
    alpha = rs_old / (p @ Ap)        # needed for (6) and (7) 
    x += alpha*p                     # ... in (6)
    r -= alpha*Ap                    # ... in (7) 
    rs_new = r @ r                   # update residual           
    if sqrt(rs_new) < 1e-10: break   # desired precision 
    p = r + (rs_new / rs_old)*p      # used in (4)
    rs_old = rs_new                  # prepare for next iteration
print(x)                             # Out: [ 1.  1.  1.  1.]

# Page 65 Example 4.4 Overdetermined Linear Equation System

from numpy import array
from scipy.linalg import cholesky, solve
A = array([[3, -6], [4, -8], [0, 1]])
b = array([-1, 7, 2])     # ".T"  not necessary 
C = A.T @ A 
U = cholesky(C)
z = solve(U.T, A.T @ b )  #  forward substitution
x = solve(U,z)            #  backward substitution
print(x)  # Out: array([ 5., 2.])

# Page 66 Gaussian Normal Equation

from numpy import array
from scipy.linalg import qr, solve, inv
A = array([[3, -6], [4, -8], [0, 1]])
Q, R = qr(A, mode='economic')
b = array([-1, 7, 2])
gne = inv(R) @ Q.T @ b  
print(gne)  # Out: array([ 5.,  2.])


# Page 68 Example 4.8 Sparse Poisson Matrix

from numpy import ones
n = 50
w = ones(n-1) 
from scipy.sparse import diags, identity, kron
sD = diags(w, 1) + diags(w, -1)
I = identity(n)               # overrides numpy identity 
D = 4*I - sD             
A = kron(I,D) + kron(sD, -I)  # overrides numpy kron 


# Pages 69 Example 4.9 Plot Sine Function

from numpy import linspace, pi, sin
xvals = linspace(0, 2*pi, 9)
yvals = sin(xvals)
from matplotlib.pyplot import plot, show
plot(xvals, yvals)
show()


# Page 70 Example 4.10 Detailed Plot

from numpy import exp, linspace
def f(x): return x**2*exp(-x**2)
def dfdx(x): return 2*x*(1 - x**2)*exp(-x**2)
xvals = linspace(0, 3)  # recall: short for linspace(0, 3, 50)
from matplotlib.pyplot import plot, xlabel, ylabel, legend, title
plot(xvals, f(xvals), 'r')
plot(xvals, dfdx(xvals), 'b--')
xlabel('x-axis'); ylabel('y-axis')
legend(['f(x) = x^2 e^(-x^2)', 'd/dx f(x)'])
title('Function and derivative')


# Page 73 Rosenbrock Function, Minimization

from numpy import linspace, meshgrid
from matplotlib.pyplot import figure, show
fig = figure()
ax = fig.add_subplot(projection='3d')
x = linspace(-2, 2, 100); y = linspace(-2, 4, 100)
X, Y = meshgrid(x,y)
Z = (1 - X)**2 + 100*(Y - X**2)**2
ax.plot_surface(X, Y, Z, 
       rstride=1, cstride=1, cmap='jet', linewidth=0) 
show() 
from numpy import array
from scipy.optimize import minimize
def f(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
x0 = array([1.3, 0.7])
res = minimize(f, x0)
print(res.x)     # Out: [0.99999552 0.99999102]
print(f(res.x))  # Out: 2.011505248124899e-11


# Page 75 Euler Method

from numpy import zeros, linspace
u = zeros(100)
n = 100
x = linspace(0, 1, n)
for i in range(n-1): u[i+1] = (u[i] + 2*x[i]/n)
from matplotlib.pyplot import plot
plot(x,u)


# Page 75 Example 4.12 solve_ivp

from scipy.integrate import solve_ivp 
def dudx(x,u): return 2*x
epts = (0, 1) 
u0 = [0]
sol = solve_ivp(dudx, epts, u0)


# Page 76 Example 4.13 solve_ivp

from numpy import linspace
def dudx(x,u): return x - u
xvals = linspace(0, 5, 100)
epts = xvals[0], xvals[-1]
u0 = [1]
from scipy.integrate import solve_ivp 
sol = solve_ivp(dudx, epts, u0, t_eval=xvals) 
from matplotlib.pyplot import plot
plot(xvals, sol.y[0])  # note: xvals = sol.t


# Page 76 Example 4.14 Lotka-Volterra

from numpy import linspace, pi
def dydt(t,y): return [y[0]*(1- y[1]), -y[1]*(1 - y[0])]
y0 = [1, 4]
tvals = linspace(0, 4*pi, 100)
epts = [0, 4*pi]
from scipy.integrate import solve_ivp 
sol = solve_ivp(dydt, epts, y0, t_eval=tvals)
from matplotlib.pyplot import plot
plot(tvals, sol.y[0])
plot(tvals, sol.y[1])


# Page 78 Example 4.15 Pendulum

from numpy import linspace, pi, sin
def dydt(t,y):
    theta, omega = y
    return omega, -0.25*omega - 5.0*sin(theta)
y0 = (pi - 0.1, 0.0)
tvals = linspace(0, 10, 100); tspan = (0, 10)
from scipy.integrate import solve_ivp 
sol = solve_ivp(dydt, tspan, y0, t_eval=tvals)
from matplotlib.pyplot import plot, legend
plot(tvals, sol.y[0], label='theta(t)')
plot(tvals, sol.y[1], label='omega(t)')
legend(loc='best')


# Page 79 Example 4.16 solve_bvp

from numpy import linspace, zeros
def dydx(x,y): return (y[1], 6*x)
def bc(yl, yr): return (yl[0], yr[0]-1)
x_init = [0, 1]
y_init = zeros((2, len(x_init)))
from scipy.integrate import solve_bvp
res = solve_bvp(dydx, bc, x_init, y_init)
x = linspace(0, 1, 100)
u = res.sol(x)[0]
from matplotlib.pyplot import plot
plot(x,u)
# y_init[0] = x_init**3    # exact function value
#y_init[1] = 3*x_init**2  # exact derivative value 


# Page 81 Example 4.17 solve_bvp

def dydx(x,y): 
    z = -2*ones(len(x))
    return (y[1], z)
def bc(yl, yr): return (yl[0], yr[0]-3)
x_init = [0, 5]
y_init = zeros((2, len(x_init)))


# Page 81 Example 4.18 Bratu Equation

from numpy import exp, linspace, zeros
def dydx(x,y): return (y[1], -exp(y[0])) 
def bc(yl, yr): return [yl[0], yr[0]]
x_init = linspace(0, 1, 3)
y_init_1 = zeros((2, len(x_init))); 
y_init_2 = zeros((2, len(x_init)))   
y_init_2[0][1] = 3 
from scipy.integrate import solve_bvp
res_1 = solve_bvp(dydx, bc, x_init, y_init_1)
res_2 = solve_bvp(dydx, bc, x_init, y_init_2)
x = linspace(0, 1, 100)
u_1 = res_1.sol(x)[0]
u_2 = res_2.sol(x)[0]
from matplotlib.pyplot import plot, legend
plot(x, u_1, label='u_1')
plot(x, u_2, label='u_2')
legend()


# Page 84 Homogeneous Poisson Equation

from numpy import zeros, ones, diag, identity, kron
n = 50                    # n x n inner grid points
h = 1/(n+1)               # distance between grid points
u = zeros((n+2,n+2))      # grid points 0-initialized
F = ones((n,n))           # encoding of right-hand side in (1)
w = ones(n-1); sD = diag(w, 1) + diag(w, -1)
I = identity(n)
D = 4*I - sD
A = kron(I,D) + kron(sD, -I)
b = F.flatten(order='F')   # equation right-hand side needs vector
from scipy.linalg import solve
u_inner = solve(A, b*h*h)  # solution of Lin. eq. syst. (5)
u[1:n+1,1:n+1] = u_inner.reshape(n,n, order='F') 
from numpy import linspace, meshgrid
lin = linspace(0, 1, n+2)
x, y = meshgrid(lin,lin) 
from matplotlib.pyplot import figure, show
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, u, rstride=1, cstride=1, 
                         cmap='jet', linewidth=0) 
show()


# Page 86 Sparse Poisson Matrix

from scipy.sparse import csr_matrix
A_csr = csr_matrix(A)
from scipy.sparse.linalg import *
u_inner = spsolve(A_csr, b*h*h)


# Page 86 Time Measurement

import time
start = time.time()
# The solver line goes here
end = time.time()
print(end - start)


# Page 87 Nonhomogeneous Boundary Conditions

def poisson_solver(f,g,m):
    from numpy import ones, zeros, linspace, meshgrid
    from scipy.sparse import diags, identity, kron
    from scipy.sparse.linalg import spsolve  
    n = m-1; h = 1/(n+1)
    lin = linspace(0, 1, n+2)
    u = zeros((n+2, n+2)) 
    for i in range(n+2):   
        u[i,0]   = g(lin[i], lin[0])
        u[i,n+1] = g(lin[i], lin[n+1])
    for j in range(1, n+1): 
        u[0,j]   = g(lin[0], lin[j])
        u[n+1,j] = g(lin[n+1], lin[j])
    F = zeros((n,n))
    for i in range(n):  
        for j in range(n):
            F[i,j] = f(lin[i+1], lin[j+1])
    F[:,0]   += u[1:n+1, 0]   / h**2   
    F[:,n-1] += u[1:n+1, n+1] / h**2
    F[0,:]   += u[0, 1:n+1]   / h**2
    F[n-1,:] += u[n+1, 1:n+1] / h**2
    w = ones(n-1); sD = diags(w, 1) + diags(w, -1)
    I = identity(n)
    D = 4*I - sD
    A = kron(I, D) + kron(sD, -I)
    b = F.flatten(order='F') 
    u_inner = spsolve(A, b*h*h)
    u[1:n+1, 1:n+1] = u_inner.reshape(n,n, order='F') 
    x, y = meshgrid(lin,lin)  # lin defined in line 6 
    from matplotlib.pyplot import figure, show
    fig = figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, u, rstride=1, cstride=1, 
                             cmap='jet', linewidth=0) 
    show()
from numpy import exp
f = lambda x, y:  1.25*exp(x + y/2)
g = lambda x, y:  exp(x + y/2) 
poisson_solver(f, g, 50)


# Page 89 Monte Carlo Method

from random import random
samples = 1_000_000  # input,  '_' to increase readability
hits = 0
for _ in range(samples):
    x, y = random(), random()
    d = x*x + y*y
    if d <= 1: hits += 1
print(4*(hits/samples))  
