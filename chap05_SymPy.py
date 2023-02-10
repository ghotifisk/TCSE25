# Page 95

import scipy, sympy 
scipy.sqrt(8)  # Out: 2.8284271247461903
sympy.sqrt(8)  # Out: 2*sqrt(2)

from sympy import *  # from here on always assumed
init_printing()


# Page 96

sqrt(8)

a = Rational(1, 3); type(a)  # Out: sympy.core.numbers.Rational
b = 9*a; type(b)             # Out: sympy.core.numbers.Integer

x = Symbol('x')
x        # Out x
type(x)  # Out: sympy.core.symbol.Symbol

sqrt(8).evalf(50)

x, y, z = symbols('x y z')  # note lower-case initial letter s 


# Page 97

from sympy.abc import x, y, z
from sympy.abc import *

x + y + x - y  # Out: 2*x


# Page 97 sympify

str_expr = 'x**2 + 3*x - 1/2'
type(str_expr)  # Out: str
expr = sympify(str_expr)
expr        # Out: x**2 + 3*x - 1/2
type(expr)  # Out: sympy.core.add.Add

f1 = 1/3              # type: float
f2 = sympify(1/3)     #       sympy.core.numbers.Float  
f3 = sympify(1)/3     #       sympy.core.numbers.Rational
f4 = sympify('1/3')   #       sympy.core.numbers.Rational


# Page 97 Value Assignments

((x + y)**2).subs(x,1)  # Out: (y + 1)**2
((x + y)**2).subs(x,y)  # Out: 4*y**2

expr = x**3 + 4*x*y - z
expr.subs([(x, 2), (y, 4), (z, 0)])  # Out: 40


# Page 98 Simplify

(x + x*y)/x  # Out: (x*y + x)/x
simplify((x + x*y)/x)  # Out: y + 1


# Page 98 trigsimp

trigsimp(sin(x)/cos(x))  # Out: tan(x)


# Page 99 expand

expand((x + y)**3)  # Out: x**3 + 3*x**2*y + 3*x*y**2 + y**3
expand(cos(x + y), trig=True)  # Out: -sin(x)*sin(y) + cos(x)*cos(y)


# Page 99 factor, collect

factor(x**3 - x**2 + x - 1)  # Out: (x - 1)*(x**2 + 1) 
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
collect(expr, x)  # Out: x**3 + x**2*(-z + 2) + x*(y + 1) - 3


# Page 99 Functions

expr = x**2
def f(x): return x**2   
def g(x): return expr 
def h(x_var): return expr.subs(x, x_var)
f(1), g(1), h(1)  # Out: (1, x**2, 1)

a = Symbol('a')
expr = a*x**2
def f(varx): return expr.subs(x, varx)
f(2)  # Out: 4*a


# Page 99 PLot

plot(expr.subs(a, 2), (x, 0, 1))
graph = plot(expr.subs(a, 2), (x, 0, 1))
graph.save('graph.pdf')


# Page 100 Example 5.1 lambdify

import sympy as sym
x = sym.Symbol('x')
expr = x**2*sym.sin(x)
sym.plot(expr, (x, 0, sym.pi))
f = sym.lambdify(x, expr)
import numpy as np
x = np.linspace(0, np.pi)
import matplotlib.pyplot as plt
plt.plot(x, f(x))


# Page 101 Equations 

from sympy import *
x = Symbol('x')
solve(Eq(x**2, 1), x)  # Out: [-1, 1]
solve(x**2 - 1, x)
solve(x**2 - 1)

x = Symbol('x')
solve(x**4 - 1, x)  # Out: ￼[‐1, 1, ‐I, I] 
x = Symbol('x', real=True)
solve(x**4 - 1, x)  # Out: [‐1, 1]  


# Page 102 Equation Systems 

eqn1 = Eq(x + 5*y, 2); eqn2 = Eq(-3*x + 6*y, 15)
sol = solve([eqn1, eqn2], [x,y]); sol  # Out: {x:-3 1, y: 1}  
eqn1.lhs.subs(sol) == eqn1.rhs         # Out: True


# Page 102 solveset

solve(x**2 - x)     # Out: [0, 1]
solveset(x**2 - x)  # Out: {0, 1}

solve(x - x)     # Out: []
solveset(x - x)  # Out: S.Complexes

solveset(x - x, x, domain=S.Reals)  # Out: Reals
solveset(sin(x) - 1, x, domain=S.Reals) 


# Page 103 LaTeX

print(latex(solveset(sin(x) - 1, x, domain=S.Reals)))


# Page 103 linsolve 

linsolve([x+y+z-1, x+y+2*z-3], (x,y,z))  # Out: {(-y-1, y, 2)}


# Page 103 Matrices and Vectors

M = Matrix([[1, -1], [3, 4], [0, 2]])
v = Matrix([1, 2, 3])  # column vector

M = Matrix([[1, 2, 3], [3, 2, 1]])
N = Matrix([0, 1, 1])
M*N

Matrix([[1, 2,], [3, 4]])**-1


# Page 104 Hilbert Matrix

def f(i,j): return 1/(1+i+j)
Matrix(3, 3, f)


# Page 104 Example 5.2 Vector Space Base

v1 = Matrix([1, 1, 1])  / sqrt(3)
v2 = Matrix([1, 0, -1]) / sqrt(2)
v3 = Matrix([1, -2, 1]) / sqrt(6)
v1.dot(v1)  # Out: 1
v1.dot(v2)  # Out: 0
a, b, c = symbols('a b c')
x = Matrix([a,b,c])
y = v1.dot(x)*v1 + v2.dot(x)*v2 + v3.dot(x)*v3
simplify(y)


# Page 105 Example 5.3 Vector Space Basis

from sympy import symbols, sympify, solve
x, a, b, c, c1, c2, c3 = symbols('x a b c c1, c2, c3')
p1 = 1
p2 = x - Rational(1)/2  # Rational to ensure correct division
p3 = x**2 - x + Rational(1)/6
q = a*x**2 + b*x + c
expr = q - (c1*p1 + c2*p2 + c3*p3)
solve(expr, [c1, c2, c3])


# Page 106 Example 5.4 Linear Independence

A = Matrix([[1, 0, 2], [0, 1, 1], [1, 2, -1]]) 
A.det()
A.columnspace()
A.nullspace()
b = Matrix([8, 2, -4]) 
u = A.LUsolve(b); u  
Eq(A*u, b)  # Out: True


# Page 106 Example 5.5 Non-Empty Kernel

C = Matrix([[1, 3, -1, 2], [0, 1, 4, 2], 
                [2, 7, 2, 6], [1, 4, 3, 4]])
C.nullspace()


# Page 107 Eigenvectors and Eigenvalues

M = Matrix([[3, -2, 4, -2], [5, 3, -3, -2],\
            [5, -2, 2, -2], [5, -2, -3, 3]])
M.eigenvals()  # Out:  {3: 1, -2: 1, 5: 2}
lamda = Symbol('lamda')   # spelling to avoid conflict 
factor(M.charpoly(lamda))
M.eigenvects()
comp0 = M.eigenvects()[0]
val0 = comp0[0]
vec0 = comp0[2][0]
Eq(M*vec0, val0*vec0)  # Out: True


# Page 108 5 x 5 Hilbert Matrix

def f(i,j): return 1/(1+i+j)
H5 = Matrix(5, 5, f)
H5.eigenvals()  


# Page 108 Limits

x = Symbol('x')
limit(sin(x)/x, x, 0)  # Out: 1
limit(1/x, x, 0, '+')  # Out: oo 
limit(1/x, x, 0, '-')  # Out: -oo


# Page 108 Differential Calculus

diff(sin(2*x), x)  # Out: 2*cos(2*x)
diff(tan(x), x)    # Out: tan(x)**2 + 1
h = Symbol('h')
limit((tan(x+h) - tan(x))/h, h, 0)  # Out: tan(x)**2 + 1 


# Page 109 Suffix Notation

expr = sin(2*x)
expr.diff(x)  # Out: 2*cos(2*x)


# Page 109 Partial Derivative

y = Symbol('y')
diff(x**4*y, x, y)  # Out: 4*x**3


# Page 109 HigheOrder Derivative

n = 3
diff(sin(2*x), x, n)  # Out: -8*cos(2*x)


# Page 109 Integration

integrate(log(x), x)          # indefinite integral 
integrate(x**3, (x, -1, 1))   # definite integral 
integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo))  # Out: pi


# Page 109  Series Expansion

series(cos(x), x, pi, 4)
series(exp(x), x)
exp(x).series(x)


# Page Example 5.6 Ordinary Differential Equation

u = Function('u')
# u = symbols('u', cls=Function)
ode = Eq(u(x).diff(x), x - u(x))
iv = {u(0): 1} 
sol = dsolve(ode, u(x), ics = iv); sol
plot(sol.rhs, (x,0,5))


# Page 110 Example 5.7 Boundary Value Problem

u = Function('u')
ode = Eq(u(x).diff(x,x), 6*x)
bds = {u(0): 0, u(1): 1}
sol = dsolve(ode, u(x), ics = bds); sol


# Page 111 Example 5.8 Bratu  Equation

u = Function('u')
ode = Eq(u(x).diff(x,x), -exp(u(x)))
bd = {u(0): 0, u(1): 1}
sol = dsolve(ode, u(x), ics = bd); sol

sol = dsolve(ode, u(x)); sol


# Page 113 Galerkin Method

from sympy import *
x = Symbol('x')
phi0 = x*(1 - x) 
phi1 = sympify('x*(1/2 - x)*(1 - x)')
phi2 = sympify('x*(1/3 - x)*(2/3 - x)*(1 - x)')
phi3 = sympify('x*(1/4 - x)*(1/2 - x)*(1/4 - x)*(1 - x)')
phi = [phi0, phi1, phi2, phi3]
a = lambda u,v: -integrate(diff(u,x,2)*v, (x, 0, 1))
A = Matrix(4, 4, lambda i, j: a(phi[i], phi[j]))
f = x*(x + 3)*exp(x)
L = lambda v: integrate(f*v, (x, 0, 1))
b = Matrix([L(phi[j]) for j in range(4)])
c = A.LUsolve(b)
u = sum(c[i]*phi[i] for i in range(4))
plot(u, (x, 0, 1))


# Page 114 Exact Solution

from sympy import symbols, solve
C1, C2 = symbols('C1, C2')
temp = integrate(-f) + C1       # here var x unique from context
expr = integrate(temp, x) + C2  # ... here x needed
sol = solve([expr.subs(x, 0), expr.subs(x, 1)], [C1, C2])
u_e = expr.subs([(C1, sol[C1]), (C2, sol[C2])])
plot(u_e, (x, 0, 1))

plot(u - u_e, (x, 0, 1))
