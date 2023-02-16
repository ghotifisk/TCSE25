# Page 325 FEniCS Implementation, One-Dimensional

from dolfin import *
mesh = UnitIntervalMesh(4)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
f = 8
bc = DirichletBC(V, 0, 'on_boundary')
a = u.dx(0)*v.dx(0)*dx
L = f*v*dx
u_sol = Function(V)
solve(a == L, u_sol, bc)
plot(u_sol)  
from matplotlib.pyplot import show
show()


# Page 325 Remark Step by Step

A = assemble(a); b = assemble(L)
A.array(); b.get_local() 
bc.apply(A,b)
A.array(); b.get_local()
u_sol.vector().get_local()


# Page 329 Poisson Equation

from dolfin import *
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V) 
v = TestFunction(V)
f = Constant(-6.0)
g = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
bc = DirichletBC(V, g, 'on_boundary')
a = inner(grad(u), grad(v))*dx
L = f*v*dx
u = Function(V)
solve(a == L, u, bc) 
plot(u, mode="warp")
from matplotlib.pyplot import show
show()


# Page 335 Time-Dependent Poisson Equation

t = 0.; h = 0.1; T = 1.0 
from dolfin import *
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, 'CG', 1)
u_sol = Function(V)  
v = TestFunction(V)
f = Constant(-6.8)
g_str = '1 + x[0]*x[0] + 3.0*x[1]*x[1] + 1.2*t'
g = Expression(g_str, t=0, degree=2)
u_n = project(g,V)  
while t <= T: 
    a = u_sol*v*dx + h*inner(grad(u_sol), grad(v))*dx  
    L = u_n*v*dx + h*f*v*dx
    g.t = t  
    bc = DirichletBC(V, g, 'on_boundary')
    solve(a - L == 0, u_sol, bc)  
    u_n.assign(u_sol)             
    t += h                        
    
plot(u_n , mode="warp")
from matplotlib.pyplot import show
show()


# Page 337 Nonlinear Equation, Picard Iteration

from dolfin import *
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)
def q(u): return (1 + u)**2
def lb(x): return near(x[0], 0)
def rb(x): return near(x[0], 1)
lc = DirichletBC(V, 0, lb)
rc = DirichletBC(V, 1, rb)
bc = [lc, rc]
u_n = project(Constant(0), V)  
u = Function(V) 
eps = 1.0e-6
from numpy import inf; from numpy.linalg import norm
for _ in range(25): 
    F = (q(u_n)*u.dx(0)*v.dx(0))*dx
    solve(F == 0, u, bc) 
    diff = u.vector().get_local() - u_n.vector().get_local()
    d = norm(diff, ord=inf)
    if d < eps: break
    u_n.assign(u) 
    
plot(u_n)  
from matplotlib.pyplot import show
show()


# Page 339 Nonlinear Equation, Direct Solution

from dolfin import *
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)
def q(u): return (1 + u)**2
def lb(x): return near(x[0], 0)
def rb(x): return near(x[0], 1)
lc = DirichletBC(V, 0, lb)
rc = DirichletBC(V, 1, rb)
bc = [lc, rc]
u_n = project(Constant(0), V)  
u = TrialFunction(V)
F = inner(q(u)*grad(u), grad(v))*dx
u = Function(V)
F = action(F,u)
J = derivative(F,u)
u_init = project(Constant(0), V)
u.assign(u_init)
problem = NonlinearVariationalProblem(F, u, bc, J)
solver = NonlinearVariationalSolver(problem)
solver.solve()  

plot(u)  
from matplotlib.pyplot import show
show()


# Page 342 Mixed Dirichlet-Neumann Boundary Conditions

from dolfin import *
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
def dir_boundary(x): return near(x[0], 0) or near(x[0], 1) 
g_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
bc = DirichletBC(V, g_D, dir_boundary)
f = Constant(-6)
g_N = Expression('4*x[1]', degree=1)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g_N*v*ds  
u = Function(V)
solve(a == L, u, bc)

plot(u, mode="warp")
from matplotlib.pyplot import show
show()


# Page 343 Pure Neumann Boundary Conditions

from dolfin import *
mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx + u*v*dx
f = Expression('(1 + 2*pi*pi)*cos(pi*x[0])*cos(pi*x[1])', degree=1)
L = f*v*dx  
u = Function(V)
solve(a == L, u)  
plot(u, mode="warp") 
from matplotlib.pyplot import show        
show()


# Page 346 Stokes Equation

from dolfin import *
mesh = UnitSquareMesh(32,32)
ue = VectorElement('CG', triangle, 2)
pe = FiniteElement('CG', triangle, 1)
W = FunctionSpace(mesh, ue*pe)
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
u_ex = Expression(('cos(pi*x[1])', 'sin(pi*x[0])'), degree=1) 
bc = DirichletBC(W.sub(0), u_ex, 'on_boundary')
f = Expression(('pi*pi*(cos(pi*x[1])-sin(pi*x[0])*cos(pi*x[1]))',  
                'pi*pi*(sin(pi*x[0])-sin(pi*x[1])*cos(pi*x[0]))'), 
                 degree=1)
a = inner(grad(u), grad(v))*dx - p*div(v)*dx + q*div(u)*dx
L = inner(f,v)*dx
w = Function(W)
solve(a == L, w, bc)
(u,p) = w.split(True)

from matplotlib.pyplot import show, figure, colorbar
figure()
v = plot(u)
colorbar(v)
normalize(p.vector())
figure()
plot(p, mode='warp')
show()


# Page 346 Adaptive Mesh Refinement

from dolfin import *
mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, 0, 'on_boundary')
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('100*exp(-100*(pow(x[0]-0.5, 2)\
                 + pow(x[1]-0.3, 2)))', degree=1)   
a = inner(grad(u), grad(v))*dx
L = f*v*dx
u = Function(V)
M = u*dx
tol = 1.e-3  
solve(a == L, u, bc, tol=tol, M=M)

from matplotlib.pyplot import show, figure
figure(); plot(u.root_node(), mode='warp')
figure(); plot(u.leaf_node(), mode='warp')
figure(); plot(mesh)  
# figure(); plot(mesh.leaf_node())  # FEniCS 1.7.2
show()


# Page 351 Mesh Generation 

from dolfin import *
from mshr import *
c1 = Circle(Point(0.0), 1)
c0 = Circle(Point(0.0), .5)
annulus = c1 - c0
mesh = generate_mesh(annulus, 20)
plot(mesh)
from matplotlib.pyplot import show
show()

file = File('annulus.xml')
file << mesh  


# Page 351 Application Example


from dolfin import *
mesh = Mesh('annulus.xml')
V = FunctionSpace(mesh, 'CG', 1)
g_out = Expression('sin(5*x[0]) + cos(5*x[1])', degree=1)
def out_boundary(x, on_boundary): 
    return (x[0]**2 + x[1]**2 > .8 ) and on_boundary
bc = DirichletBC(V, g_out, out_boundary)
f = Constant('0.0')
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = f*v*dx  
u = Function(V)
solve(a == L, u, bc)

plot(u, mode='warp')
from matplotlib.pyplot import show
show()


# Page 351 Manual Mesh Refinement

from dolfin import *
from mshr import *
dom = Rectangle(Point(0,0),Point(1.2, .4))\
      - Circle(Point(.3,.2),.15)                               
mesh = generate_mesh(dom, 8)
mesh_fine = refine(mesh)
markers = MeshFunction('bool', mesh, 2)  
markers.set_all(False)   
for c in cells(mesh):
    if (c.midpoint()[0] < .6): markers[c] = True
mesh_partial_fine = refine(mesh, markers)

from matplotlib.pyplot import show, figure
figure(); plot(mesh)
figure(); plot(mesh_fine)
figure(); plot(mesh_partial_fine)
show()