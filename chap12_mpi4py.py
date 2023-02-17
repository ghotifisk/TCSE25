# Page 272 hello.py

print("Hello World")


# Page 273 hello_from.py

from mpi4py import MPI
comm = MPI.COMM_WORLD
print("Hello World from process", comm.rank, "of", comm.size)


# Page 273 Example 12.2 hello_to.py

from mpi4py import MPI
comm = MPI.COMM_WORLD
if comm.rank == 0: 
    comm.send("Hello World", dest=1)
if comm.rank == 1:
    msg = comm.recv(source=0)
    print("Message received:", msg)
    
    
# Page 274 Deadlock dl.py
     
from mpi4py import MPI
comm = MPI.COMM_WORLD
msg = comm.recv(source = comm.size - 1 - comm.rank)
comm.send(22, dest = comm.size - 1 - comm.rank)
print(msg)


# From now on always include:

from mpi4py import MPI

# and wherever needed: 

size = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank
send = MPI.COMM_WORLD.send
recv = MPI.COMM_WORLD.recv
reduce = MPI.COMM_WORLD.reduce
bcast = MPI.COMM_WORLD.bcast
Scatter = MPI.COMM_WORLD.Scatter
Gather = MPI.COMM_WORLD.Gather
Scatterv = MPI.COMM_WORLD.Scatterv


# Page 275 Example 12.3 Summation sum.py

n = 100
assert n % size == 0    
k = rank * n // size + 1
l = (rank + 1) *  n // size + 1
sm_loc = sum(range(k, l))
if rank == 0:                  
    sm = sm_loc                
    for i in  range(1, size):  
        sm += recv(source=i)   
else: send(sm_loc, dest=0)     
if rank == 0: print(sm)        


# Page 276 Example 12.4 "reduce" 

n = 100
assert n % size == 0    
k = rank * n // size + 1
l = (rank + 1) *  n // size + 1
sm_loc = sum(range(k, l))
sm = reduce(sm_loc)
if rank == 0: print(sm)        


# Page 276 Example 12.4 "bcast"

q_loc = None
sm_loc = None
if rank == 0: 
    n = 100
    assert n % size == 0 
    q_loc = n//size
q = bcast(q_loc)
k =  rank*q + 1
l = (rank + 1)*q + 1
sm_loc = sum(range(k,l))
if rank == 0:                  
    sm = sm_loc                
    for i in  range(1, size):  
        sm += recv(source=i)   
else: send(sm_loc, dest=0)     
if rank == 0: print(sm)        


# Page 277 Example 12.6 Load Balancing 

tup = None                      
if rank == 0: 
    n = 100
    q = n//size; u = q*size; r = n % size
    tup = q, u, r               
q, u, r = bcast(tup)
k =  rank*q + 1
l = (rank + 1)*q + 1
sm = sum(range(k,l))
if rank < r: sm += u+1  + rank  
sm = reduce(sm) 
if rank == 0: print(sm)


# Page 278 Example 12.7 Midpoint Rule

def f(x): return 4/(1 + x*x)
tup = None
if rank == 0:
    n = 100_000
    a = 0.0; b = 1.0; h = (b - a)/n
    tup = n, a, h
n, a, h = bcast(tup)
add = 0.0
for i in range(rank, n, size):  
    x = a + h*(i + 0.5)
    add += f(x)
int_loc = h*add
int = reduce(int_loc)
if rank == 0: print(int)


# Page 280 Trapezoidal Rule

def f(x): return 4/(1 + x*x)
tup = None
if rank == 0:             
    a = 0; b = 1; n = 100_000
    h = (b - a)/n
    from numpy import linspace
    x = linspace(a, b, n+1)
    q = n//size
    tup = q, x, h
q, x, h = bcast(tup)
k = rank*q; l = k + q 
int_loc = h/2*(f(x[k]) + 2*sum(f(x[k+1: l])) + f(x[l]))
int = reduce(int_loc)
if rank == 0: print(int)   


# Page 280, 281 Dot Product Evenly Divided

from numpy import array, zeros
v = w = n_loc = None      
if rank == 0: 
    n = 12               
    assert n % size == 0  
    v = array([float(i) for i in range(1, n+1)])
    w = array([float(i) for i in range(n, 0, -1)])
    n_loc = n//size           
n_loc = bcast(n_loc)
v_loc = zeros(n_loc); w_loc = zeros(n_loc)
Scatter(v, v_loc); Scatter(w, w_loc)  # note capital letter S, see below
dot_loc = v_loc @ w_loc  # recall: @ denotes matrix multiplication
dot = reduce(dot_loc)
if rank == 0: print(dot)


# Page 282 Dot Product with Load Balance

from numpy import array, zeros
v = w = sendcts = None
if rank == 0:
    n = 10   
    v = array([float(i) for i in range(1, n+1)])
    w = array([float(i) for i in range(n, 0, -1)])
    n_loc = n//size  # even divisibility not required
    sendcts = [n_loc for i in range(size)]
    for i in range(n % size): sendcts[i] += 1 
sendcts = bcast(sendcts)  
v_loc = zeros(sendcts[rank]); w_loc = zeros(sendcts[rank])  
Scatterv([v, sendcts, MPI.DOUBLE], v_loc)
Scatterv([w, sendcts, MPI.DOUBLE], w_loc)
dot_loc = v_loc @ w_loc
dot = reduce(dot_loc)
if rank == 0: print(dot)  


# Page 284 Laplace Equation

if rank == 0:
    from numpy import linspace, zeros  
    n = 10; m = n//2                   
    u = zeros((n,n))                   
    x = linspace(0, 1, n)
    u[0,:] = x*(1-x)                   
    ul = u[:, :m]                     
    ur = u[:, m:]                     
    send(ur, dest=1)                   
if rank == 1: ur = recv(source=0)      
def stencil_it(u):  
    u_it = u.copy()                    
    r, c = u.shape                     
    for i in range(1, r-1):                      
        for j in range(1, c-1):        
             u_it[i,j] = (u[i+1,j] + u[i-1,j] 
                        + u[i,j+1] + u[i,j-1]) / 4 
    return u_it
from numpy import column_stack, delete    
iter = 40
for i in range(iter):
    if rank == 0:
        send(ul[:, -1], dest=1)         
        v = recv(source=1) 
        ul_v = column_stack([ul, v])           
        ul_v = stencil_it(ul_v)         
        ul = delete(ul_v, -1, 1)        
    if rank == 1:     
       send(ur[:, 0], dest=0)           
       v = recv(source=0)
       v_ur = column_stack([v, ur])      
       v_ur = stencil_it(v_ur)
       ur = delete(v_ur, 0, 1)               
if rank == 1: send(ur, dest=0)
if rank == 0: 
    ur = recv(source=1)
    u = column_stack([ul, ur])   
if rank == 0:
    from matplotlib.pyplot import figure, show 
    fig = figure()
    ax = fig.add_subplot(projection='3d')
    from numpy import meshgrid
    x, y = meshgrid(x,x)  # linspace from line 5
    ax.plot_surface(x, y, u, 
        rstride=1, cstride=1, cmap='jet', linewidth=0)                                       
    show() 
    
    
# Page 287 Conjugate Gradient

from numpy import array, zeros, sqrt
A = n = p = None
if rank == 0:
    A = array([[  9.,   3.,  -6.,  12.], [  3.,  26.,  -7., -11.],
               [ -6.,  -7.,   9.,   7.], [ 12., -11.,   7.,  65.]])
    b = array([ 18.,  11.,   3.,  73.])
    n = len(b) 
    x = zeros(n)          
    r = b.copy(); p = r.copy()           
    rs_old = r @ r     
n = bcast(n)
n_loc = n//size     
A_loc = zeros((n_loc, n))  
Scatter(A, A_loc)
for i in range(n):       
    p = bcast(p)         
    Ap_loc = A_loc @ p   
    Ap = zeros(n)        
    Gather(Ap_loc, Ap)    
    if rank == 0: 
        alpha = rs_old / (p @ Ap)        
        x += alpha*p                   
        r -= alpha*Ap                 
        rs_new = r @ r                             
        if sqrt(rs_new) < 1e-10: break   
        p = r + (rs_new / rs_old)*p    
        rs_old = rs_new         
if rank == 0: print(x) 