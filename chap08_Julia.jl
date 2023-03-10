# Page 158 Example 8.2 Hello World

println("Hello World")  


# Page 161 Example 8.3 Collatz Problem

n = 100
while (n > 1) global n = (n % 2 == 0) ? div(n, 2) : 3n + 1 end
println("reached 1")


# Page 161 Example 8.4 Chessboard and Rice Problem

grains = UInt64(0)    
fieldval = UInt64(0)  
for fieldno in 1:64   
    global fieldval = fieldno == 1 ? 1 : fieldval *= 2
    global grains += fieldval end
println(grains)


# Page 162 Example 8.5 Monte Carlo Method

const samples = 10_000_000  
hits = 0
for _ = 1 : samples
    x, y  = rand(), rand()
    d = x*x + y*y
    if d <= 1 global hits += 1 end end
println(4*(hits/samples))  


# Page 163 Example 8.6 Factorial Function

function factorial(n)   
    res = 1 
    for i = 1:n res *= i end
    return res end
factorial(5)  


# Page 163 Example 8.7 Return Value Pair

function swap(n,m) return (m,n) end
swap(1,2)  


# Page 164 Example 8.8  Ackermann Function

function ack(x,y)
    if x == 0 return y + 1 end
    if y == 0 return ack(x-1, 1) end
    return ack(x-1, ack(x, y-1)) end
    

# Page 164 Example 8.9  Fibonacci Sequence

fib(n) = (n == 1 || n == 2) ? 1 : fib(n-1) + fib(n-2)


# Pagee 164 Example 8.10  Mutually Dependent Functions

f(n) = n < 0 ? 22 : g(n); 
g(n) = f(n-1);
f(4)  


# Page 164 Example 8.11  Pointwise Derivative

# function ddx(f, x) 
function ddx(f::Function, x::Float64)
    h = 1.e-14
    return (f(x+h) - f(x)) / h  end
g(x) = 4x*(1 - x);  
ddx(g, 0.5)         


# Page 164 Example 8.12  Derivative Function

function ddx(f)  
# function ddx(f::Function)
    h = 1.e-14
    function f_prime(x) return (f(x+h) - f(x)) / h end
    return f_prime end
ddx(g)(0.5)       
g_prime = ddx(g);
g_prime(0.5)      

ddx  


# Page 168 Example 8.14 Mutating Function 

function swap!(v) temp = v[1]; v[1] = v[2]; v[2] = temp end
v = [1, 2]
swap!(v); println(v)  


# Page 169 Example 8.15 Filtering

v = collect(1: 5)
filter(x -> x % 2 == 0, v)  
println(v)                  
filter!(x -> x % 2 == 0, v) 
println(v)                  


# Page 169 Example 8.16 Sieve of Eratosthenes

n = 30                  
L = collect(2:n)        
P = Int[]       
while L != [] 
    p = L[1]     
    push!(P, p) 
    for i in L  
        if i % p == 0  filter!(x -> x != i, L) end end end
println(P)


# Page 170 Example 8.17 Quicksort

function qsort(lst::Vector{Int})
    if lst == [] return Int[] end
    p = lst[1]
    sml = qsort([x for x in lst[2: end] if x < p])
    grt = qsort([x for x in lst[2: end] if x >= p])
    return vcat(sml, [p], grt) end
testList = [4, 5, 7, 3, 8, 3]
println(qsort(testList))


# Page 171 Example 8.18 Linked List

v = [0, 2, 0, 0, 8, 10, 0, 14, 16, 18]
struct node index::Int; value::Int end
sp = node[]
for i in 1 : length(v)
    if v[i] != 0 push!(sp, node(i, v[i])) end end
for nd in sp print("($(nd.index), $(nd.value)) ") end


# Page 173 Example 8.19 Fraction Structure

struct fract num::Int; den::Int end
a = fract(3, 4); b = fract(1, 2)
add(x::fract, y::fract) = 
    fract(x.num*y.den+x.den*y.num, x.den*y.den) 
c = add(a,b) 
import Base: + 
+(x::fract, y::fract) = 
    fract(x.num*y.den + x.den*y.num, x.den*y.den) 
ans: + (generic function with 209 methods)
a + b 
import Base: ==
==(x::fract, y::fract) = x.num*y.den == x.den*y.num
a == b            
b == fract(2, 4) 


# Page 175 Linear Equations

A = [9. 3. -6. 12.; 3. 26. -7. -11.; 
    -6. -7. 9. 7.; 12. -11. 7. 65.]
b = [ 18.;  11.;  3.;  73.]
x = A \ b   
println(x)  


# Page 176 Example 8.20 Conjugate Gradient

# include A and b from above
n = length(b)
x = zeros(n)                              
p = r = b
rs_old = r'*r                      
for _ in 1:n  
    Ap = A*p                       
    alpha = rs_old / (p'*Ap)           
    global x += alpha*p
    global r -= alpha*Ap
    rs_new = r'*r                     
    if sqrt(rs_new) < 1e-10 break end
    global p = r + (rs_new / rs_old)*p
    global rs_old = rs_new end
println(x)


# Page 178 Example 8.21 Poisson Matrix

using LinearAlgebra
function poisson(n::Int)
    v = 4*ones(n)     
    w = ones(n-1)
    D = SymTridiagonal(v, -w) 
    sD = SymTridiagonal(zeros(n), w)
    Id = Diagonal(ones(n))  
    A = kron(Id, D) + kron(sD, -Id) 
    return A end
A = poisson(3)


# Page 178 Example 8.22 Sparse Poisson Matrix

using SparseArrays  
spA = sparse(A)     


# Page 179 Example 8.23 Speed Comparison

n = 100
A = poisson(n)
b = ones(n*n)
@elapsed A \ b    
spA = sparse(A)
@elapsed spA \ b  


# Page 179 Ordinary Differential Equations

import Pkg;                                            
Pkg.add("DifferentialEquations")  
using DifferentialEquations       


# Page 180  Example 8.24 Simple IVP

dudx(u, p, x) = x - u  
epts = (0.0, 5.0)      
u0 = 1.0               
prob = ODEProblem(dudx, u0, epts)
sol = solve(prob)
Pkg.add("Plots")
using Plots
plot(sol, leg=false)  
plot!(sol.t, t-> t-1+2*exp(-t), seriestype=:scatter, leg=false)


# Page 181 Example 8.25 Lotka-Volterra

p = (1.0, 0.2, 0.5, 0.2);
tspan = (0.0, 4pi); u0 = [1.0, 2.0];
function lotvol!(dudt, u, p, t) 
    dudt[1] =  p[1]*u[1] - p[2]*u[1]*u[2]
    dudt[2] = -p[3]*u[2] + p[4]*u[1]*u[2] end
prob = ODEProblem(lotvol!, u0, tspan, p);
sol = solve(prob);
plot(sol)


# Page 181 Example 8.26 Second Order ODE Oscillation

function u!(dudt, u, p, t) dudt[1] = u[2]; dudt[2] = -u[1] end
tspan = (0.0, 2pi); u0 = [0.0, 1.0]
prob = ODEProblem(u!, u0, tspan);
sol = solve(prob)
plot(sol)


# Page 183 Example 8.27 Simple BVP

function u!(dudx, u, p, x) dudx[1] = u[2]; dudx[2] = 6x end
bds = (0.0, 1.0)
function bc!(residual, u, p, x) 
    residual[1] = u[1][1]        
    residual[2] = u[end][1] - 1 end
bvp = TwoPointBVProblem(u!, bc!, [0, 0], bds)
sol = solve(bvp, MIRK4(), dt=0.05); 
plot(sol)


# Page 184 Example 8.28 Bratu Equation 

function u!(dudx, u, p, x) dudx[1]=u[2]; dudx[2] = -exp(u[1]) end
bds = (0.0, 1.0)
function bc!(residual, u, p, x) 
    residual[1] = u[1][1]        
    residual[2] = u[end][1] end
solinit = [0, 0]  # for second solution set solinit = [3, 0]
bvp = TwoPointBVProblem(u!, bc!, solinit, bds) 
sol = solve(bvp, MIRK4(), dt=0.05)
plot(sol)  


# Page 185 Poisson Equations Dirichlet Boundary Conditions. 

using LinearAlgebra, Plots
function poisson(n::Int)
    v = 4*ones(n)     
    w = ones(n-1)
    D = SymTridiagonal(v, -w) 
    sD = SymTridiagonal(zeros(n), w)
    Id = Diagonal(ones(n))
    A = kron(Id, D) + kron(sD, -Id) 
    return A end   
function poisson_solver(f::Function, g::Function, m::Int)
    x = y = range(0, stop=1, length=m)  
    h = 1/(m-1)
    u = zeros(m,m)
    for i = 1:m, j = 1:m  
        if i ==  1 || i == m || j == 1 || j == m 
            u[i,j] = g(x[i], y[j]) end end
    n = m - 2  
    F = zeros(n,n)
    for i = 1:n, j = 1:n F[i,j] = f(x[i+1], y[j+1]) end  
    F[:,1] += u[2:n+1, 1]   / h^2
    F[:,n] += u[2:n+1, n+2] / h^2
    F[1,:] += u[1, 2:n+1]   / h^2
    F[n,:] += u[n+2, 2:n+1] / h^2
    b = vec(F) 
    A = poisson(n)
    u_inner = A \ b*h*h
    u[2:n+1, 2:n+1] = reshape(u_inner, n, n)  
    return x, y, u
end

f(x,y) = 1.25*exp(x + y/2)
g(x,y) = exp(x + y/2)
sol = poisson_solver(f, g, 30)
plot(sol, seriestype=:wireframe)


# Page 187 Writing, Reading File

fh = open("test.txt", "w")      
write(fh, "The parrot is\n")    
write(fh, "a Norwegian Blue.")  
close(fh)                       

fh = open("test.txt", "r")      
s = read(fh, String)            
println(s)                      

     
# Page 187 Example 8.29 Julia Set
   
const c = -0.8 + 0.156im
fout = open("julia.pgm", "w")
s = 1000
write(fout, "P2\n# Julia Set image\n$s $s \n255\n")
for y in range(2, stop=-2, length=s)      
    for x in range(-2, stop=2, length=s)  
        z = complex(x,y); n = 0 
        while (abs(z) < 2 && n < 255) z = z*z + c; n += 1 end
        write(fout, "$(255-n) ") end 
    write(fout, "\n") end 
close(fout)
