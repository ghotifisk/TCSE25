# Page 309 Basics

using Distributed
addprocs(8);
nprocs()    
procs()     
nworkers()  
workers()   


# Page 310 Point-to-Point Communication

r = remotecall(sqrt, 2, 2.0);
fetch(r)  

r = @spawnat 2 sqrt(2.0);


# Page 310 Example 14.1 Babylonian Approximation

sq = @spawnat 3 begin 
    r = 1; eps = 1.e-6
    while abs(r^2 - 2) > eps  r = (r + 2/r)/2 end
    return r end;
fetch(sq) 
 

r = @spawn sqrt(2.0);


# Page 310 Example 14.2 Distributed Loops

@distributed (+) for i = 1:10  1/2^i end  

@distributed (*) for i = 1:10  i end   


# Page 311 Remark Load Balancing

@distributed for i = 1:10 println("index $i processed here") end;


# Page 311 Example 14.3 Vector Dot Product

v = collect(1. : 10.);  w = collect(10.: -1.: 1.);  
@distributed (+) for i = 1:10 v[i] * w[i] end       


# Page 311 Example 14.4 Monte Carlo Pi Approximation

function pi_loc(samples) 
# @everywhere function pi_loc(samples)    
    hits = 0
    for _ = 1:samples
        x, y = rand(), rand()
        if(x^2 + y^2 <= 1) hits += 1 end end
    return 4*hits/samples end  
     
samples = 1_000_000_000;
@elapsed pi_loc(samples)  

function pi_dist(samples, p)
    r = @distributed (+) for i = 1:p  pi_loc(samples/p) end
    return r/p end
p = nworkers()
pi_dist(samples, p)  

@elapsed pi_dist(samples, p)  # ans: 0.332873125


# Page 312 Example 14.5

samples = 1_000_000_000;
hits = @distributed (+) for _ = 1:samples 
           Int(rand()^2 + rand()^2 <= 1) end;
mypi = 4*hits/samples

@elapsed begin
 # code goes here
end


# Page 313 Example 14.6 Fibonacci Function

@everywhere function fib(n) 
    return (n < 3) ? 1 :  fib(n-1) + fib(n-2) end 
    
@everywhere function fib_parallel(n)
    if (n < 40) return fib(n) end
    x = @spawn fib_parallel(n-1)
    y = fib_parallel(n-2)
    return fetch(x) + y end
    
@elapsed fib(50)           # ans: 52.752892042
@elapsed fib_parallel(50)  # ans:  7.603331375


# Page 313 Shared Arrays

using SharedArrays  

# Page 313 Example 14.7 Different Array Behavior

v = [0, 0];
sv = convert(SharedVector, v);
@spawn sv[1] = v[1] = 1;
println("$v, $sv")  


# Page 314 Example 14.8 Laplace Equation

n = 10;
S = SharedMatrix{Float64}((n,n));
x = collect(range(0, stop=1, length=n));
S[1,:] = x .* (1 .- x);
for _ = 1 : 40 
    T = copy(S)
    @sync @distributed  for i in 2:n-1 
       for j in 2:n-1  
          S[i,j] = (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) / 4 
end end end 

using Pkg; Pkg.add("Plots")  
using Plots
plot(x, x, S, seriestype=:wireframe)  


# Page 315 Distributed Arrays

import Pkg;                   # Julia's package manager                       
Pkg.add("DistributedArrays")  # only required once


# Page 315 Example 14.9 

@everywhere using DistributedArrays
u = collect(1:10); 
du = distribute(u);
for i in workers() 
    r = @spawnat i localpart(du)
    lu = fetch(r)
    println("$i: $lu") end;
    
      
# Page 316 Example 14.10 Vector Dot Product

function dot_dist(u,v)
    s = 0.0 
    du = distribute(u); dv = distribute(v)
	for i in workers() 
        r = @spawnat i localpart(du)'*localpart(dv)  # transpose
        s += fetch(r) end
    return s end
u = collect(1. : 10.); v = collect(10.: -1: 1.0);
dot_dist(u,v) 


# Page 316, 317 Example 14.11 Matrix-Vector Product

function mat_vec_dist(A,v)
    w = Float64[]                   # empty vector
    m = min(size(A,1), nworkers())  # see remark above
    dA = distribute(A, dist=[m, 1])
    for i = 2: m+1
        r = @spawnat i localpart(dA)*v
        w = vcat(w, fetch(r))       # append to w  
    end
    return w end
A = [9. 3. -6. 12.; 3. 26. -7. -11.; 
     -6. -7. 9. 7.; 12. -11. 7. 65.];
v = [1 1 1 1]';
mat_vec_dist(A,v)'  


# Page 317 Example 14.12 Conjugate Gradient 

A = [9. 3. -6. 12.; 3. 26. -7. -11.; 
     -6. -7. 9. 7.; 12. -11. 7. 65.];
b = [18.; 11.; 3.; 73.];
n = length(b) 
x = zeros(n)                          
p = r = b
rs_old = dot_dist(r,r)                
for _ in 1 : n 
    Ap = mat_vec_dist(A,p)            
    alpha = rs_old  / dot_dist(p,Ap)  # using dot_dist
    global x += alpha*p
    global r -= alpha*Ap
    rs_new = dot_dist(r,r)            
    if sqrt(rs_new) < 1e-10 break end
    global p = r + (rs_new / rs_old)*p
    global rs_old = rs_new end
println(x)                            