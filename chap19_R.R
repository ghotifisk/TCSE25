# Page 471 Example 19.1 (Machine Epsilon)

eps <- 1; n <- 0
while (1 + eps > 1) {
  n <- n + 1
  eps <- eps / 2 }
n  ## 53  


# Page 472 Example 19.2 

eps <- 1; n <- 52  
for (i in 1:n) eps <- eps/2
eps + 1 > 1  ## TRUE
eps <- eps/2
eps + 1 > 1  ## FALSE


# Page 472  Example 19.3 (Collatz Problem) 

n <- 100
while (n > 1) {
  if (n %% 2 == 0) n <- n/2 
  else n <- 3*n + 1 }
print("reached 1")

# n <- ifelse(n %% 2 == 0, n/2, 3*n + 1)


# Page 473 Example 19.4 

fac <- function(n) {
  res <- 1;
  for (i in 1:n) res <- i*res
  return(res) }
fac(5)  ## 120 

# (function(n) {res <- 1; for (i in 1:n) res <- i*res; res}) (5)
## 120


# Page 473  Example 19.5 

recip <- function (f) { function(x) { 1/f(x) } }
r <- recip(exp)
r(1)  ## 0.3678794


# Page 474 Example 19.6 

fac_rec <- function(n) ifelse(n == 0, 1, n * fac_rec(n-1))
fac_rec(5)  ## 120


# Page 476 Example 19.7

s <- 0
t0 <- Sys.time()
for (i in 1:1e6) s <- s + i
t1 <- Sys.time()
t1 - t0  

t0 <- Sys.time()
s <- sum(1:1e6)
t1 <- Sys.time()
t1 - t0  


# Page 478 Example 19.8 (Sieve of Eratosthenes)

n <- 100
L <- 2:n; P <- c()
while (length(L) > 0) {
  p <- L[1]; P <- c(P, p) 
  L <- L[L %% p != 0] }
print(P)



# Page 478 Example 19.9 (Quicksort)

qsort <- function(v) {
  if (length(v) <= 1) return(v)
  p <- v[1]; v <- v[-1]
  sml <- v[v < p]
  grt <- v[v >= p]
  c(qsort(sml), p, qsort(grt)) }
v <- c(4, 5, 7, 3, 8, 3)
qsort(v)  ## 3 3 4 5 7 8



# Page 481 Example 19.10

n <- 3
sD <- matrix(0, n, n)
diag(sD[-n, -1]) <- 1
diag(sD[-1, -n]) <- 1


# Page 481 Example 19.11

A <- matrix(c(4, 3, 0, 3, 4, -1, 0, -1, 4), nrow=3, byrow=TRUE)
L <- A; U <- A
L[upper.tri(L, diag = FALSE)] <- 0
U[lower.tri(U, diag = TRUE)] <- 0


# Page 483 Example 19.12

n <- 3
sD <- matrix(0, n, n)
diag(sD[-n, -1]) <- 1
diag(sD[-1, -n]) <- 1
Matrix(sD)

sD <- Matrix(0, n, n)        
diag(sD[-n, -1]) <- 1  
diag(sD[-1, -n]) <- 1 


# Page 484  Example 19.14 

A <- matrix(c(4, 3, 0, 3, 4, -1, 0, -1, 4), nrow=3, byrow=TRUE)
b <- c(24, 30, -24)
x <- solve(A, b); x  ## 3  4 -5
as.vector(A %*% x)   # returns b


# Page 485 Example 19.15 

A <- matrix(c(4, 3, 0, 3, 4, -1, 0, -1, 4), nrow=3, byrow=TRUE)
b <- c(24, 30, -24)
L <- A; U <- A
L[upper.tri(L, diag = FALSE)] <- 0
U[lower.tri(U, diag = TRUE)] <- 0
Linv <- solve(L)
x <- c(3, 3, 3)
for (i in 1:10) x <- Linv %*% (b - U %*% x) 
as.vector(x)  ## 2.974898  4.020918 -4.994770


# Page 485 Example 19.16 

library(Matrix)
n <- 100
sD <- Matrix(0, n, n)        
diag(sD[-n, -1]) <- 1  
diag(sD[-1, -n]) <- 1  
I <- Diagonal(n, 1)
D <- 4*I - sD
A <- I %x% D + sD %x% -I  # sparse Poisson matrix
b <- rep(1, n^2)
x <- solve(A, b)          # specific solver for sparse matrices


# Page 486 Example 19.17

library(Matrix)
A <- matrix(c(1, 2, 3, 1, 1, 1, 3, 3, 1), nrow = 3, byrow = TRUE)
LU <- lu(A)    # lu function from Matrix package
dec <- expand2(LU)
P <- dec$P; L <- dec$L; U <- dec$U
P %*% L %*% U  # returns A as "dgeMatrix"


# Page 487 Example 19.18

A <- matrix(c(11,12,7,12,14,10,7,10,11), nrow = 3, byrow = TRUE)
isSymmetric(A)   ## TRUE
eigen(A)$values  ## 31.71590754  4.25444827  0.02964419
U <- chol(A)
isTriangular(U, upper = TRUE)  ## TRUE
t(U) %*% U  # returns A


# Page 487 Example 19.19 (Method of Least Squares)

A <- matrix(c(3, -6, 4, -8, 0, 1), nrow = 3, byrow = TRUE)
b <- c(-1, 7, 2)
B <- t(A) %*% A               # symmtric positive definite
U <- chol(B)                  # upper triangle
z <- solve(t(U), t(A) %*% b)  # forward substitution
x <- solve(U,z)               # backward substitution
as.vector(x)  ## 5 2


# Page 487 Example 19.20

A <- matrix(c(12,-51,4,6,167,-68,4,-24,-41), nrow=3, byrow=TRUE)
qr_decomp <- qr(A)
Q <- qr.Q(qr_decomp)
R <- qr.R(qr_decomp)
Q %*% R     # returns A
Q %*% t(Q)  # returns identity matrix


# Page 488 Example 19.21 

A <- matrix(c(3, -6, 4, -8, 0, 1), nrow = 3, byrow = TRUE)
b <- c(-1, 7, 2)
qr_decomp <- qr(A)
Q <- qr.Q(qr_decomp)     # returns 3 x 2 matrix
R <- qr.R(qr_decomp); R  # returns 2 x 2 square matrix:


# Page 488 Example 19.22

x <- seq(0, 1, length = 100)
plot(x, x^2, type = "l", xlab="", ylab="")
lines(x, 2*x)


# Page 489 Example 19.23

x <- seq(0, 1, length = 100)
matplot(x, cbind(x^2, 2*x), type = "l", xlab="", ylab="")


# Page 489 Definition 19.24, Example 19.2

ddx <- function(f) {
  h <- 1e-6
  function(x) (f(x+h) - f(x)) / h }

d_atan <- ddx(atan)
d2_atan <- ddx(d_atan)
x <- seq(-4, 4, length = 100)
matplot(x, cbind(atan(x), d_atan(x), d2_atan(x)), 
        type = "l", lty = 1, xlab="", ylab="")
        
        
# Page 490 Example 19.26

expr <- expression(atan(x))
d_expr <- D(expr, "x"); d_expr  ## 1/(1 + x^2) 
df <- function(x) eval(d_expr)
d2_expr <- D(d_expr, "x"); d2_expr  ## -(2 * x/(1 + x^2)^2)
d2f <- function(x) eval(d2_expr)
x <- seq(-4, 4, length = 100)
matplot(x, cbind(atan(x), df(x), d2f(x)), 
        type = "l", lty = 1, xlab="", ylab="")
        
        
# Page 491 Example 19.27

f <- function(x) 1/(1 + x^2)
I <- integrate(f, lower = 0, upper = 1)
I$value      ## 0.7853982 
I$abs.error  ## 8.719671e-15


# Page 491 Example 19.28 (Antiderivative)

aderiv <- function(f) { function(x) {integrate(f, 0, x)$value } }
f <- function(x) 1/(1 + x^2) 
F <- aderiv(f)
F <- Vectorize(F)
g <- ddx(F)      # ddx as in Definition 19.24
x <- seq(-4, 4, length = 100)
plot(x, g(x), type = "l", xlab="", ylab="")
x <- seq(-4, 4, length = 21)
points(x, f(x))  # adds selected points to plot


# Page 492 Composite Midpoint, Trapezoid and Simpson’s Rules

midpoint <- function(f, a, b) { (b - a) * f((a + b)/2) }
trapeze <- function(f, a, b) { (b - a) * (f(a) + f(b))/2 }
simpson <- function(f, a, b) { 
  (b - a) / 6*(f(a) + 4*f((a + b)/2) + f(b)) }
my_integrate <- function(rule, f, a, b, n) {
  pts <- seq(a, b, length = n+1)
  int <- 0
  for (i in 1:n) int <- int + rule(f, pts[i], pts[i+1])
  return(int) } 
f <- function(x) 1/(1 + x^2) 
my_integrate(midpoint, f, 0, 1, 50)  ## 0.7854065
my_integrate(trapeze, f, 0, 1, 50)   ## 0.7853815
my_integrate(simpson, f, 0, 1, 50)   ## 0.7853982


# Page 493 Example 19.29

library(deSolve)
du <- function (x, u, p) { list(x - u) }
xvals <- seq(0, 5, length = 100)
u0 <- 1
sol <- ode(u0, xvals, du)
plot(sol, xlab="", main="")


# Page 494 Example 19.30 (Equation System)

library(deSolve)
lotvol <- function (t, u, p) {
  du1 <-  p[1]*u[1] - p[2]*u[1]*u[2]
  du2 <- -p[3]*u[2] + p[4]*u[1]*u[2]
  list(c(du1, du2)) }
p = c(1, 0.2, 0.5, 0.2)          # parameter values
u0 = c(1, 2)                     # initial values for u1 and u2
tspan = seq(0, 4*pi, length = 100)
sol = ode(u0, tspan, lotvol, p)  # here parameters needed
matplot(sol[,1], sol[,c(2, 3)], type = "l", xlab="", ylab="")


# Page 494 Example 19.31 (Second Order IVP)

library(deSolve)
pend <- function(t, u, p) {
  du <- u[2]
  d2u <- -p[1]*u[2] - p[2]*sin(u[1])
  list(c(du, d2u)) }
p <- c(0.25, 5)
u0 <- c(pi - 0.1, 0)
tspan <- seq(0, 10, length = 100)
sol <- ode(u0, tspan, pend, p)
matplot(sol[,1], sol[,c(2, 3)], 
        type = "l", lty = 1, xlab="", ylab="")
        
        
# Page 495 Example 19.32 (Bratu Equation)

library(bvpSolve)
bratu <- function(x, u, p) {
  du <- u[2]
  d2u <- -exp(u[1])
  list(c(du, d2u)) }
xvals <- seq(0, 1, length = 100)
uinit <- c(0, NA); uend <- c(0, NA)
sol <- bvpcol(uinit, xvals, bratu, uend
 #, xguess = c(0, 0.5, 1), yguess = matrix(4, nrow = 2, ncol = 3)
)
plot(sol, which = 1, xlab="", main="")


# Page 496 Example 19.34 (Shooting Method)

library(bvpSolve)
bratu <- function(x, u, p) {
  du <- u[2]
  d2u <- -exp(u[1])
  list(c(du, d2u)) }
xvals <- seq(0, 1, length = 100)
uinit <- c(0, NA); uend <- c(0, NA)
sol <- bvpshoot(uinit, xvals, bratu, uend, guess = 0)  # guess = 6
plot(sol, which = 1, xlab="", main="")


# Page 496 Partial Differential Equation

n <- 10
sD <- diag(0, n); diag(sD[-n, -1]) <- 1; diag(sD[-1, -n]) <- 1
I <- diag(1, n)
D <- 4*I - sD
A <- I %x% D + sD %x% -I  # Poisson matrix
x <- seq(0, 1, length = n)
F <- matrix(0, nrow = n, ncol = n)
F[,1] <- x*(1 - x)
b <- as.vector(F)
solVec <- solve(A, b)
solMat <- matrix(solVec, nrow = n)
persp(solMat, theta=30, phi=30, expand=0.5, ticktype="detailed",
      xlab="", ylab="", zlab="")
      
      
# Page 497 Example 19.35 (Fractions) 
      
fract <- function(num, den) {
  structure(list(num = num, den = den), class = "fract") }
print.fract <- function(a) cat(a$num, "/", a$den, sep = "") 
"+.fract" <- function(a, b) { 
  fract(a$num * b$den + b$num * a$den,  a$den * b$den) }
"*.fract" <- function(a, b) fract(a$num * b$num, a$den * b$den) 
"==.fract" <- function(a, b) a$num * b$den == a$den * b$num


# Page 498 Example 19.36 (Polynomial Class)

pol <- function(c) structure(list(coeff = c), class = "pol")
"+.pol" <- function(p, q) {
  print("polynomial addition")
  v <- p$coeff; w <- q$coeff
  n <- max(length(v), length(w))
  v <- c(v, rep(0, n - length(v)))
  w <- c(w, rep(0, n - length(w)))
  pol(v + w) }
  
  
# Page 499 Example 19.37 (Parabola Subclass) 

par <- function(c) { 
  stopifnot(length(c) == 3)  ## abort if no parabola
  structure(list(coeff = c), class = c("par", "pol")) }
"+.par" <- function(p, q) { 
  print("parabola addition")
  v <- p$coeff; w <- q$coeff
  par(v + w) }
  
  
# Page 499 Example 19.38 (Generic Evaluation Function) 

evalp <- function(p, ...) { UseMethod("evalp", p) }
evalp.pol <- function(p, x) { 
  print("evalp.pol")
  res <- 0
  for (c in rev(p$coeff)) res <- x * res + c
  return(res) }
evalp.par <- function(p, x) {
  print("evalp.par")
  v <- p$coeff
  v[1] + v[2]*x + v[3]*x*x }
  

# Page 501 Example 19.40 (R6 Polynomial Class)

library(R6)
Polynomial <- R6Class(
  classname = "Polynomial", 
  public = list(
    coeff = NULL,
    initialize = function(coeff) { self$coeff <- coeff }, 
    evalp = function(x) {
      res <- 0
      for (c in rev(self$coeff)) res <- x * res + c 
      cat("Polynomial evaluation:", res) } ) )
"+.Polynomial" <- function(p, q) {
  print("Polynomial addition")
  v <- p$coeff; w <- q$coeff
  n <- max(length(v), length(w))
  v <- c(v, rep(0, n - length(v)))
  w <- c(w, rep(0, n - length(w)))
  Polynomial$new(v + w) }
  
  
# Page 502 Example 19.41 (R6 Parabola Subclass)

Parabola <- R6Class(
  classname = "Parabola", 
  inherit = Polynomial,
  public = list(
    initialize = function(coeff) {
      stopifnot(length(coeff) == 3)
      super$initialize(coeff) } ) )
Parabola$set("public", "evalp", 
  function(x) {
    v <- self$coeff
    res <- v[1] + v[2]*x + v[3]*x*x
    cat("Parabola evaluation:", res) } )
"+.Parabola" <- function(p, q) { 
  print("Parabola addition")
  v <- p$coeff; w <- q$coeff  # line 14 in Example 19.37
  Parabola$new(v + w) }
  

# Page 503 Linked Lists

library(R6)
Node <- R6Class(
  public = list(val = NULL, nxt = NULL, 
    initialize = function(val) self$val <- val ) ) 
LinkedList <- R6Class(
  classname = "LinkedList",
  public = list(head = NULL, tail = NULL,
    addNode = function(node) {
      if (is.null(self$head)) self$head <- node  
      else self$tail$nxt <- node
      self$tail <- node } ) )
print.LinkedList <- function(lst) {
  node <- lst$head; w <- c()
  while (!is.null(node)) {
    w <- c(w, node$val)
    node <- node$nxt }
  print(w) } 
  
lst <- LinkedList$new() 
for (i in 1:10) lst$addNode(Node$new(i))
print(lst)


      
      


  
        

  
  

  



  
  
  
  
  


      















      
        















































































