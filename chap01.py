# Page.19 Rice and Chessboard Problem

s = 0; n = 1; fv = 1
while n < 65:
    s = s + fv
    n = n +1
    fv = 2 * fv
print(s)


# Page 24 Example 3.1 Sqrt

from math import sqrt
n = 10; x = 2
for _ in range(n): x = sqrt(x)
for _ in range(n): x = x**2
print(x)


# Page 26 While Loop

index = 1
while index < 3:
    print(f"{index} times 5 is {5*index}")
    index += 1
    
# Page 27 For Loop

for index in range(1,3):
    print(f"{index} times 5 is {5*index}")


# Page 27 Example 3.2 Sum

n, sm = 100, 0
for i in range(1, n+1): sm += i
print(f"The sum is {sm}")  


# Page 28 Example 3.3 Collatz

n = 100  # input
while n > 1:
    if n % 2 == 0: n //= 2  # integer division
    else: n = 3*n + 1
    print(n)
print('reached 1')


# Page 29 Machine Epsilon

eps, i = 1.0, 0
while 1 + eps > 1:
    eps /= 2
    i += 1
eps *= 2  
print(eps); print(i)

eps, i = 1.0, 0
while eps > 0:
    eps /= 2
    i += 1
print(eps); print(i) 


# Page 31 Example 3.4 Sieve of Eratosthenes

n = 10                    # input upper limit
L = list(range(2, n+1))   # constructs a list from  range()
P = []                    # [] denotes the empty list
while L != []: 
    p = L[0]              # the smallest number still in L
    P.append(p)           # is appended to the end of P
    for i in L:
        if i % p == 0: L.remove(i)
print(P)


# Page 34 Example 3.5 Factorial

def factorial(n):
    res = 1
    for i in range(1, n+1): res *= i
    return res
print(factorial(4))  # Out: 24


# Page 34 Example 3.6 Function as Argument

def vmax(f,n):
    max_y = f(0)
    h = 1.0/n
    for i in range(1, n+1): 
        y = f(i*h)
        if y > max_y: max_y = y
    return max_y  
def g(x): return 4*x*(1 - x)
print(vmax(g,7))  # Out: 0.9795918367346939


# Page 36  Example 3.7 Function as Return Value

def ddx(f):
    h = 1.e-6
    def f_prime(x): return (f(x+h) - f(x)) / h
    return f_prime
def g(x): return 4*x*(1 - x)
print(ddx(g)(0.3))  # Out: 1.5999960000234736
dgdx = ddx(g)
print(dgdx(0.5))    # Out: -4.0000225354219765e-06


# Page 37 Example 3.8 Recursive Factorial

def factorial_rec(n):
    if n == 0: return 1
    return n*factorial_rec(n-1)
    
def factorial_rec(n): return 1 if n == 0 else n*factorial_rec(n-1)

    
# Page 37 Example 3.10 Euclidean Algorithm

def gcd(a,b): return a if b == 0 else gcd(b, a % b)


# Page 37 Example 3.11 Fibonacci Function

def fib(n): return 1 if n == 1 or n == 2 else fib(n-1) + fib(n-2)


# Page 37 Example 3.12 Ackermann Function

def ack(x,y):
    if x == 0: return y + 1
    if y == 0: return ack(x-1, 1) 
    return ack(x-1, ack(x, y-1)) 
    
    
# Page 38 Example 3.13 Quicksort

def qsort(lst):
    if lst == []: return []
    p = lst[0]  # pivot element
    sml = qsort([x for x in lst[1:] if x < p])
    grt = qsort([x for x in lst[1:] if x >= p])
    return sml + [p] + grt    
testList = [4, 5, 7, 3, 8, 3]
print(qsort(testList))  # out: [3, 3, 4, 5, 7, 8]


# Pages 38, 39 Examples 3.14, 3.15  String Formatting, Interpolation

n = 1
fstr = f"{n} divided by 3 is {n/3}"
print(fstr)
# Out: 1 divided by 3 is 0.3333333333333333
fstr = f"{n} divided by 3 is {n/3 :.4}"
print(fstr)
# Out: 1 divided by 3 is 0.3333

# Page 39, Example 3.16 Numeric Formatting

print(f"|{12.34567 :.3f}|")   # Out: '|12.346|'
print(f"|{12.34567 :7.3f}|")  # Out: '| 12.346|'
print(f"|{12 :3d}|")          # Out: '| 12|'
   
# Page 39 Writing Strings

wf = open('parrot.txt', 'w')
wf.write('The parrot is dead!\n')
wf.write('It is only resting.')
wf.close()

# Page 40 Readiing Strings

rf = open('parrot.txt', 'r')
fstr = rf.read()
print(fstr)
    
    
# Page 41 Example 3.17 Number Strings

tbl = [[1, 2, 3], [4, 5, 6]]
tblstrg = '' 
for r in tbl: 
    for num in r: tblstrg += f" {num}" 
    tblstrg += '\n'      
row_lst = tblstrg.split('\n')
in_tbl = []
for r in row_lst: 
    nums  = [int(c) for c in r.split()]
    if nums == []: break  
    in_tbl.append(nums)
print(in_tbl)

    
# Page 42 Example 3.18 Pickle

from pickle import dump, load
tbl = [[1, 2, 3], [4, 5, 6]]  # input
fwb = open('datafile.pkl', 'wb') 
dump(tbl, fwb)  # write
fwb.close()
frb = open('datafile.pkl', 'rb')
in_tbl = load(frb)  # read
print(in_tbl)

     
# Page 43. Example 3.19 Fraction Class

class Fraction:
    def __init__(self, num, den):  # initialization
        self.num = num             # internal storage
        self.den = den     
    def add(self, b):             # fraction addition
        return Fraction(self.num * b.den + b.num * self.den, 
                        self.den * b.den ) 
     #  __add__ infix operator           
    def isEqualTo(self, b):       # equality between fractions
        return True if self.num * b.den == self.den * b.num else False 
     # __eq__ infix operator    
a = Fraction(3,4)
b = Fraction(1,2)
c = a.add(b)
print(c.num, c.den) # 10, 8 
d = b.add(a)  
print(c.isEqualTo(d))  # Out: True 
# c = a + b; d = b + a
# print(c = d) 


# Pages 44-46 Example 3.20  Polynomial Class

class Polynomial:
    def __init__(self, coeff):  # data initialization
        self.coeff = coeff      # internal storage  
    def __call__(self, x):      # method polynomial application
        s = 0
        for i in range(len(self.coeff)): s += self.coeff[i] * x**i
        return s   
    def __add__(self, q):   # method polynomial addition
        l = []
        if len(self.coeff) > len(q.coeff):
            l += self.coeff   
            for i in range(len(q.coeff)): l[i] += q.coeff[i]                
        else: 
            l += q.coeff
            for i in range(len(self.coeff)): l[i] += self.coeff[i]
        return Polynomial(l) 
    def __mul__(self, q):   # method polynomial multiplication
        d1, d2 = len(self.coeff), len(q.coeff)
        l = [0 for i in range(d1 + d2 - 1)]
        for i in range(d1):
            for j in range(d2): l[i+j] += self.coeff[i] * q.coeff[j]
        return Polynomial(l) 
    def __eq__(self, q): 
        d = len(self.coeff) 
        if d != len(q.coeff): return False
        for i in range(d):
            if abs(self.coeff[i] - q.coeff[i]) > 1.e-14: return False
        return True  
    def __add__(self, q):  # method polynomial addition
        lst = []
        if len(self.coeff) > len(q.coeff):
            lst += self.coeff   
            for i in range(len(q.coeff)): 
                lst[i] += q.coeff[i]                
        else: 
            lst += q.coeff
            for i in range(len(self.coeff)): 
                lst[i] += self.coeff[i]
        return Polynomial(lst)
    def __mul__(self, q):  # method polynomial multiplication
        d1, d2 = len(self.coeff), len(q.coeff)
        lst = [0 for i in range(d1 + d2 - 1)]
        for i in range(d1):
            for j in range(d2): 
                lst[i+j] += self.coeff[i]*q.coeff[j]
        return Polynomial(lst)   
    def __eq__(self, q): 
        d = len(self.coeff) 
        if d != len(q.coeff): return False
        for i in range(d):
            if abs(self.coeff[i] - q.coeff[i]) > 1.e-14: 
                return False
        return True       
p = Polynomial([1, 2, 3])
print(p(4))  # Out: 57
q = Polynomial([3,4,5])
r = Polynomial([6,7,8,9])
print(p*(q + r) == p*q + p*r)  # Out: True 

        
# Pages 47, 48 Parabola Class
      
class Parabola(Polynomial):   # subclass, needs code above
    def __init__(self, coefficients):  
            if len(coefficients) != 3: 
                print('no parabola')
                return
            else: super().__init__(coefficients)     
    def roots(self): 
        print('To be implemented')
        return
        
    def __add__(self, q): 
        l = [0, 0, 0]
        for i in range(3): l[i] += self.coeff[i] + q.coeff[i]
        return Parabola(l)      
# >>> p = Parabola([1, 2, 3])
# >>> (p + p).roots()) Out: To be implemented
        