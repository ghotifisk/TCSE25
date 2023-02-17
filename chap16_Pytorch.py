# Page 363 Example 16.3 Gradient Descent

from torch import tensor
inputs = tensor([1., 2., 3.]) 
targets = tensor([1., 2., 2.])
w, b = 1, 0                 
lr = 0.1                    
epochs = 500  
for _ in range(epochs):
    preds = w * inputs + b  
    errors = preds - targets
    w_grad = 2 * (inputs * errors).mean()
    b_grad = 2 * errors.mean() 
    w -= lr * w_grad  
    b -= lr * b_grad
print(w, b)  


from torch import tensor
x = tensor(3., requires_grad=True)
y = x**2; z = 3*y
z.backward() 
x.grad
x.grad.zero_()
v = 2*x
v.backward()
x.grad 


# Page 363 Example 16.5 Automatic Differentiation

from torch import tensor, no_grad 
from torch.nn import MSELoss
inputs = tensor([1., 2., 3.]);  targets = tensor([1., 2., 2.])
learning_rate = 0.1;  epochs = 500 
w = tensor(1., requires_grad=True)
b = tensor(0., requires_grad=True)
loss_fn = MSELoss()
for _ in range(epochs):
    preds = w * inputs + b                           
    loss = loss_fn(preds, targets)  
    loss.backward()
        with no_grad(): 
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
print(w.detach(), b.detach()) 


# Page 367 Example 16.6 Optimizer

from torch import tensor, optim 
from torch.nn import MSELoss
inputs = tensor([1., 2., 3.]);  targets = tensor([1., 2., 2.])
learning_rate = 0.1;  epochs = 500 
w = tensor(1., requires_grad=True)
b = tensor(0., requires_grad=True)
loss_fn = MSELoss()
optimizer = optim.SGD([w,b], lr=learning_rate)  
for _ in range(epochs):
    preds = w * inputs + b                           
    loss = loss_fn(preds, targets)  
    loss.backward()
    optimizer.step()       
    optimizer.zero_grad()  
print(w.detach(), b.detach()) 


# Page 368 Example 16.7 "Linear" Module

from torch import tensor, optim
from torch.nn import Linear, MSELoss
model = Linear(1,1)
# model.weight.detach()  
# model.bias.detach()  
from torch.nn import Parameter
model.weight = Parameter(tensor([[1.]]))
model.bias = Parameter(tensor([0.]))
inputs = tensor([[1.], [2.], [3.]])   
targets = tensor([[1.], [2.], [2.]])
learning_rate = 0.1; epochs = 500 
loss_fn = MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for _ in range(epochs):
    preds = model(inputs)  
    loss = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(model.weight.detach(), model.bias.detach())


# Page 370 Example 16.8 Local Minimum 

from torch import tensor, sin, optim
x = tensor(4., requires_grad = True)
f = lambda x: -x*sin(x)
optimizer = optim.Adam([x], lr=0.05)
for _ in range(200):
    optimizer.zero_grad()
    loss = f(x)
    loss.backward()
    optimizer.step()
print(x.detach(), f(x).detach())  


# Page 371 Example 16.9 Rosenbrock Function

from torch import tensor, optim 
x = tensor(0., requires_grad = True)  
y = tensor(0., requires_grad = True)
rosen = lambda x,y: (1 - x)**2 + 100*(y - x*x)**2
optimizer = optim.Adam([x,y], lr=0.05)
for _ in range(1_000):
    optimizer.zero_grad()
    loss = rosen(x,y)
    loss.backward()
    optimizer.step()
print(x.detach(), y.detach(), rosen(x,y).detach()) 


# Page 371 Example 16.10 Nonlinear Equation

from torch import tensor, cos, optim
f = lambda x: x + 2*cos(x)
x = tensor(0.5, requires_grad=True)
optimizer = optim.SGD([x], lr=0.1)
for _ in range(20):
    loss = f(x)**2 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(x.detach())  


# Page 372 Example 16.11

from torch import tensor, optim 
x = tensor(1., requires_grad=True)
y = tensor(1., requires_grad=True)
optimizer = optim.Adam([x,y], lr=0.007)
for _ in range(300):
    loss = (y - x**3 - 2 * x**2 + 1)**2 + (y + x**2 - 1)**2 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(x.detach(), y.detach()) 


# Page 372 Example 16.12 Matrix Equation

from torch import tensor, zeros, optim, mm
from torch.nn import MSELoss
A = tensor([[1., 2., 3.], [1., 1., 1.], [3., 3., 1.]])
b = tensor([[2.], [2.], [0.]]) 
x = zeros(3).unsqueeze(1)
x.requires_grad=True  
loss_fn = MSELoss()
optimizer = optim.Adam([x], lr=1.)
for e in range(1_000):
    optimizer.zero_grad()  
    p = mm(A, x)       
    loss = loss_fn(p, b) 
    # if e % 100 == 0: print(loss)
    loss.backward()
    optimizer.step()
print(x.detach().T)  


# Page 372 Example 16.12 MSELoss

from torch import tensor, zeros, optim, mm
from torch.nn import MSELoss
A = tensor([[ 9.,  3., -6., 12.], [ 3.,  26., -7., -11.],
            [-6., -7.,  9.,  7.], [12., -11.,  7.,  65.]])
b = tensor([[18.], [11.], [3.], [73.]])
x = zeros(4).unsqueeze(1)
x.requires_grad=True
optimizer = optim.Adam([x], lr=0.001)  
loss_fn = MSELoss()         
for _ in range(5_000):     
    optimizer.zero_grad()
    p = mm(A, x)           
    loss = loss_fn(p,b)    
    loss.backward()
    optimizer.step()
print(x.detach().T) 


# Page 372 Example 16.14 Descent along Quadratic Form

from torch import tensor, zeros, optim, mm
A = tensor([[ 9.,  3., -6., 12.], [ 3.,  26., -7., -11.],
            [-6., -7.,  9.,  7.], [12., -11.,  7.,  65.]])
b = tensor([[18.], [11.], [3.], [73.]])
x = zeros(4).unsqueeze(1)
x.requires_grad=True
optimizer = optim.Adam([x], lr=0.09) 
for _ in range(160):
    optimizer.zero_grad()
    loss = 1/2 * mm(x.T, mm(A, x)) -  mm(x.T, b)
    loss.backward()
    optimizer.step()
print(x.detach().T)  


# Page 377 Example 16.18 Threshold-Function Approximation

from torch import tensor, linspace, heaviside, sin 
theta = lambda x: heaviside(x, tensor(1.))
f = sin
def fn(x, n): 
    xn = linspace(0, 6.283, n)
    res = f(x[0]) * theta(x - xn[0])
    for i in range(n-1): 
        res += (f(xn[i+1]) - f(xn[i])) * theta(x - xn[i+1])
    return res
x = linspace(0, 6.283, 200)
from matplotlib.pyplot import plot
plot(x, fn(x, 10));  plot(x, fn(x, 100))


# Page 379 Example 16.21 Sigmoid-Function Approximation

from torch import linspace, sigmoid, sin 
f = sin
def fn(x, n): 
    xn = linspace(0, 6.283, n)
    res = f(x[0]) * sigmoid(50*(x - xn[0]))
    for i in range(n-1): 
        res += (f(xn[i+1]) - f(xn[i])) * sigmoid(50*(x - xn[i+1]))
    return res
x = linspace(0, 6.283, 200)
from matplotlib.pyplot import plot
plot(x, fn(x, 10));  plot(x, fn(x, 100))


# Page 380 Definition 16.22

from torch.nn import Sequential, Linear, Sigmoid 
N = Sequential(
      Linear(1, n),                    
      Sigmoid(),                 
      Linear(n, 1, bias=False)) 
      
      
# Page 382 Example 16.23 Runge Function

from torch import linspace, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
f = lambda x: 1/(1 + 25*x**2)
x = linspace(-1, 1, 50).unsqueeze(1)
N = Sequential(Linear(1,30), Sigmoid(), Linear(30, 1, bias=False))
loss_fn = MSELoss()
optimizer = optim.LBFGS(N.parameters())
for _ in range(100):
    def closure():  
        optimizer.zero_grad()
        loss = loss_fn(N(x), f(x))  
        loss.backward()
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot 
plot(x, N(x).detach())
# plot(x, N(x).detach() - f(x))


# Page 383 Example 16.24 Rosenbrock Function

from torch import linspace, optim, cartesian_prod, meshgrid
from torch.nn import Sequential, Linear, Sigmoid, MSELoss  
rosen = lambda x,y: (1 - x)**2 + 100*(y - x*x)**2
n = 30
x = linspace(-2, 2, n)  
y = linspace(-2, 4, n)
xy = cartesian_prod(x,y)
z = rosen(xy[:, 0], xy[:, 1]).unsqueeze(1)
N = Sequential(
    Linear(2, n*n),  
    Sigmoid(),  
    Linear(n*n, 1, bias=False))
optimizer = optim.LBFGS(N.parameters())
loss_fn = MSELoss()
for _ in range(100):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(N(xy), z)  
        loss.backward()
        return loss
    optimizer.step(closure)
X, Y = meshgrid(x, y, indexing='ij')
Z = N(xy).detach().numpy().reshape(n,n)
Z_ex = rosen(X,Y).numpy()
from matplotlib.pyplot import figure
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap="jet") 
ax.plot_wireframe(X, Y, Z - Z_ex,  color='#ff7f0e') 
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z - Z_ex, cmap="jet") 


# Page 385 Example 16.25

from torch import tensor, autograd
x = tensor(1., requires_grad=True)
u = x**3
dudx, = autograd.grad(u, x)
print(dudx)  


# Page 386 Example 16.26 1st Derivative

from torch import tensor, autograd
x = tensor([1., 2.], requires_grad=True)
u = x**3
v = tensor([1., 1.])
dudx, = autograd.grad(u, x, grad_outputs = v) 
print(dudx)  


# Page Example 16.27 2st Derivative
# Note: Original Book Example Corrected  

from torch import tensor, autograd
x = tensor([1., 2.], requires_grad=True)
u = x**3
v = tensor([1., 1.])
# dudx, = autograd.grad(u, x, grad_outputs = v) 
dudx, = autograd.grad(u, x,  grad_outputs = v, create_graph=True)
d2udx2, = autograd.grad(dudx, x, grad_outputs = v)   
print(d2udx2)


# Page Definition 16.28 "diff, diff2" Operators
# to be stored in "derivate.py" 

def diff(u, x):
  from torch import autograd, ones_like
  grad, =  autograd.grad(outputs = u,   
                          inputs = x,  
                    grad_outputs = ones_like(u),
                    create_graph = True)
  return grad
  
def diff2(u,x): return diff(diff(u,x), x)


# Page 387 Example 16.29 Atan Derivatives

from torch import linspace, atan
from derivate import diff, diff2

x = linspace(-6., 6., 100).unsqueeze(1)
x.requires_grad=True

u = atan(x)           
dudx = diff(u, x)     
d2udx2 = diff2(u, x)  

from matplotlib.pyplot import plot, legend  
plot(x.detach(), u.detach()) 
plot(x.detach(), dudx.detach())
plot(x.detach(), d2udx2.detach())
legend(['atan(x)', 'd/dx atan(x)', 'd2/dx2 atan(x)'])


# Page 388 Definition 16.30 Laplace Operator
# to be stored in "derivate.py"

def laplace(u, xy):
    grads = diff(u, xy)
    dudx, dudy = grads[:, 0], grads[:, 1]
    du2dx2 = diff(dudx, xy)[:, 0]        
    du2dy2 = diff(dudy, xy)[:, 1]
    return (du2dx2 + du2dy2).unsqueeze(1) 
    

# Page 388 Example 16.31 Partial Derivative

from torch import linspace, cartesian_prod
from derivate import laplace
x = linspace(0,1,50); y = linspace(0,1,50)
xy = cartesian_prod(x,y)
xy.requires_grad=True
f = lambda x,y: 1 + x**2 + 2 * y**2
u = f(xy[:,0], xy[:,1])
lap = laplace(u, xy)
print(lap.detach().T)


# Page 389, 390 Example 16.33 Integration

from torch import linspace, optim, atan
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
from derivate import diff
f = lambda x: 1/(1 + x**2)
x = linspace(-6., 6., 100, requires_grad=True).unsqueeze(1)
N = Sequential(Linear(1,50), Sigmoid(), Linear(50, 1, bias=False))
loss_fn = MSELoss()
optimizer = optim.LBFGS(N.parameters())
for _ in range(10):
    def closure():
        optimizer.zero_grad()
        global u               
        u = x * N(x)                      
        loss = loss_fn(diff(u, x), f(x))   
        loss.backward()
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot, figure 
fig = figure()    
plot(x.detach(), u.detach())
fig = figure()
plot(x.detach(), u.detach() - atan(x).detach())


# Page 391 Example 16.34 IVP

from torch import linspace, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss 
from derivate import diff
x = linspace(0., 5., 50, requires_grad=True).unsqueeze(1)
N = Sequential(Linear(1,50), Sigmoid(), Linear(50, 1, bias=False))
loss_fn = MSELoss()
optimizer = optim.LBFGS(N.parameters())
for _ in range(10):
    def closure():
        optimizer.zero_grad()
        global u              
        u = (1-x) + x*N(x)           
        dudx = diff(u,x)
        loss = loss_fn(dudx, x - u)  
        loss.backward()
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot     
plot(x.detach(), u.detach())


# Page 392 Example 16.35 Coupled Equation Systems

from torch import linspace, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
from derivate import diff
x = linspace(0., 6.283, 50, requires_grad=True).unsqueeze(1)
N_u = Sequential(Linear(1,50), Sigmoid(), Linear(50,1,bias=False))
N_v = Sequential(Linear(1,50), Sigmoid(), Linear(50,1,bias=False))
loss_fn = MSELoss(reduction="sum")
params = list(N_u.parameters()) + list(N_v.parameters())
optimizer = optim.LBFGS(params, lr=0.1)
for _ in range(150):
    def closure():
        optimizer.zero_grad()
        global u, v              
        u = x * N_u(x)           
        v = (1-x) + x * N_v(x)   
        dudx = diff(u, x)
        dvdx = diff(v, x)
        loss = loss_fn(dudx, v) + loss_fn(dvdx, -u)
        loss.backward()
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot     
plot(x.detach(), u.detach())
plot(x.detach(), v.detach())


# Page 394 Example 16.36 BVP

from torch import linspace, optim, exp
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
from derivate import diff2
f = lambda x: x*(x+3)*exp(x)
x = linspace(0., 1., 50, requires_grad=True).unsqueeze(1)
N = Sequential(Linear(1,50), Sigmoid(), Linear(50, 1, bias=False))
loss_fn = MSELoss(reduction="sum")
optimizer = optim.LBFGS(N.parameters(), lr=0.01)
for _ in range(50):
    def closure():
        optimizer.zero_grad()
        global u
        u = (1-x) * x * N(x)
        d2udx2 = diff2(u,x)
        loss = loss_fn(-d2udx2, f(x))
        loss.backward()
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot
plot(x.detach(), u.detach())
 

# Page 395 Example 16.37 2nd Order IVP

from torch import linspace, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss 
from derivate import diff2
x = linspace(0., 6.283, 50, requires_grad=True).unsqueeze(1)
N = Sequential(Linear(1,50), Sigmoid(), Linear(50, 1, bias=False))
loss_fn = MSELoss()
optimizer = optim.LBFGS(N.parameters())
for _ in range(20):
    def closure():
        optimizer.zero_grad()
        global u
        u = x + x*x * N(x)  
        d2udx2 = diff2(u, x)
        loss = loss_fn(d2udx2, -u)  
        loss.backward()
        print(loss)
        return loss
    optimizer.step(closure)
from matplotlib.pyplot import plot     
plot(x.detach(), u.detach())


# Page 396 Example 16.38 Homogeneous Poisson Equation

from torch import linspace, optim, cartesian_prod, meshgrid, ones
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
from derivate import laplace 
n = 30
xs, ys = linspace(0,1,n), linspace(0,1,n)
xy = cartesian_prod(xs, ys)
xy.requires_grad=True
x, y = xy[:,0], xy[:,1]
B = (x*(1 - x)*y*(1 - y)).unsqueeze(1) 
f = ones(n*n, 1) 
N = Sequential(Linear(2,n*n), Sigmoid(), Linear(n*n,1,bias=False)) 
optimizer = optim.LBFGS(N.parameters())
loss_fn = MSELoss() 
for _ in range(20):
    def closure():
        optimizer.zero_grad()
        global u
        u = B * N(xy)
        loss = loss_fn(-laplace(u,xy), f)
        loss.backward(retain_graph=True)
        return loss  
    optimizer.step(closure) 
X, Y = meshgrid(xs, ys)
Z = u.reshape(n,n).detach().numpy()

from matplotlib.pyplot import figure, show
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, 
       rstride=1, cstride=1, cmap='jet', linewidth=0) 
show() 


# Page 398 Example 16.39 Laplace Equation

from torch import linspace, optim, cartesian_prod, meshgrid, zeros
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
from derivate import laplace 
n = 30
xs, ys = linspace(0,1,n), linspace(0,1,n)
xy = cartesian_prod(xs, ys)
xy.requires_grad=True
x, y = xy[:,0], xy[:,1]
B = (x*(1 - x)*y*(1 - y)).unsqueeze(1)
C = ((1 - y)*x*(1 - x)).unsqueeze(1) 
f = zeros(n*n,1)
N = Sequential(Linear(2,n*n), Sigmoid(), Linear(n*n,1,bias=False)) 
optimizer = optim.LBFGS(N.parameters())
loss_fn = MSELoss() 
for _ in range(20):
    def closure():
        optimizer.zero_grad()
        global u
        u = C + B * N(xy)
        loss = loss_fn(-laplace(u,xy), f)
        loss.backward(retain_graph=True)
        return loss  
    optimizer.step(closure) 
X, Y = meshgrid(xs, ys)
Z = u.reshape(n,n).detach().numpy()

from matplotlib.pyplot import figure, show
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, 
       rstride=1, cstride=1, cmap='jet', linewidth=0) 
show() 


# Page 404 Example 16.41 Negation 

from torch import tensor, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
inputs = tensor([[0.], [1.]])
targets = tensor([[1.], [0.]])
N = Sequential(Linear(1, 1), Sigmoid()) 
loss_fn = MSELoss()
optimizer = optim.Adam(N.parameters(), lr=1.)
for _ in range(50):
    optimizer.zero_grad()           
    preds =  N(inputs)              
    loss = loss_fn(preds, targets)  
    loss.backward()                 
    optimizer.step()                
print(preds.detach().T)  


# Page 404 Example 16.42 Conjunction, Disjunction, Negated Conjunction 

from torch import tensor, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
inputs = tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
targets = tensor([[0.], [0.], [0.], [1.]])  # conjunction
#targets = tensor([[0.], [1.], [1.], [1.]])  # disjunction
#targets = tensor([[1.], [1.], [1.], [0.]])  # negated conjunction
N = Sequential(Linear(2, 1), Sigmoid()) 
loss_fn = MSELoss()
optimizer = optim.Adam(N.parameters(), lr=1.)
for _ in range(500):
    optimizer.zero_grad()           
    preds =  N(inputs)              
    loss = loss_fn(preds, targets)  
    loss.backward()                 
    optimizer.step()                
print(preds.detach().T)  


# Page 405 Example 16.43 Exclusive Disjunction

from torch import tensor, optim
from torch.nn import Sequential, Linear, Sigmoid, MSELoss
inputs = tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
targets = tensor([[0.], [1.], [1.], [0.]])
N = Sequential(
    Linear(2, 2), Sigmoid(),     
    Linear(2, 1), Sigmoid())
loss_fn = MSELoss()
optimizer = optim.SGD(N.parameters(), lr=10.)
for _ in range(6_000):              
    optimizer.zero_grad()           
    preds =  N(inputs)              
    loss = loss_fn(preds, targets)  
    loss.backward()                 
    optimizer.step()                
print(preds.detach().T)


# Page 407 Digit Recognition
# Runtime on Test-Machine 1.5 Min

from torch import optim, utils
from torch.nn import Sequential, Linear, Flatten, Sigmoid, Softmax   
from torchvision import datasets, transforms
trainset = datasets.MNIST('./', download = True, 
    transform = transforms.ToTensor())
model = Sequential(
    Flatten(),  
    Linear(784, 128), Sigmoid(), Linear(128, 64), Sigmoid(),
    Linear(64, 10), Softmax(dim=1))
def predicted(image):
    output = model(image)
    _, pred = output.max(dim=1)  
    return pred
batches = utils.data.DataLoader(trainset, batch_size = 64) 
optimizer = optim.Adam(model.parameters()) 
for _ in range(15):
    for images, labels in batches:     
        optimizer.zero_grad()
        outputs = model(images)
        gain = 0    
        for i in range(len(outputs)): 
            gain += outputs[i][labels[i]]      
        loss =  1 - gain/len(outputs)  
        loss.backward()
        optimizer.step()

testset = datasets.MNIST('./', train = False, 
    transform = transforms.ToTensor())
correct = 0
for image, label in testset:
    if predicted(image) == label: correct += 1
print("Correct:", correct)