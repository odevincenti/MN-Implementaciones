##Apendix
'''
Chapter 1
1.7: error Error-handling routine

Chapter 2
2.2: gaussElimin Gauss elimination
2.3: LUdecomp LU decomposition
2.3: choleski Choleski decomposition
2.4: LUdecomp3 LU decomposition of tridiagonal matrices
2.4: LUdecomp5 LU decomposition of pentadiagonal matrices
2.5: swap Interchanges rows or columns of a matrix
2.5: gaussPivot Gauss elimination with row pivoting
2.5: LUpivot LU decomposition with row pivoting
2.7: gaussSeidel Gauss-Seidel method with relaxation
2.7: conjGrad Conjugate gradientmethod

Chapter 3
3.2: newtonPoly Newton’s method of polynomial interpolation
3.2: neville Neville’s method of polynomial interpolation
3.2: rational Rational function interpolation
3.3: cubicSpline Cubic spline interpolation
3.4: polyFit Polynomial curve fitting
3.4: plotPoly Plots data points and the fitting polynomial

Chapter 4
4.2: rootsearch Brackets a root of an equation
4.3: bisection Method of bisection
4.4: ridder Ridder’smethod
4.5: newtonRaphson Newton-Raphson method
4.6: newtonRaphson2 Newton-Raphson method for systems of equations
4.7: evalPoly Evaluates a polynomial and its derivatives
4.7: polyRoots Laguerre’s method for roots of polynomials

Chapter 6
6.2: trapezoid Recursive trapezoidal rule
6.3: romberg Romberg integration
6.4: gaussNodes Nodes and weights for Gauss-Legendre quadrature
6.4: gaussQuad Gauss-Legendre quadrature
6.5: gaussQuad2 Gauss-Legendre quadrature over a quadrilateral
6.5: triangleQuad Gauss-Legendre quadrature over a triangle

Chapter 7
7.2: euler Eulermethod for solution of initial value problems
7.2: printSoln Prints solution of initial value problems in tabular form
7.3: run kut4 4th order Runge-Kutta method
7.5: run kut5 Adaptive (5th order) Runge-Kutta method
7.6: midpoint Midpoint method with Richardson extrapolation
7.6: bulStoer Simplified Bulirsch-Stoermethod

Chapter 8
8.2: linInterp Linear interpolation
8.2: example8 1 Shooting method example for second-order differential eqs.
8.2: example8 3 Shooting method example for third-order linear diff. eqs.
8.2: example8 4 Shooting method example for fourth-order differential eqs.
8.2: example8 5 Shooting method example for fourth-order differential eqs.
8.3: example8 6 Finite difference example for second-order linear diff. eqs.
8.3: example8 7 Finite difference example for second-order differential. eqs.
8.4: example8 8 Finite difference example for fourth-order linear diff. eqs.

Chapter 9
9.2: jacobi Jacobi’s method
9.2: sortJacobi Sorts eigenvectors in ascending order of eigenvalues
9.2: stdForm Transforms eigenvalue problem into standard form
9.3: inversePower Inverse power method with eigenvalue shifting
9.3: inversePower5 Inverse power method for pentadiagonal matrices
9.4: householder Householder reduction to tridiagonal form
9.5: sturmSeq Sturmsequence for tridiagonal matrices
9.5: gerschgorin Computes global bounds on eigenvalues
9.5: lamRange Bracketsmsmallest eigenvalues of a 3-diag.matrix
9.5: eigenvals3 Findsmsmallest eigenvalues of a tridiagonal matrix
9.5: inversePower3 Inverse power method for tridiagonal matrices

Chapter 10
10.2: goldSearch Golden section search for theminimum of a function
10.3: powell Powell’s method of minimization
10.4: downhill Downhill simplexmethod ofminimization
'''
## libraries
import numpy as np
import math
import error


##>>>Metodo de Eliminacion Gaussiana_________________________________________
## module gaussElimin
'''
x = gaussElimin(a,b).
Solves [a]{b} = {x} by Gauss elimination.
'''

def gaussElimin(a,b):
    n = len(b)
    # Elimination Phase
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    # Back substitution
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

##>>>Metodo de Descomposicion LU______________________________________________
## module LUdecomp
'''
a = LUdecomp(a)
LUdecomposition: [L][U] = [a]
x = LUsolve(a,b)
Solution phase: solves [L][U]{x} = {b}
'''

def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                 lam = a [i,k]/a[k,k]
                 a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                 a[i,k] = lam
    return a

def LUsolve(a,b):
    n = len(a)
    for k in range(1,n):
        b[k] = b[k] - np.dot(a[k,0:k],b[0:k])
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b


## module choleski
'''
L = choleski(a)
Choleski decomposition: [L][L]transpose = [a]
x = choleskiSol(L,b)
Solution phase of Choleski’s decomposition method
'''

def choleski(a):
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = math.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            error.err('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    return a

def choleskiSol(L,b):
    n = len(b)
    # Solution of [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
    # Solution of [L_transpose]{x} = {y}
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b


##>>>Matrices de coedicientes simetricos y de bandas__________________________
## module LUdecomp3
'''
c,d,e = LUdecomp3(c,d,e).
LU decomposition of tridiagonal matrix [c\d\e]. On output
{c},{d} and {e} are the diagonals of the decomposed matrix.
= LUsolve(c,d,e,b).
Solves [c\d\e]{x} = {b}, where {c}, {d} and {e} are the
vectors returned from LUdecomp3.
'''

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b


## module LUdecomp5
'''
d,e,f = LUdecomp5(d,e,f).
LU decomposition of symmetric pentadiagonal matrix [a], where
{f}, {e} and {d} are the diagonals of [a]. On output
{d},{e} and {f} are the diagonals of the decomposed matrix.
x = LUsolve5(d,e,f,b).
Solves [a]{x} = {b}, where {d}, {e} and {f} are the vectors
returned from LUdecomp5.
'''

def LUdecomp5(d,e,f):
    n = len(d)
    for k in range(n-2):
        lam = e[k]/d[k]
        d[k+1] = d[k+1] - lam*e[k]
        e[k+1] = e[k+1] - lam*f[k]
        e[k] = lam
        lam = f[k]/d[k]
        d[k+2] = d[k+2] - lam*f[k]
        f[k] = lam
    lam = e[n-2]/d[n-2]
    d[n-1] = d[n-1] - lam*e[n-2]
    e[n-2] = lam
    return d,e,f

def LUsolve5(d,e,f,b):
    n = len(d)
    b[1] = b[1] - e[0]*b[0]
    for k in range(2,n):
        b[k] = b[k] - e[k-1]*b[k-1] - f[k-2]*b[k-2]
    b[n-1] = b[n-1]/d[n-1]
    b[n-2] = b[n-2]/d[n-2] - e[n-2]*b[n-1]
    for k in range(n-3,-1,-1):
        b[k] = b[k]/d[k] - e[k]*b[k+1] - f[k]*b[k+2]
    return b


##>>>Pivoteo__________________________________________________________________
## module swap
'''
swapRows(v,i,j).
Swaps rows i and j of a vector or matrix [v].
swapCols(v,i,j).
Swaps columns of matrix [v].
'''

def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]
    

## module gaussPivot
'''
x = gaussPivot(a,b,tol=1.0e-12).
Solves [a]{x} = {b} by Gauss elimination with
scaled row pivoting
'''

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)
    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))
    for k in range(0,n-1):
        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: error.err('Matrix is singular')
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)
        # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    if abs(a[n-1,n-1]) < tol: error.err('Matrix is singular')

    # Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    
    return b


## module LUpivot
'''
a,seq = LUdecomp(a,tol=1.0e-9).
LU decomposition of matrix [a] using scaled row pivoting.
The returned matrix [a] = contains [U] in the upper
triangle and the nondiagonal terms of [L] in the lower triangle.
Note that [L][U] is a row-wise permutation of the original [a];
the permutations are recorded in the vector {seq}.
x = LUsolve(a,b,seq).
Solves [L][U]{x} = {b}, where the matrix [a] = and the
permutation vector {seq} are returned from LUdecomp.
'''

def LUdecomp_pivot(a,tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))
    # Set up scale factors
    s = np.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i,:]))
    for k in range(0,n-1):
        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: error.err('Matrix is singular')
        if p != k:
            swapRows(s,k,p)
            swapRows(a,k,p)
            swapRows(seq,k,p)
        # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a,seq

def LUsolve_pivot(a,b,seq):
    n = len(a)
    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]
# Solution
    for k in range(1,n):
        x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x


##>>>Metodos iterativos_______________________________________________________
## module gaussSeidel
'''
x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
Gauss-Seidel method for solving [A]{x} = {b}.
The matrix [A] should be sparse. User must supply the
function iterEqs(x,omega) that returns the improved {x},
given the current {x} (’omega’ is the relaxation factor).
'''


def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega
        # Compute relaxation factor after k+p iterations
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
    print('Gauss-Seidel failed to converge')
    

## module conjGrad
'''
x, numIter = conjGrad(Av,x,b,tol=1.0e-9)
Conjugate gradient method for solving [A]{x} = {b}.
The matrix [A] should be sparse. User must supply
the function Av(v) that returns the vector [A]{v}.
'''

def conjGrad(Av,x,b,tol=1.0e-9):
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = np.dot(s,r)/np.dot(s,u)
        x = x + alpha*s
        r = b - Av(x)
        if(math.sqrt(np.dot(r,r))) < tol:
            break
        else:
            beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta*s
    return x,i


##>>>Metodo de busqueda incremental___________________________________________
## module rootsearch
'''
x1,x2 = rootsearch(f,a,b,dx).
Searches the interval (a,b) in increments dx for
the bounds (x1,x2) of the smallest root of f(x).
Returns x1 = x2 = None if no roots were detected.
'''

def rootsearch(f,a,b,dx):
    x1 = a; f1 = f(a)
    x2 = a + dx; f2 = f(x2)
    while np.sign(f1) == np.sign(f2):
        if x1 >= b: return None,None
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2)
    else:
        return x1,x2
    

##>>>Metodo de biseccion______________________________________________________
## module bisection
'''
root = bisection(f,x1,x2,switch=0,tol=1.0e-9).
Finds a root of f(x) = 0 by bisection.
The root must be bracketed in (x1,x2).
Setting switch = 1 returns root = None if f(x) increases upon bisection.
'''

def bisection(f,x1,x2,switch=1,tol=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    if f2 == 0.0:
        return x2
    if np.sign(f1) == np.sign(f2):
        error.err('La raiz no esta en el intervalo')
    n = int(math.ceil(math.log(abs(x2 - x1)/tol)/math.log(2.0))) # n es el numero de iteraciones
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if np.sign(f2)!= np.sign(f3):
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
    return (x1 + x2)/2.0


##>>>Metodos de Newton-Raphson________________________________________________
## module newtonRaphson
'''
root = newtonRaphson(f,df,a,b,tol=1.0e-9).
Finds a root of f(x) = 0 by combining the Newton-Raphson
method with bisection. The root must be bracketed in (a,b).
Calls user-supplied functions f(x) and its derivative df(x).
'''

def newtonRaphson(f,df,a,b,tol=1.0e-9):
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if (fa) == np.sign(fb): error.err('Root is not bracketed')
    x = 0.5*(a + b)
    for i in range(30):
        fx = f(x)
        if fx == 0.0: return x
        # Tighten the brackets on the root
        if np.sign(fa) != np.sign(fx): b = x
        else: a = x
        # Try a Newton-Raphson step
        dfx = df(x)
        # If division by zero, push x out of bounds
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        x = x + dx
        # If the result is outside the brackets, use bisection
        if (b - x)*(x - a) < 0.0:
            dx = 0.5*(b - a)
            x = a + dx
        # Check for convergence
        if abs(dx) < tol*max(abs(b),1.0): return x
    print('Too many iterations in Newton-Raphson')
    

##>>>Raices de Sistemas de Ecucaciones__________________________________________________
## module newtonRaphson2
'''
soln = newtonRaphson2(f,x,tol=1.0e-9).
Solves the simultaneous equations f(x) = 0 by
the Newton-Raphson method using {x} as the initial
guess. Note that {f} and {x} are vectors.
'''

def newtonRaphson2(f,x,tol=1.0e-9):
    def jacobian(f,x):
        h = 1.0e-4
        n = len(x)
        jac = np.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:,i] = (f1 - f0)/h
        return jac,f0
  
    for i in range(30):
        jac,f0 = jacobian(f,x)
        if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x
        dx = gaussPivot(jac,-f0)
        x = x + dx
        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
            return x
    print('Too many iterations')
    

##>>>Metodo de Euler__________________________________________________________
## module euler
'''
X,Y = integrate(F,x,y,xStop,h).
Euler’s method for solving the
initial value problem {y}’ = {F(x,{y})}, where
{y} = {y[0],y[1],...y[n-1]}.
x,y = initial conditions
xStop = terminal value of x
h = increment of x used in integration
F = user-supplied function that returns the
array F(x,y) = {y’[0],y’[1],...,y’[n-1]}.
'''

def integrate_euler(F,x,y,xStop,h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h,xStop - x)
        y = y + h*F(x,y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)


## module printSoln
'''
printSoln(X,Y,freq).
Prints X and Y returned from the differential
equation solvers using printout frequency ’freq’.
    freq = n prints every nth step.
    freq = 0 prints initial and final values only.
'''

def printSoln(X,Y,freq):
    def printHead(n):
        print("\n x ",end=" ")
        for i in range (n):
            print(" y[",i,"] ",end=" ")
    print()
    
    def printLine(x,y,n):
        print("{:13.4e}".format(x),end=" ")
        for i in range (n):
            print("{:13.4e}".format(y[i]),end=" ")
    print()
    
    m = len(Y)
    try: n = len(Y[0])
    except TypeError: n = 1
    if freq == 0: freq = m
    printHead(n)
    for i in range(0,m,freq):
        printLine(X[i],Y[i],n)
    if i != m - 1: printLine(X[m - 1],Y[m - 1],n)
    
    
##>>>Metodo de Runge-Kutta____________________________________________________
## module run_kut4
'''
X,Y = integrate(F,x,y,xStop,h).
4th-order Runge-Kutta method for solving the initial value problem {y}’ = {F(x,{y})}, where
{y} = {y[0],y[1],...y[n-1]}.
x,y = initial conditions
xStop = terminal value of x
h = increment of x used in integration
F = user-supplied function that returns the array F(x,y) = {y’[0],y’[1],...,y’[n-1]}.
'''

def integrate_rk4(F,x,y,xStop,h):
    def run_kut4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h,xStop - x)
        y = y + run_kut4(F,x,y,h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)


##>>>Metodo de Runge-Kutta adaptativo_________________________________________
## module run_kut5
'''
X,Y = integrate(F,x,y,xStop,h,tol=1.0e-6).
Adaptive Runge-Kutta method with Dormand-Prince
coefficients for solving the
initial value problem {y}’ = {F(x,{y})}, where
{y} = {y[0],y[1],...y[n-1]}.
x,y = initial conditions
xStop = terminal value of x
h = initial increment of x used in integration
tol = per-step error tolerance
F = user-supplied function that returns the
array F(x,y) = {y’[0],y’[1],...,y’[n-1]}.
'''

def integrate_rk5(F,x,y,xStop,h,tol=1.0e-6):
    a1 = 0.2; a2 = 0.3; a3 = 0.8; a4 = 8/9; a5 = 1.0
    a6 = 1.0
    c0 = 35/384; c2 = 500/1113; c3 = 125/192
    c4 = -2187/6784; c5 = 11/84
    d0 = 5179/57600; d2 = 7571/16695; d3 = 393/640
    d4 = -92097/339200; d5 = 187/2100; d6 = 1/40
    b10 = 0.2
    b20 = 0.075; b21 = 0.225
    b30 = 44/45; b31 = -56/15; b32 = 32/9
    b40 = 19372/6561; b41 = -25360/2187; b42 = 64448/6561
    b43 = -212/729
    b50 = 9017/3168; b51 =-355/33; b52 = 46732/5247
    b53 = 49/176; b54 = -5103/18656
    b60 = 35/384; b62 = 500/1113; b63 = 125/192;
    b64 = -2187/6784; b65 = 11/84
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    stopper = 0 # Integration stopper(0 = off, 1 = on)
    k0 = h*F(x,y)
    for i in range(500):
        k1 = h*F(x + a1*h, y + b10*k0)
        k2 = h*F(x + a2*h, y + b20*k0 + b21*k1)
        k3 = h*F(x + a3*h, y + b30*k0 + b31*k1 + b32*k2)
        k4 = h*F(x + a4*h, y + b40*k0 + b41*k1 + b42*k2 + b43*k3)
        k5 = h*F(x + a5*h, y + b50*k0 + b51*k1 + b52*k2 + b53*k3 + b54*k4)
        k6 = h*F(x + a6*h, y + b60*k0 + b62*k2 + b63*k3 + b64*k4 + b65*k5)
        dy = c0*k0 + c2*k2 + c3*k3 + c4*k4 + c5*k5
        E = (c0 - d0)*k0 + (c2 - d2)*k2 + (c3 - d3)*k3 + (c4 - d4)*k4 + (c5 - d5)*k5 - d6*k6
        e = math.sqrt(np.sum(E**2)/len(y))
        hNext = 0.9*h*(tol/e)**0.2
        
        # Accept integration step if error e is within tolerance
        if e <= tol:
            y = y + dy
            x = x + h
            X.append(x)
            Y.append(y)
            if stopper == 1: break # Reached end of x-range
            if abs(hNext) > 10.0*abs(h): hNext = 10.0*h
    
            # Check if next step is the last one; if so, adjust h
            if (h > 0.0) == ((x + hNext) >= xStop):
                hNext = xStop - x
                stopper = 1
            k0 = k6*hNext/h
        else:
            if abs(hNext) < 0.1*abs(h): hNext = 0.1*h
            k0 = k0*hNext/h
        h = hNext
    return np.array(X),np.array(Y)

##>>>Metodo de Jacobi_________________________________________________________
## module jacobi
'''
lam,x = jacobi(a,tol = 1.0e-8).
Solution of std. eigenvalue problem [a]{x} = lam{x}
by Jacobi’s method. Returns eigenvalues in vector {lam}
and the eigenvectors as columns of matrix [x].
'''

def jacobi(a,tol = 1.0e-8): # Jacobi method
    def threshold(a):
        sum = 0.0
        for i in range(n-1):
            for j in range (i+1,n):
                sum = sum + abs(a[i,j])
        return 0.5*sum/n/(n-1)
    
    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/math.sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k): # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l): # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n): # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n): # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
            
    n = len(a)
    p = np.identity(n,float)
    for k in range(20):
        mu = threshold(a) # Compute new threshold
        for i in range(n-1): # Sweep through matrix
            for j in range(i+1,n):
                if abs(a[i,j]) >= mu:
                    rotate(a,p,i,j)
        if mu <= tol: return np.diagonal(a),p
    print('Jacobi method did not converge')
    
    
## module sortJacobi
'''
sortJacobi(lam,x).
Sorts the eigenvalues {lam} and eigenvectors [x]
in order of ascending eigenvalues.
'''

# def sortJacobi(lam,x):
#     n = len(lam)
#     for i in range(n-1):
#         index = i
#         val = lam[i]
#         for j in range(i+1,n):
#             if lam[j] < val:
#                 index = j
#                 val = lam[j]
#         if index != i:
#             swapRows(lam,i,index)
#             swapCols(x,i,index)
#

A = [[1, -1, -3], [-2, -4, 5], [2, 3, -1]]

print(A)
print(gaussElimin(A))
