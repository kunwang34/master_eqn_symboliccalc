from sympy import *

order = 5
for i in range(order):
    for j in range(order):
        var = 'x{}y{}'.format(i, j)
        var = var.replace('1','')
        var = var.replace('x0','')
        var = var.replace('y0','')
        if var == '':
            continue
        exec(var+'= symbols(\'' + var + '\')')
        
def Exmyn(eqn, m, n):
    ddt = 0
    for term in eqn:
        tmp = (term*i**m*j**n).subs(i, x-eqn.get(term)[0])
        tmp = tmp.subs(j, y-eqn.get(term)[1])
        ddt += tmp
    ddt = expand(ddt)
    for ii in range(order):
        for jj in range(order):
            old = 'x**{}*y**{}'.format(order-ii-1, order-jj-1)
            old = old.replace('**1', '')
            old = old.replace('x**0*', '')
            old = old.replace('*y**0', '')
            old = old.replace('y**0', '')
            if old == '':
                continue
            new = old.replace('*', '')       
            ddt = eval('ddt.subs({},{})'.format(old,new))
    return collect(ddt, x)


if __name__ == '__main__':
    i, j = symbols('i j')
    r,p,q,d = symbols('r p q d')
    '''
     phi(i+a0,j+b0)*f(i,j) write as {f(i,j):(a0,b0)}
    '''
    eqn = {r*p*(i-1):(-1,0), r*q*(i+1):(1,-2), (j+1)*d:(0,1), -(r*i+d*j):(0,0)}
    ex = Exmyn(eqn,1,0)
    ey = Exmyn(eqn,0,1)
    ex2 = Exmyn(eqn,2,0)
    ey2 = Exmyn(eqn,0,2)
    exy = Exmyn(eqn,1,1)
    
    ##if p+q==1
    print('dx/dt='+str(collect(expand(ex.subs(p,1-q)),x)))
    print('dy/dt='+str(collect(expand(ey.subs(p,1-q)),x)))
    print('dx2/dt='+str(collect(expand(ex2.subs(p,1-q)),x)))
    print('dy2/dt='+str(collect(expand(ey2.subs(p,1-q)),x)))
    print('dxy/dt='+str(collect(expand(exy.subs(p,1-q)),x)))
        
    from scipy.integrate import odeint
    import numpy as np
    import matplotlib.pyplot as plt

    def func(t, xx, r, p ,d):
        q = 1 - p
        x, y, x2, y2, xy = xx
        return np.array([eval(str(ex)),eval(str(ey)),eval(str(ex2)),\
                         eval(str(ey2)),eval(str(exy))]).astype('float')
                        
    t = np.linspace(0,10,20)
    x0 = [1,0,0,0,0]
    sol = odeint(func, x0, t, args=(1, 0.7, 0.05), tfirst=True)
    plt.plot(t, sol[:,0], label='Ex')
    plt.plot(t, sol[:,1], label='Ey')
    plt.legend()
    plt.show()
    plt.plot(t, sol[:,2]-sol[:,0]**2, label='Varx')
    plt.plot(t, sol[:,3]-sol[:,1]**2, label='Vary')
    plt.legend()
    plt.show()    
