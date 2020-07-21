import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import ode
##--------sample from https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations--
#class Process:
#    def __init__(self):
#        super(Process, self).__init__()
#        self.Kp = 2.0
#        self.Tau = 5.0
#
#    def ode_model(self, z, t, u):
#        x = z[0]
#        y = z[1]
#        dxdt = (-x + u)/self.Kp
#        dydt = (-y + x)/self.Tau
#        dzdt = [dxdt,dydt]
#        return dzdt
##---------------  CSTR   ---------
class Process:
    def __init__(self):
        super(Process, self).__init__()
        self.Fih=1.0
        self.q = 100.0
        self.Caf = 1.0
        self.Tf = 350.0
        self.Tcf = 350.0
        self.V = 100.0
        self.ha = 7.0*10**5
        self.k0 = 7.2*10**10
        self.ER=9.95*10**3
        self.deltaH=-2*10**5
        self.ER=9.95*10**3
        self.Ro=1000.0
        self.Roc=1000.0
        self.Cp=1.0
        self.Cpc=1.0
        self.Fic=1.0
        self.alfah=0.01


    def ode_model(self, t, z, u):
        x = z[0]
        y = z[1]
        dxdt=(self.q/self.V)*(self.Caf-x)-self.k0*x*np.exp(-self.ER/y)*self.Fic
#        dxdt = (-x + u)/self.Kp
        hd=(1-self.alfah*t)*self.ha
        dydt=(self.q/self.V)*(self.Tf-y)+(-self.deltaH*self.k0*x/(self.Ro*self.Cp))*np.exp(-self.ER/y)*self.Fic+(self.Roc*self.Cpc/(self.Ro*self.Cp*self.V))*u*(1-np.exp(-hd/(u*self.Ro*self.Cpc)*self.Fih))*(self.Tcf-y)
#        dydt = (-y + x)/self.Tau
        dzdt = [dxdt,dydt]
        return dzdt
##------------------


def simulate_process1(f, t, u, x0):
    ys = odeint(f, x0, t, args=(u,))
    return t,ys
#----------------------

def simulate_process(f, t, u, x0):
    r = ode(f).set_integrator('zvode', method='bdf')
    xk = x0
    t_values = [t[0]]
    x_values = [np.array(xk)]

    deltas_t = t[1:] - t[:-1]
    for it in range(len(deltas_t)):
        r.set_initial_value(xk, t[it]).set_f_params(u[it])
        xk = np.real(r.integrate(r.t + deltas_t[it]))
        if r.successful():
            t_values.append(r.t)
            x_values.append(xk)

    return t_values, x_values


def pseudo_random_pulse(num_value, period, average_value, min_val, max_val):
    sig = np.zeros(num_value)
    deviation = abs(max_val - min_val)
    j = 0
    new_val = np.clip(average_value + deviation*(np.random.rand()-0.5), min_val, max_val)
    for k in range(num_value):
        sig[k] = new_val
        j = j + 1
        if j == period:
            new_val = np.clip(average_value + deviation*(np.random.rand()-0.5), min_val, max_val)
            j = 0
    return sig

# initial condition
#z0 = [0,0]
z0=[0.07,441.0]
# number of time points
#n = 401

# time points
#t = np.linspace(0,40,n)

# step input
#u = np.zeros(n)


dt = 0.01
tf = 60
n = int(tf/dt)
t = np.linspace(0, tf-dt, num=n)

u = pseudo_random_pulse(n, int(n/20), 100, 95, 105)
# u = 100*np.ones(n)

# change to 2.0 at time = 5.0
#u[51:] = 2.0

# store solution
x = np.empty_like(t)
y = np.empty_like(t)
# record initial conditions
x[0] = z0[0]
y[0] = z0[1]
p = Process()
# solve ODE
t_sim, z_sim = simulate_process(f=p.ode_model, t=t, u=u, x0=z0)

x_sim = [z_[0] for z_ in z_sim]
y_sim = [z_[1] for z_ in z_sim]

# plot results
#plt.plot(t,u,'g:',label='u(t)')
plt.subplot(311)
plt.plot(t,u,'b-',label='u(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.xlabel('time')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(t_sim,x_sim,'b-',label='x(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.subplot(313)
plt.plot(t_sim,y_sim,'b-',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()