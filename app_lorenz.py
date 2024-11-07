# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *


par = "beta"
par_lb = 1
par_ub = 6
step_size = 0.5
rho = 28
sigma = 10
beta = 8. / 3.
vpar = 1.5
par_arr = np.arange(par_lb, par_ub, step_size)
J_arr = np.zeros(par_arr.shape)
dJdpar_arr = np.zeros(par_arr.shape)


def ddt(uwvs, par, value):
    u = uwvs[0]
    x, y, z = u
    w = uwvs[1]
    vstar = uwvs[2]

    global rho, sigma, beta
    dfdrho = np.array([0, x, 0])
    dfdsigma = np.array([y - x, 0, 0])
    dfdbeta = np.array([0, 0, -z])
    dfdpar = None

    if par == "rho":
        dfdpar = dfdrho
        rho = value
    elif par == "sigma":
        sigma = value
        dfdpar = dfdsigma
    elif par == "beta":
        beta = value
        dfdpar = dfdbeta


    dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    Df = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
    dwdt = np.dot(Df, w.T)
    dvstardt = np.dot(Df, vstar) + dfdpar

    return [dudt, dwdt.T, dvstardt]


# parameter passing to nilss
dt = 0.001
nseg = 10 #number of segments on time interval
T_seg = 2 # length of each segment
nseg_ps = 10 # #segments of pre-smoothing
nc = 3 # number of component in u
nus = 2 # number of unstable direction, notice that nus=3 is illegal since we already mod off the neutral CLV
assert(nus < nc)


# functions passing to nilss
def fJJu(u, par, value):
    x, y, z = u
    global rho, sigma, beta
    if par == "rho":
        rho = value
    elif par == "sigma":
        sigma = value
    elif par == "beta":
        beta = value
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z]),\
            z, np.array([0,0,1])


def RK4(u, w, vstar, rho):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs, rho) 
    k1 = dt * ddt(uwvs + 0.5 * k0, rho)
    k2 = dt * ddt(uwvs + 0.5 * k1, rho)
    k3 = dt * ddt(uwvs + k2, rho)
    uwvs_new = uwvs + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return uwvs_new


def Euler(u, w, vstar, par, value):
    uwvs = [u, w, vstar]
    k0 = [dt*vec for vec in ddt(uwvs, par, value)] 
    uwvs_new = [v1+v2 for v1,v2 in zip(uwvs,k0)] 
    return uwvs_new


# main loop
for i, par_value in enumerate(par_arr):
    print(par_value)
    u0 =  (np.random.rand(nc)-0.5) * 100 + np.array([0,0,50]) #[-10, -10, 60]
    J, dJdpar = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, par_value, Euler, fJJu)
    J_arr[i] = J
    dJdpar_arr[i] = dJdpar


# plot preparations
plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


par_latex_map = {
    "rho": r"\rho",
    "sigma": r"\sigma",
    "beta": r"\beta"
}
par_label = par_latex_map.get(par, par)

# plot J vs parameter
plt.figure(figsize=[12, 12])
plt.subplot(2, 1, 1)
plt.plot(par_arr, J_arr)
plt.xlabel(fr'${par_label}$')
plt.ylabel(r'$\langle J \rangle$')

# plot dJ/dpar vs parameter
plt.subplot(2, 1, 2)
plt.plot(par_arr, dJdpar_arr)
plt.xlabel(fr'${par_label}$')
plt.ylabel(fr'$d \langle J \rangle / d {par_label}$')
plt.ylim([0, 2.0])
plt.savefig(f'lorenz_{par}.png')
plt.show()
