# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *

np.random.seed(20241202)

# Parameter Selection
par = "sigma"
par_lb = 5
par_ub = 15
step_size = 0.2
rho = 28
sigma = 10
beta = 8. / 3.
vpar = 1.5
par_arr = np.arange(par_lb, par_ub + step_size, step_size)
J_arr = np.zeros(par_arr.shape)
dJdpar_arr = np.zeros(par_arr.shape)


def ddt(uwvs, par, value):
    assert len(uwvs) == 3, f"uwvs must have 3 elements, got {len(uwvs)}"
    u = np.asarray(uwvs[0])
    assert u.shape == (3,), f"u must have shape (3,), got {u.shape}"
    x, y, z = u
    w = uwvs[1]
    vstar = uwvs[2]

    global rho, sigma, beta
    dfdsigma = np.array([y - x, 0, 0])
    dfdrho = np.array([0, x, 0])
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
    Df = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
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


def RK4(u, w, vstar, par, value):
    # Ensure that u, w, and vstar are numpy arrays
    u = np.array(u)
    w = np.array(w)
    vstar = np.array(vstar)
    
    # Keep uwvs as a list, not a numpy array
    uwvs = [u, w, vstar]  # Keep as a list to avoid creating a ragged array
    
    # Compute k0
    k0_u, k0_w, k0_vstar = ddt(uwvs, par, value)
    k0_u, k0_w, k0_vstar = dt * np.array(k0_u), dt * np.array(k0_w), dt * np.array(k0_vstar)

    # Compute k1
    k1_u, k1_w, k1_vstar = ddt([u + 0.5 * k0_u, w + 0.5 * k0_w, vstar + 0.5 * k0_vstar], par, value)
    k1_u, k1_w, k1_vstar = dt * np.array(k1_u), dt * np.array(k1_w), dt * np.array(k1_vstar)

    # Compute k2
    k2_u, k2_w, k2_vstar = ddt([u + 0.5 * k1_u, w + 0.5 * k1_w, vstar + 0.5 * k1_vstar], par, value)
    k2_u, k2_w, k2_vstar = dt * np.array(k2_u), dt * np.array(k2_w), dt * np.array(k2_vstar)

    # Compute k3
    k3_u, k3_w, k3_vstar = ddt([u + k2_u, w + k2_w, vstar + k2_vstar], par, value)
    k3_u, k3_w, k3_vstar = dt * np.array(k3_u), dt * np.array(k3_w), dt * np.array(k3_vstar)

    # Update state vectors
    u_new = u + (k0_u + 2 * k1_u + 2 * k2_u + k3_u) / 6.0
    w_new = w + (k0_w + 2 * k1_w + 2 * k2_w + k3_w) / 6.0
    vstar_new = vstar + (k0_vstar + 2 * k1_vstar + 2 * k2_vstar + k3_vstar) / 6.0

    return u_new, w_new, vstar_new


def Euler(u, w, vstar, par, value):
    uwvs = [u, w, vstar]
    k0 = [dt*vec for vec in ddt(uwvs, par, value)] 
    uwvs_new = [v1+v2 for v1,v2 in zip(uwvs,k0)] 
    return uwvs_new

def main():
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
    plt.savefig(f'lorenz_{par}.png')
    plt.show()

if __name__ == '__main__':
    main()