import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *

np.random.seed(20250301)

a0_global = 0.1
a1_global = 1.0
iota_global = 0.5
lam_global = 0.1
G_global = 1.0
dt = 0.001

precomputed_derivatives = None
precomputed_spatial_derivatives = None

def set_params(par, value):
    global a0_global, a1_global, iota_global, lam_global, G_global
    if par == "a0":
        a0_global = value
    elif par == "a1":
        a1_global = value
    elif par == "iota":
        iota_global = value
    elif par == "lambda":
        lam_global = value
    elif par == "G":
        G_global = value

def compute_derivatives():
    x, y, z, a0, a1, iota, G, lambda_ = sp.symbols('x y z a0 a1 iota G lambda', real=True)
    B = 1 + a0 * sp.sqrt(x) * sp.cos(y - a1*z)
    V = sp.sqrt(1 - lambda_ * B)

    df_dx = - (1/B) * sp.diff(B, y) * (2/lambda_ - B)
    df_dy = (1/B) * sp.diff(B, x) * (2/lambda_ - B) + (iota * V * B) / G
    df_dz = (V * B) / G

    parameters = (x, y, z, a0, a1, iota, G, lambda_)
    derivatives = {
        'a0': [sp.lambdify(parameters, sp.diff(df_dx, a0), "numpy"),
               sp.lambdify(parameters, sp.diff(df_dy, a0), "numpy"),
               sp.lambdify(parameters, sp.diff(df_dz, a0), "numpy")],
        
        'a1': [sp.lambdify(parameters, sp.diff(df_dx, a1), "numpy"),
               sp.lambdify(parameters, sp.diff(df_dy, a1), "numpy"),
               sp.lambdify(parameters, sp.diff(df_dz, a1), "numpy")],
        
        'iota': [sp.lambdify(parameters, sp.diff(df_dx, iota), "numpy"),
                 sp.lambdify(parameters, sp.diff(df_dy, iota), "numpy"),
                 sp.lambdify(parameters, sp.diff(df_dz, iota), "numpy")],
        
        'G': [sp.lambdify(parameters, sp.diff(df_dx, G), "numpy"),
              sp.lambdify(parameters, sp.diff(df_dy, G), "numpy"),
              sp.lambdify(parameters, sp.diff(df_dz, G), "numpy")],
        
        'lambda': [sp.lambdify(parameters, sp.diff(df_dx, lambda_), "numpy"),
                   sp.lambdify(parameters, sp.diff(df_dy, lambda_), "numpy"),
                   sp.lambdify(parameters, sp.diff(df_dz, lambda_), "numpy")]
    }
    return derivatives

def compute_spatial_derivatives():
    x, y, z, a0, a1, iota, G, lambda_ = sp.symbols('x y z a0 a1 iota G lambda', real=True)
    B = 1 + a0 * sp.sqrt(x) * sp.cos(y - a1*z)
    V = sp.sqrt(1 - lambda_ * B)

    df_dx = - (1/B) * sp.diff(B, y) * (2/lambda_ - B)
    df_dy = (1/B) * sp.diff(B, x) * (2/lambda_ - B) + (iota * V * B) / G
    df_dz = (V * B) / G

    parameters = (x, y, z, a0, a1, iota, G, lambda_)
    spatial_derivatives = np.array([
        [sp.lambdify(parameters, sp.diff(df_dx, x), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dy, x), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dz, x), "numpy")],

        [sp.lambdify(parameters, sp.diff(df_dx, y), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dy, y), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dz, y), "numpy")],

        [sp.lambdify(parameters, sp.diff(df_dx, z), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dy, z), "numpy"),
         sp.lambdify(parameters, sp.diff(df_dz, z), "numpy")]
    ])
    return spatial_derivatives

def B_func(x, y, z):
    return 1.0 + a0_global * np.sqrt(x) * np.cos(y - a1_global*z)

def V_func(x, y, z):
    return np.sqrt(1.0 - lam_global * B_func(x, y, z))

def f_ode(x, y, z):
    B = B_func(x, y, z)
    invB = 1.0 / B
    factor = 2.0 / lam_global - B
    
    dBdy = -a0_global * np.sqrt(x) * np.sin(y - a1_global*z)
    dfdx = -invB * dBdy * factor
    
    dBdx = a0_global * (1.0 / (2.0 * np.sqrt(x))) * np.cos(y - a1_global*z)
    dfdy = invB * dBdx * factor + (iota_global * V_func(x, y, z) * B) / G_global
    
    dfdz = (B * V_func(x, y, z)) / G_global
    
    return np.array([dfdx, dfdy, dfdz])

def ddt(uwvs, par, value):
    set_params(par, value)
    u = np.asarray(uwvs[0])
    w = uwvs[1]
    vstar = uwvs[2]
    x, y, z = u
    
    param_values = (x, y, z, a0_global, a1_global, iota_global, G_global, lam_global)

    dfdpar = np.array([func(*param_values) for func in precomputed_derivatives[par]])
    Df = np.array([[func(*param_values) for func in row] for row in precomputed_spatial_derivatives])

    dudt = f_ode(x, y, z)
    dwdt = np.dot(Df, w.T)
    dvstardt = np.dot(Df, vstar) + dfdpar
    
    return [dudt, dwdt.T, dvstardt]


def fJJu(u, par, value):
    set_params(par, value)
    x, y, z = u
    f_val = f_ode(x, y, z)
    return f_val, x, np.array([1, 0, 0])

def RK4(u, w, vstar, par, value):
    uwvs = [u, w, vstar]

    k0 = [dt * vec for vec in ddt(uwvs, par, value)]
    k1 = [dt * vec for vec in ddt([uwvs[i] + 0.5*k0[i] for i in range(3)], par, value)]
    k2 = [dt * vec for vec in ddt([uwvs[i] + 0.5*k1[i] for i in range(3)], par, value)]
    k3 = [dt * vec for vec in ddt([uwvs[i] + k2[i] for i in range(3)], par, value)]
    uwvs_new = [uwvs[i] + (k0[i] + 2*k1[i] + 2*k2[i] + k3[i]) / 6.0 for i in range(3)]
    return uwvs_new


def Euler(u, w, vstar, par, value):
    uwvs = [u, w, vstar]
    k0 = [dt*vec for vec in ddt(uwvs, par, value)] 
    uwvs_new = [v1+v2 for v1,v2 in zip(uwvs,k0)] 
    return uwvs_new

def main():
    nseg = 100
    T_seg = 2
    nseg_ps = 100
    nc = 3
    nus = 2
    
    global precomputed_derivatives, precomputed_spatial_derivatives
    precomputed_derivatives = compute_derivatives()
    precomputed_spatial_derivatives = compute_spatial_derivatives()
    
    par = "a1"
    par_lb = 0.9
    par_ub = 1.1
    step_size = 0.01
    num_steps = int(round((par_ub - par_lb) / step_size)) + 1
    par_arr = np.linspace(par_lb, par_ub, num_steps)
    
    J_arr = np.zeros(par_arr.shape)
    dJdpar_arr = np.zeros(par_arr.shape)
    
    for i, par_value in enumerate(par_arr):
        x = 0.1 * np.random.rand()  
        y = 2.0 * np.pi * np.random.rand()
        z = 2.0 * np.pi * np.random.rand()
        u0 = np.array([x, y, z])
        print(f'{par}={par_value}, u0={u0}')

        J_val, dJdpar_val = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, par_value, RK4, fJJu)
        J_arr[i] = J_val
        dJdpar_arr[i] = dJdpar_val
    
    plt.figure(figsize=[12, 12])
    plt.subplot(2,1,1)
    plt.plot(par_arr, J_arr, marker='o')
    plt.xlabel(rf'${par}$')
    plt.ylabel(r'$\langle x \rangle$')
    plt.subplot(2,1,2)
    plt.plot(par_arr, dJdpar_arr, marker='s')
    plt.xlabel(rf'${par}$')
    plt.ylabel(rf'$\frac{{d\langle J \rangle}}{{d {par}}}$')
    plt.savefig(f'guiding_center_{par}.png')

if __name__ == '__main__':
    main()