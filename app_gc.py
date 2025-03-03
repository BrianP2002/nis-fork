import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *

np.random.seed(20250220)

a0_global = 0.1
a1_global = 1.0
iota_global = 0.5
lam_global = 0.1
G_global = 1.0
dt = 0.001

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

precomputed_derivatives = compute_derivatives()
precomputed_spatial_derivatives = compute_spatial_derivatives()

def B_func(x, y, z):
    if x > 0:
        return 1.0 + a0_global * np.sqrt(x) * np.cos(y - a1_global*z)
    else:
        return 1.0

def V_func(x, y, z):
    return np.sqrt(1.0 - lam_global * B_func(x, y, z))

def f_ode(x, y, z):
    B = B_func(x, y, z)
    invB = 1.0 / B
    factor = 2.0 / lam_global - B
    if x > 0:
        dBdy = -a0_global * np.sqrt(x) * np.sin(y - a1_global*z)
    else:
        dBdy = 0.0
    f1 = -invB * dBdy * factor
    if x > 0:
        dBdx = a0_global * (1.0 / (2.0 * np.sqrt(x))) * np.cos(y - a1_global*z)
    else:
        dBdx = 0.0
    f2 = invB * dBdx * factor + (iota_global * V_func(x, y, z) * B) / G_global
    f3 = (B * V_func(x, y, z)) / G_global
    return np.array([f1, f2, f3])

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
    nseg = 10
    T_seg = 2
    nseg_ps = 10
    nc = 3
    nus = 2
    par = "a1"
    par_lb = 0.9
    par_ub = 1.1
    step_size = 0.01
    par_arr = np.arange(par_lb, par_ub, step_size)
    J_arr = np.zeros(par_arr.shape)
    dJdpar_arr = np.zeros(par_arr.shape)
    for i, par_value in enumerate(par_arr):
        x = 0.1 * np.random.rand()  
        y = 2.0 * np.pi * np.random.rand()
        z = 2.0 * np.pi * np.random.rand()
        u0 = np.array([x, y, z])

        print(par_value, u0)

        J_val, dJdpar_val = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, par_value, Euler, fJJu)
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