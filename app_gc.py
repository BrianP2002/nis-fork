import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *

a0_global = 0.1
a1_global = 1.0
iota_global = 0.5
lam_global = 0.1
G_global = 1.0

def B_func(x, y, z):
    return 1.0 + a0_global * np.sqrt(x) * np.cos(y - a1_global * z)

def V_func(x, y, z):
    return np.sqrt(1.0 - lam_global * B_func(x, y, z))

def ddt(uwvs, par, value):
    set_params(par, value)

    u = np.asarray(uwvs[0])
    x, y, z = u
    w = uwvs[1]
    vstar = uwvs[2]

    Bb = B_func(x, y, z)
    # partials of B wrt (x,y,z)
    if x > 0:
        dBdx = a0_global * (1.0 / (2.0 * np.sqrt(x))) * np.cos(y - a1_global * z)
    else:
        dBdx = 0.0
    dBdy = -a0_global * np.sqrt(x) * np.sin(y - a1_global * z)
    dBdz =  a0_global * np.sqrt(x) * a1_global * np.sin(y - a1_global * z)
    Vb = V_func(x, y, z)

    factor = (2.0 / lam_global - Bb)
    invB = 1.0 / Bb

    dxdt = -(invB) * dBdy * factor
    dydt =  (invB) * dBdx * factor + (iota_global * Vb * Bb) / G_global
    dzdt =  (Vb * Bb) / G_global
    dudt = np.array([dxdt, dydt, dzdt])

    # Finite-difference for Df w.r.t. (x,y,z)
    base_f = np.array([dxdt, dydt, dzdt])
    eps = 1e-8
    def f_xyz(xx, yy, zz):
        Bb_ = B_func(xx, yy, zz)
        invB_ = 1.0 / Bb_
        factor_ = (2.0 / lam_global - Bb_)
        Vb_ = np.sqrt(1.0 - lam_global*Bb_)
        dx_ = -(invB_) * ( -a0_global*np.sqrt(xx)*np.sin(yy - a1_global*zz ) ) * factor_
        dy_ =  (invB_) * ( a0_global*(1.0/(2.0*np.sqrt(xx)))*np.cos(yy - a1_global*zz) if xx>0 else 0.0 ) * factor_ + (iota_global*Vb_*Bb_)/G_global
        dz_ =  (Vb_*Bb_)/G_global
        return np.array([dx_, dy_, dz_])

    Df = np.zeros((3, 3))
    for j, shift in enumerate([(eps,0,0),(0,eps,0),(0,0,eps)]):
        xx = x + shift[0]
        yy = y + shift[1]
        zz = z + shift[2]
        f_shift = f_xyz(xx, yy, zz)
        Df[:, j] = (f_shift - base_f)/eps

    eps_p = 1e-8
    set_params(par, value + eps_p)
    f_shift_p = f_xyz(x, y, z)
    set_params(par, value)  # reset
    dfdpar = (f_shift_p - base_f)/eps_p

    # w' = Df * w^T
    dwdt = np.dot(Df, w.T)
    # vstar' = Df * vstar + dfdpar
    dvstardt = np.dot(Df, vstar) + dfdpar
    return [dudt, dwdt.T, dvstardt]

def fJJu(u, par, value):
    set_params(par, value)
    x, y, z = u
    Bb = B_func(x, y, z)
    invB = 1.0 / Bb
    factor = (2.0 / lam_global - Bb)
    dxdt = -(invB) * (-a0_global*np.sqrt(x)*np.sin(y - a1_global*z)) * factor
    dydt =  (invB) * (a0_global*(1.0/(2.0*np.sqrt(x)))*np.cos(y - a1_global*z) if x>0 else 0.0 ) * factor + (iota_global*np.sqrt(1.0 - lam_global*Bb)*Bb)/G_global
    dzdt =  (np.sqrt(1.0 - lam_global*Bb)*Bb)/G_global
    f_val = np.array([dxdt, dydt, dzdt])
    J = x
    return f_val, J, np.array([1.0, 0.0, 0.0])

def RK4(u, w, vstar, par, value):
    dt_ = 0.001
    uwvs = [np.array(u), np.array(w), np.array(vstar)]
    k0 = ddt(uwvs, par, value)
    k1 = ddt([u + 0.5*dt_*k0[0], w + 0.5*dt_*k0[1], vstar + 0.5*dt_*k0[2]], par, value)
    k2 = ddt([u + 0.5*dt_*k1[0], w + 0.5*dt_*k1[1], vstar + 0.5*dt_*k1[2]], par, value)
    k3 = ddt([u + dt_*k2[0],      w + dt_*k2[1],      vstar + dt_*k2[2]],      par, value)
    u_new     = u     + (dt_/6.0)*(k0[0] + 2*k1[0] + 2*k2[0] + k3[0])
    w_new     = w     + (dt_/6.0)*(k0[1] + 2*k1[1] + 2*k2[1] + k3[1])
    vstar_new = vstar + (dt_/6.0)*(k0[2] + 2*k1[2] + 2*k2[2] + k3[2])
    return u_new, w_new, vstar_new

def Euler(u, w, vstar, par, value):
    dt_ = 0.001
    k0 = ddt([u, w, vstar], par, value)
    return [
        u     + dt_*k0[0],
        w     + dt_*k0[1],
        vstar + dt_*k0[2],
    ]

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

def main():
    dt = 0.001
    nseg = 10
    T_seg = 2
    nseg_ps = 10
    nc = 3
    nus = 2

    par = "a0"

    par_lb = 0.0
    par_ub = 1.0
    step_size = 0.2
    par_arr = np.arange(par_lb, par_ub + step_size, step_size)

    J_arr = np.zeros(par_arr.shape)
    dJdpar_arr = np.zeros(par_arr.shape)

    for i, par_value in enumerate(par_arr):
        u0 = (np.random.rand(nc) - 0.5)*2.0 + np.array([1.0, 0.0, 0.0])
        J, dJdpar = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, par_value, Euler, fJJu)
        J_arr[i] = J
        dJdpar_arr[i] = dJdpar

    plt.rc('axes', labelsize='xx-large', labelpad=12)
    plt.rc('xtick', labelsize='xx-large')
    plt.rc('ytick', labelsize='xx-large')
    plt.rc('legend', fontsize='xx-large')
    plt.rc('font', family='sans-serif')

    plt.figure(figsize=[12, 12])
    plt.subplot(2, 1, 1)
    plt.plot(par_arr, J_arr, marker='o')
    plt.xlabel(rf'${par}$')
    plt.ylabel(r'$\langle J \rangle = \langle x \rangle$')

    plt.subplot(2, 1, 2)
    plt.plot(par_arr, dJdpar_arr, marker='s')
    plt.xlabel(rf'${par}$')
    plt.ylabel(rf'$\frac{{d\langle J \rangle}}{{d {par}}}$')

    plt.savefig(f'guiding_center_{par}.png')
    plt.show()

if __name__ == '__main__':
    main()
