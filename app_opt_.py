import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from nilss import nilss
from app_lorenz import RK4, Euler, fJJu

np.random.seed(20241202)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMETERS = {
    "sigma": (5, 15),
    "rho": (15, 30),
    "beta": (1, 3)
}

def precompute_J(par_arr, par_name, dt, nseg, T_seg, nseg_ps, nus, integrator, fJJu, nc):
    J_arr = []
    for par_value in par_arr:
        u0 = (np.random.rand(nc) - 0.5) * 100 + np.array([0, 0, 50])
        J, _ = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_name,
            s=par_value,
            integrator=integrator,
            fJJu=fJJu
        )
        J_arr.append(J)
    return np.array(J_arr)

def optimize_lorenz(par_name, par_bounds, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, regular_jac=False):
    def objective(par_value):
        J, dJdpar = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_name,
            s=par_value,
            integrator=integrator,
            fJJu=fJJu
        )
        return J, dJdpar

    def scipy_objective(par_value):
        J, dJdpar = objective(par_value[0])
        if regular_jac:
            return J
        return J, np.array([dJdpar])

    jac_method = '2-point' if regular_jac else True

    result = minimize(
        fun=scipy_objective,
        x0=[(par_bounds[0] + par_bounds[1]) / 2],
        jac=jac_method,
        method='L-BFGS-B',
        bounds=[par_bounds],
        options={
            # 'maxiter': 20,
            # 'ftol': 1e-2,
            'disp': True
        }
    )

    return result

def plot_trajectory(par_arr, J_arr, nilss_opt, regular_opt, par_name):
    plt.figure(figsize=[10, 6])
    plt.plot(par_arr, J_arr, label='Trajectory of J', linewidth=2)
    plt.axvline(nilss_opt, color='r', linestyle='--', label='NILSS Optimization')
    plt.axvline(regular_opt, color='b', linestyle='--', label='Regular Optimization ("2-point")')
    plt.xlabel(fr'${par_name}$', fontsize=16)
    plt.ylabel(r'$\langle J \rangle$', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'optimization_trajectory_{par_name}.png'))

if __name__ == "__main__":
    par_name = "rho"
    par_bounds = PARAMETERS[par_name]

    nc = 3
    nus = 2
    dt = 0.001
    nseg = 10
    T_seg = 2.0
    nseg_ps = 10
    integrator = Euler
    step_size = 0.2
    par_arr = np.arange(par_bounds[0], par_bounds[1] + step_size, step_size)

    J_arr = precompute_J(
        par_arr=par_arr,
        par_name=par_name,
        dt=dt,
        nseg=nseg,
        T_seg=T_seg,
        nseg_ps=nseg_ps,
        nus=nus,
        integrator=integrator,
        fJJu=fJJu,
        nc=nc
    )

    u0 = (np.random.rand(nc) - 0.5) * 100 + np.array([0, 0, 50])

    nilss_result = optimize_lorenz(
        par_name=par_name,
        par_bounds=par_bounds,
        u0=u0,
        nus=nus,
        dt=dt,
        nseg=nseg,
        T_seg=T_seg,
        nseg_ps=nseg_ps,
        integrator=integrator,
        fJJu=fJJu,
        regular_jac=False
    )

    regular_result = optimize_lorenz(
        par_name=par_name,
        par_bounds=par_bounds,
        u0=u0,
        nus=nus,
        dt=dt,
        nseg=nseg,
        T_seg=T_seg,
        nseg_ps=nseg_ps,
        integrator=integrator,
        fJJu=fJJu,
        regular_jac=True
    )

    print(f'{par_name} results generated')

    nilss_opt = nilss_result.x[0]
    regular_opt = regular_result.x[0]

    nilss_J = nilss_result.fun
    regular_J = regular_result.fun

    output_file = os.path.join(RESULTS_DIR, f'optimization_results_{par_name}.txt')
    with open(output_file, 'w') as f:
        f.write(f"Parameter: {par_name}\n")
        f.write(f"NILSS Optimization:\n  Optimized Value: {nilss_opt:.4f}\n  Corresponding J: {nilss_J:.4f}\n")
        f.write(f"Regular Optimization (2-point):\n  Optimized Value: {regular_opt:.4f}\n  Corresponding J: {regular_J:.4f}\n")

    plot_trajectory(par_arr, J_arr, nilss_opt, regular_opt, par_name)