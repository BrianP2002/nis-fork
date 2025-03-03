import os
import numpy as np
from scipy.optimize import minimize
from nilss import nilss
from app_lorenz import RK4, Euler, fJJu

np.random.seed(20250212)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def optimize_lorenz_nilss(par_name, par_bounds, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter=100, tol=1e-6):
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
        return J , np.array([dJdpar])

    # print(scipy_objective([(par_bounds[0] + par_bounds[1]) / 2]))
    # exit()
    result = minimize(
        fun=scipy_objective,
        x0=[(par_bounds[0] + par_bounds[1]) / 2],
        jac=True,
        method='L-BFGS-B',
        bounds=[par_bounds],
        options={
            # 'maxiter': 20,
            # 'ftol': 1e-2,
            'disp': True
        }
    )

    return result

if __name__ == "__main__":
    par_name = "rho"
    par_bounds = (15, 30)
    nc = 3
    u0 = (np.random.rand(nc)-0.5) * 100 + np.array([0,0,50])
    nus = 2
    dt = 0.001
    nseg = 10
    T_seg = 2.0
    nseg_ps = 10
    integrator = Euler

    result = optimize_lorenz_nilss(
        par_name, par_bounds, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu
    )

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    text_result_path = os.path.join(results_dir, f"optimization_results_{par_name}.txt")
    with open(text_result_path, "w") as f:
        f.write("Optimization Result:\n")
        f.write(f"  Optimal {par_name}: {result.x[0]:.4f}\n")
        f.write(f"  Minimum cost J: {result.fun:.4e}\n")