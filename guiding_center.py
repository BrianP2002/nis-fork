import jax
import jax.numpy as jnp
import diffrax
import numpy as np

def B(x, y, z, params):
    return 1.0 + params["a0"] * jnp.sqrt(x) * jnp.cos(y - params["a1"] * z)

def V(x, y, z, params):
    return jnp.sqrt(1.0 - params["lambda"] * B(x, y, z, params))

def f(state, t, params):
    x, y, z = state
    Bb = B(x, y, z, params)
    dBx = params["a0"] * (1.0 / (2.0 * jnp.sqrt(x))) * jnp.cos(y - params["a1"] * z)
    dBy = -params["a0"] * jnp.sqrt(x) * jnp.sin(y - params["a1"] * z)
    Vb = V(x, y, z, params)
    factor = (2.0 / params["lambda"] - Bb)
    dx = -(1.0 / Bb) * dBy * factor
    dy = (1.0 / Bb) * dBx * factor + (params["iota"] * Vb * Bb) / params["G"]
    dz = (Vb * Bb) / params["G"]
    return jnp.array([dx, dy, dz])

def ode_system(t, y, args):
    return f(y, t, args)

params = {
    "a0": 0.1,
    "a1": 0.95,
    "iota": 0.5,
    "G": 1.0,
    "lambda": 0.1
}
term = diffrax.ODETerm(ode_system)
solver = diffrax.Tsit5()
x = 0.1 * np.random.rand()  
y = 2.0 * np.pi * np.random.rand()
z = 2.0 * np.pi * np.random.rand()
y0 = jnp.array([x, y, z])
t0, t1 = 0, 20
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 200))
sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=1e-3, y0=y0, args=params, max_steps=100000, saveat = saveat)
print(sol.ys)