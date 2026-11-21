"""Random network: connectivity, explicit time steppers, batched simulation loop.

Dynamics (eq:system with G = 0):  dx/dt = -x + J phi(x),   phi = tanh.
State x has shape (N, B): B independent trajectories advanced together so that
J @ phi(x) is a matrix-matrix product.
"""
import numpy as np
from .backend import xp, to_device


def make_J(N, sigma2, seed):
    """i.i.d. Gaussian J_ij with variance sigma2/N, generated on the CPU (backend independent)."""
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, np.sqrt(sigma2 / N), size=(N, N))
    return to_device(J)


def rhs(x, J):
    return -x + J @ xp.tanh(x)


def step_euler(x, J, dt):
    return x + dt * rhs(x, J)


def step_heun(x, J, dt):
    k1 = rhs(x, J)
    k2 = rhs(x + dt * k1, J)
    return x + 0.5 * dt * (k1 + k2)


def step_rk4(x, J, dt):
    k1 = rhs(x, J)
    k2 = rhs(x + 0.5 * dt * k1, J)
    k3 = rhs(x + 0.5 * dt * k2, J)
    k4 = rhs(x + dt * k3, J)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


STEPPERS = {"euler": step_euler, "heun": step_heun, "rk4": step_rk4}


def initial_state(N, B, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return to_device(rng.normal(0.0, scale, size=(N, B)))


def simulate(x, J, dt, n_steps, stepper="rk4", hooks=()):
    """Advance x for n_steps. After each step, call hook.update(t, x, phi) for every hook
    (t = 0 .. n_steps-1). Returns the final state."""
    step = STEPPERS[stepper]
    for t in range(n_steps):
        x = step(x, J, dt)
        if hooks:
            phi = xp.tanh(x)
            for h in hooks:
                h.update(t, x, phi)
    return x


def lyapunov_exponent(x, J, dt, T, stepper="rk4", delta0=1e-7, renorm_every=20, seed=0):
    """Largest Lyapunov exponent for each trajectory in the batch (N, B), by tracking a small
    perturbation with periodic renormalization (Benettin). Returns (lambda_per_trajectory, x_final)."""
    step = STEPPERS[stepper]
    rng = np.random.default_rng(seed)
    N, B = x.shape
    d = to_device(rng.normal(size=(N, B)))
    d *= delta0 / xp.sqrt(xp.sum(d ** 2, axis=0))
    y = x + d
    n_steps = int(round(T / dt))
    logsum = xp.zeros(B)
    for t in range(1, n_steps + 1):
        xy = step(xp.concatenate([x, y], axis=1), J, dt)
        x, y = xy[:, :B], xy[:, B:]
        if t % renorm_every == 0:
            d = y - x
            nrm = xp.sqrt(xp.sum(d ** 2, axis=0))
            logsum += xp.log(nrm / delta0)
            y = x + d * (delta0 / nrm)
    return logsum / (n_steps * dt), x


def trajectory_decorrelation(x):
    """Mean |correlation coefficient| between distinct trajectories of the batch (N, B) at one time.
    Independent chaotic trajectories give ~ 1/sqrt(N)."""
    xc = x - xp.mean(x, axis=0)
    xc = xc / xp.sqrt(xp.sum(xc ** 2, axis=0))
    c = xc.T @ xc
    B = x.shape[1]
    off = ~np.eye(B, dtype=bool)
    return float(xp.mean(xp.abs(c[xp.asarray(off)])))
