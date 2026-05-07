"""
ITT Stage 1: Simulation validation (Lorenz vs white noise)
Uses improved core functions.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itt_core import (mutual_info_first_min, false_nearest_neighbor,
                      reconstruct, compute_lambda, lambda_significance)

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

def generate_lorenz(dt=0.01, tmax=100, init=(1,1,1)):
    t_eval = np.arange(0, tmax, dt)
    sol = solve_ivp(lorenz, (0,tmax), init, t_eval=t_eval, method='RK45')
    return sol.y[0]  # x component

if __name__ == "__main__":
    dt = 0.01
    tmax = 100
    fs = 1/dt

    lorenz_sig = generate_lorenz(dt, tmax)
    white_noise = np.random.randn(len(lorenz_sig))

    tau = mutual_info_first_min(lorenz_sig, max_lag=50)
    d = false_nearest_neighbor(lorenz_sig, tau=tau, max_dim=15)

    lam_l, thresh_l, p_l = lambda_significance(lorenz_sig, tau, d, dt)
    lam_w, thresh_w, p_w = lambda_significance(white_noise, tau, d, dt)

    print("=== Stage 1 Results ===")
    print(f"Lorenz: Λ={lam_l:.4f}, 95% threshold={thresh_l:.4f}, p={p_l:.4f} -> significant: {p_l<0.05}")
    print(f"Noise:  Λ={lam_w:.4f}, 95% threshold={thresh_w:.4f}, p={p_w:.4f} -> significant: {p_w<0.05}")

    # plot
    fig, ax = plt.subplots(2,2, figsize=(10,6))
    ax[0,0].plot(lorenz_sig[:2000], 'r-', lw=0.5)
    ax[0,0].set_title('Lorenz (x)')
    ax[0,1].hist(lorenz_sig, bins=50, color='red', alpha=0.7)
    ax[0,1].set_title('Lorenz distribution')
    ax[1,0].plot(white_noise[:2000], 'k-', lw=0.5)
    ax[1,0].set_title('White noise')
    ax[1,1].hist(white_noise, bins=50, color='gray', alpha=0.7)
    ax[1,1].set_title('Noise distribution')
    plt.tight_layout()
    plt.savefig('stage1_results.png')
    plt.show()
