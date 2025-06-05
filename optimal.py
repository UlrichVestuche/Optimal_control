import numpy as np
import matplotlib.pyplot as plt

import tqdm
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------- #
#  RHS with adaptive spectrum                                          #
# --------------------------------------------------------------------- #
def rhs_T(S: float, T: float, nu: float) -> float:
    if S < 1e-12:
        k = np.arange(1, 200_001, dtype=float)
        p = k**(-nu)
        return -0.5 * T * T * (p**3).sum() / (p**2).sum()
    k_max = max(50, int((25.0 / S)**(1/nu)) + 1)
    k     = np.arange(1, k_max+1, dtype=float)
    p     = k**(-nu)
    e     = np.exp(-p * S)
    G2    = np.sum(p**2 * e)
    G3    = np.sum(p**3 * e)
    return -0.5 * T * T * (G3 / G2)

# --------------------------------------------------------------------- #
#  Integrator: returns t, T(t), and S(t)                                #
# --------------------------------------------------------------------- #
def solve_profile(T0: float, S0: float, nu: float,
                  dt: float = 5e-4, tol_S: float = 1e-6):
    """
    Integrate dS/dt = -T, dT/dt = rhs_T(S,T,nu) using solve_ivp.
    Stops when S drops below tol_S.
    """
    TOL_S = tol_S

    def ode_system(t, y):
        S, T = y
        return [-T, rhs_T(S, T, nu)]

    def hit_zero(t, y):
        return y[0] - TOL_S
    hit_zero.terminal = True
    hit_zero.direction = -1

    sol = solve_ivp(
        ode_system,
        (0, np.inf),
        [S0, T0],
        method='RK23',      # cheaper explicit solver for modest accuracy
        args=(),
        rtol=1e-4,          # relaxed tolerances speed up integration
        atol=1e-6,
        events=hit_zero,
        max_step=0.01       # allow larger steps; adapt down if needed
    )

    ts = sol.t
    Ss = sol.y[0]
    Ts = sol.y[1]
    return ts, Ts, Ss

# --------------------------------------------------------------------- #
#  Compute total loss L(t)                                             #
# --------------------------------------------------------------------- #
def compute_total_loss(ts, Ts, Ss, nu, T0, S0, k_max_mode=50):
    """
    Vectorised evaluation of L(t)=Σ_k ℓ_k(t) for k = 1 … k_max_mode.
    Avoids the Python‑level loop over k; ~30‑50× faster for 50 modes.
    """
    # build actual time increments for non-uniform ts
    dt_array = np.diff(ts, prepend=0.0)  # ts[0]=0 → dt_array[0]=0
        
    p = np.arange(1, k_max_mode + 1, dtype=float)**(-nu)          # (k,)
    l0 = 0.5 * p * T0                                             # (k,)
    exp_factor = np.exp(p[:, None] * (S0 - Ss))                   # (k, N)
    integrand  = 0.5 * (p**2)[:, None] * Ts**2 * exp_factor       # (k, N)
    I = np.cumsum(integrand * dt_array[None, :], axis=1)                         # (k, N)

    losses = np.exp(-p[:, None] * (S0 - Ss)) * I + np.exp(-p[:, None] * (S0 - Ss)) * l0[:, None]  # (k, N)
    return losses.sum(axis=0)


# --------------------------------------------------------------------- #
#  Compute total loss L at the final time point only                   #
# --------------------------------------------------------------------- #
def compute_final_loss(ts, Ts, Ss, nu, T0, S0, k_max_mode=50):
    """
    Compute the total loss L at the final time point only.
    """
    # build time increments
    dt_array = np.diff(ts, prepend=0.0)  # ts[0]=0 → dt_array[0]=0

    # per-mode parameters
    p = np.arange(1, k_max_mode + 1, dtype=float) ** (-nu)  # (k,)
    l0 = 0.5 * p * T0                                     # (k,)

    # exponential factor over all times (k, N)
    exp_factor = np.exp(p[:, None] * (Ss[-1] - Ss))

    # integrand over all times (k, N)
    integrand = 0.5 * (p**2)[:, None] * Ts**2 * exp_factor

    # perform time integral to get I_k at final time (k,)
    I_k_final = np.sum(integrand * dt_array[None, :], axis=1)

    # damping factor at final time (k,)
    damping_final = np.exp(-p * (S0 - Ss[-1]))

    # per-mode losses at final time and sum
    losses_final = I_k_final + l0*damping_final
    return losses_final.sum()


# --------------------------------------------------------------------- #
#  MAIN: single-case plotting                                          #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # parameters
    nu  = 2.5
    T0  = 1.0
    S0  = 500.0
    dt  = 5e-4
    k_modes = 100

    # solve
    ts, Ts, Ss = solve_profile(T0, S0, nu, dt)
    t_max      = ts[-1]
    T_end      = Ts[-1]

    # analytic ansatz and late exponential
    u_analytic = (1 - ts/t_max) ** (2*nu - 1)
    k_big   = np.arange(1, 200_001, dtype=float)
    p_big   = k_big**(-nu)
    ratio_c = np.sum(p_big**3)/np.sum(p_big**2)
    T_exp   = T_end * np.exp(0.5 * ratio_c * T_end * (t_max - ts))

    # total loss
    L_t = compute_total_loss(ts, Ts, Ss, nu, T0, S0, k_modes)
    # analytic-profile stock S_an(t) from u_analytic
    Ss_an = S0 - (t_max/(2*nu)) * (1 - (1 - ts/t_max)**(2*nu))
    # compute loss for the analytic temperature profile
    L_t_analytic = compute_total_loss(ts, u_analytic, Ss_an, nu, T0, S0, k_modes)

    # Loss for power-law schedule u ~ 1/t^0.5
    # safe power-law schedule u ~ 1/t^0.5
    u_power = np.full_like(ts, T0)
    mask = ts > 0
    u_power[mask] = ts[mask] ** (-0.5)
    # build time increments for schedule-based S profile
    dt_array = np.diff(ts, prepend=0.0)
    # compute S trajectory for power-law schedule
    Ss_power = np.maximum(S0 - np.cumsum(u_power * dt_array), 0.0)
    # compute total loss for the power-law schedule
    L_t_power = compute_total_loss(ts, u_power, Ss_power, nu, T0, S0, k_modes)

    # plot T & total loss
    fig, ax1 = plt.subplots(figsize=(8,5))

    ##### Making T-ts variable 
    #ts = ts[-1] - ts


    ax1.plot(ts, Ts,        lw=2,  label='numerical T(t)')
    ax1.plot(ts, u_analytic,'--', lw=1.4, label='analytic $(1-t/T)^{2\\nu-1}$')
    ax1.plot(ts, T_exp,     ':',  lw=1.4, label='late exponential')
    ax1.set_xlabel('t'); ax1.set_ylabel('T(t)')
    ax2 = ax1.twinx()


    # # Use log-log scale for both axes
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # # Enforce a minimum y-value for the temperature axis
    # ax1.set_xlim(left = 9*1e+2)
    # ax2.set_xlim(left = 9*1e+2)
    # ax1.set_ylim(bottom=1e-3)


    ax2.plot(ts, L_t, 'k-', lw=1.8, label='numeric loss $L(t)$')
    ax2.plot(ts, L_t_analytic, 'r--', lw=1.6, label='analytic-profile loss')
    # plot loss for power-law schedule
    ax2.plot(ts, L_t_power, '--', lw=1.6, label='power-law 1/t^0.5 loss')
    ax2.set_ylabel('Total loss $L(t)$')
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right', frameon=False)
    plt.title(f"Profiles & Total Loss (ν={nu}, S₀={S0})")
    plt.grid(ls='--', lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------- #
    #  Plot ratio of numerical temperature to numeric loss                  #
    # --------------------------------------------------------------------- #
    fig_ratio, ax_ratio = plt.subplots(figsize=(8,5))
    ratio_num = Ts / L_t
    ax_ratio.plot(ts, ratio_num, lw=2, label='T(t)/L(t)')
    ax_ratio.set_xscale('log')
    ax_ratio.set_yscale('log')
    ax_ratio.set_xlabel('t')
    ax_ratio.set_ylabel('Ratio T/L')
    ax_ratio.legend(frameon=False)
    plt.title(f'Ratio of Numerical Temperature to Numeric Loss (ν={nu}, S₀={S0})')
    plt.grid(ls='--', lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.show()

    # # per-mode loss curves
    # plt.figure(figsize=(8,5))
    # losses = []
    # for k in range(1, k_modes+1):
    #     p_k  = k**(-nu)
    #     l0   = p_k * T0
    #     integrand = 0.5 * p_k**2 * Ts**2 * np.exp(p_k * (S0 - Ss))
    #     I_k       = np.concatenate(([0.0], np.cumsum(integrand[:-1]) * dt))
    #     losses.append(np.exp(-p_k * (S0 - Ss)) * (I_k + l0))
    # for k, l_k in enumerate(losses, start=1):
    #     plt.plot(ts, l_k, lw=1.0, label=f'ℓ_{k}(t)')
    # plt.xlabel('t'); plt.ylabel('ℓ_k(t)')
    # plt.title('Per-mode loss curves')
    # plt.legend(frameon=False, ncol=5, fontsize='small')
    # plt.grid(ls='--', lw=0.4, alpha=0.6)
    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------------------------------- #
    #  Sweep S0 to compare total loss for different initial S₀         #
    # ----------------------------------------------------------------- #
        # ----------------------------------------------------------------- #
    #  Sweep S0 to compare total loss; label curves by t_final         #
    # ----------------------------------------------------------------- #
    S0_list = np.linspace(100,1000,5)
    # plt.figure(figsize=(7,5))
    # for S0_i in S0_list:
    #     tsi, Tsi, Ssi = solve_profile(T0, S0_i, nu, dt)
    #     L_t_i = compute_total_loss(tsi, Tsi, Ssi, nu, T0, S0_i, k_modes)
    #     t_final = tsi[-1]
    #     plt.plot(tsi, L_t_i, lw=2, label=f'$t_{{final}}={t_final:.3f}$')
    # plt.xlabel('t')
    # plt.ylabel('Total loss $L(t)$')
    # plt.title(f'Total loss vs time; curves labeled by $t_{{final}}$ (ν={nu})')
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='--', lw=0.4, alpha=0.6)
    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------------------------------- #
    #  Plot final loss vs final time                                   #
    # ----------------------------------------------------------------- #
    finals_t = []
    finals_L = []
    for S0_i in tqdm.tqdm(S0_list, desc='Sweeping S0'):
        tsi, Tsi, Ssi = solve_profile(T0, S0_i, nu, dt)
        L_t_i = compute_final_loss(tsi, Tsi, Ssi, nu, T0, S0_i, k_modes)
        finals_t.append(tsi[-1])
        finals_L.append(L_t_i)
    plt.figure(figsize=(6,4))
    plt.plot(finals_t, finals_L, 'o-', lw=2, label='final loss vs time')
    plt.xlabel('Final time $t_{\\mathrm{final}}$')
    plt.ylabel('Final loss $L(t_{\\mathrm{final}})$')
    plt.title(f'Final loss vs Final time (ν={nu})')
    plt.grid(ls='--', lw=0.4, alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------- #
    #  Fit power-law: L_final ~ t_final^alpha                           #
    # ----------------------------------------------------------------- #
    logs_t = np.log(finals_t)
    logs_L = np.log(finals_L)
    slope, intercept = np.polyfit(logs_t, logs_L, 1)
    print(f"Fitted power-law dependence: L_final ∝ t_final^{slope:.3f}")

    # Log-Log plot of final loss vs final time with fitted power-law
    plt.figure(figsize=(6,4))
    plt.loglog(finals_t, finals_L, 'o', label='Data')
    # Construct fitted curve: L_fit = exp(intercept) * t^slope
    L_fit = np.exp(intercept) * np.array(finals_t)**slope
    plt.loglog(finals_t, L_fit, '-', label=f'Fit: slope={slope:.2f}')
    plt.xlabel('Final time $t_{\\mathrm{final}}$')
    plt.ylabel('Final loss $L(t_{\\mathrm{final}})$')
    plt.title(f'Log-Log final loss vs final time (ν={nu})')
    plt.legend(frameon=False)
    plt.grid(ls='--', lw=0.4, alpha=0.6, which='both')
    plt.tight_layout()
    plt.show()