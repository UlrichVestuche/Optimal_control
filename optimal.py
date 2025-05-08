import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# --------------------------------------------------------------------- #
#  RHS with adaptive spectrum                                          #
# --------------------------------------------------------------------- #
def rhs_T(S: float, T: float, nu: float) -> float:
    """
    dT/dt with adaptive cut‑off: keep modes with p_k S ≤ 25.
    """
    if S < 1e-12:
        # plateau: all modes active → constant ratio Σp³/Σp²
        k = np.arange(1, 200_001, dtype=float)
        p = k ** (-nu)
        return -0.5 * T * T * (p**3).sum() / (p**2).sum()

    k_max = max(50, int((25.0 / S)**(1/nu)) + 1)
    k     = np.arange(1, k_max + 1, dtype=float)
    p     = k ** (-nu)
    e     = np.exp(-p * S)
    G2    = np.sum(p**2 * e)
    G3    = np.sum(p**3 * e)
    return -0.5 * T * T * (G3 / G2)

def rk4_step(S: float, T: float, dt: float, nu: float):
    f = rhs_T
    k1S, k1T = -T, f(S, T, nu)
    k2S = -(T + 0.5*dt*k1T); k2T = f(S + 0.5*dt*k1S, T + 0.5*dt*k1T, nu)
    k3S = -(T + 0.5*dt*k2T); k3T = f(S + 0.5*dt*k2S, T + 0.5*dt*k2T, nu)
    k4S = -(T +     dt*k3T); k4T = f(S +     dt*k3S, T +     dt*k3T, nu)
    S += dt * (k1S + 2*k2S + 2*k3S + k4S) / 6
    T += dt * (k1T + 2*k2T + 2*k3T + k4T) / 6
    return S, T

def solve_profile(T0: float, S0: float, nu: float,
                  dt: float = 5e-4, tol_S: float = 1e-10):
    """
    Integrate until S <= tol_S. Return arrays (t, T(t), S(t)).
    """
    t, S, T = 0.0, S0, T0
    ts, Ts, Ss = [t], [T], [S]
    while S > tol_S:
        S, T = rk4_step(S, T, dt, nu)
        t   += dt
        ts.append(t); Ts.append(T); Ss.append(S)
    return np.asarray(ts), np.asarray(Ts), np.asarray(Ss)

# --------------------------------------------------------------------- #
#  Parameters                                                          #
# --------------------------------------------------------------------- #
nu  = 2.5     # spectral exponent
T0  = 1.0     # initial T(0)
S0  = 50.0    # initial ∫₀^{t_max} T
dt  = 5e-4    # time‑step for RK4

# --------------------------------------------------------------------- #
#  Solve                                                                #
# --------------------------------------------------------------------- #
ts, Ts, Ss = solve_profile(T0, S0, nu, dt)
t_max      = ts[-1]
T_end      = Ts[-1]

# --------------------------------------------------------------------- #
#  Analytic ansatz  u(t) = (1 - t/T)^{2ν-1}                            #
# --------------------------------------------------------------------- #
u_analytic = (1 - ts / t_max)**(2*nu - 1)

# --------------------------------------------------------------------- #
#  Late exponential tail                                               #
# --------------------------------------------------------------------- #
k_big   = np.arange(1, 200_001, dtype=float)
p_big   = k_big**(-nu)
ratio_c = np.sum(p_big**3) / np.sum(p_big**2)
T_exp   = T_end * np.exp(0.5 * ratio_c * T_end * (t_max - ts))

# --------------------------------------------------------------------- #
#  Loss functions ℓ_k(t) and total loss L(t)                           #
# --------------------------------------------------------------------- #
k_vals        = np.arange(1, 51)  # sum over first 50 modes
integrand_base = 0.5 * Ts**2 * np.exp(S0 - Ss)
losses        = {}
for k in k_vals:
    p_k = k**(-nu)
    I_k = np.cumsum(p_k * integrand_base) * dt
    l_k = np.exp(-p_k * (S0 - Ss)) * I_k
    losses[k] = l_k

# total loss
L_t = np.zeros_like(ts)
for l_k in losses.values():
    L_t += l_k

# --------------------------------------------------------------------- #
#  Plot everything                                                      #
# --------------------------------------------------------------------- #
plt.figure(figsize=(8, 5))
# T-profiles
plt.plot(ts, Ts,         lw=2, label='numerical T(t)')
plt.plot(ts, u_analytic, '--', lw=1.4, label='analytic $(1-t/T)^{2\\nu-1}$')
plt.plot(ts, T_exp,       ':', lw=1.4, label='late exponential')

# total loss (secondary axis)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(ts, L_t,  '-', lw=1.8, c='black', label='total loss $L(t)$')
ax2.set_ylabel('Total loss $L(t)$', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

plt.xlabel('t')
ax1.set_ylabel('T(t)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
plt.title(f"Profiles & total loss (ν={nu}, S₀={S0})")
plt.grid(ls="--", lw=0.4, alpha=0.6)
plt.tight_layout()
plt.show()