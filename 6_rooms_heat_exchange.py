import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim
from scipy.linalg import solve_continuous_lyapunov as solve_lyap
from scipy.linalg import svd

# Generate the state-space matrices for a thermal system with 6 rooms
def generate_lti_thermal_system():
    n = 6 
    # Tridiagonal matrix A modeling heat exchange between adjacent rooms
    A = -2 * np.eye(n) + np.diag(np.ones(n - 1), k=1) + np.diag(np.ones(n - 1), k=-1)
    B = np.zeros((n, 2))
    B[0, 0] = 1  # heater in room 1
    B[-1, 1] = 1  # heater in room 6
    C = np.eye(n) 
    D = np.zeros((n, 2)) # no direct feedthrough
    return A, B, C, D

# Perform balanced truncation model order reduction
def balanced_truncation(A, B, C, order_r):
    Wc = solve_lyap(A, -B @ B.T)
    Wo = solve_lyap(A.T, -C.T @ C)
    U, s, Vh = svd(Wc @ Wo) # svd decomposition
    # Compute transformation matrix for balanced truncation
    T = U[:, :order_r] @ np.diag(np.sqrt(s[:order_r]))
    T_inv = np.linalg.pinv(T)
    # Reduced system matrices
    A_r = T_inv @ A @ T
    B_r = T_inv @ B
    C_r = C @ T
    return A_r, B_r, C_r

# Simulate a linear state-space system with input u over time t
def simulate_system(A, B, C, D, u, t):
    sys = StateSpace(A, B, C, D)
    _, y, _ = lsim(sys, U=u, T=t)
    return y

# Compute L2 error between full and reduced model outputs
def compute_error(y_full, y_reduced):
    return np.linalg.norm(y_full - y_reduced, axis=0)

def run_balanced_truncation_demo():
    A, B, C, D = generate_lti_thermal_system()
    t = np.linspace(0, 10, 1000)
    u = np.zeros((len(t), 2))
    u[:, 0] = 1  # step input to heater 1

    y_full = simulate_system(A, B, C, D, u, t)

    # Plot output of full and reduced systems for Room 1
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_full[:, 0], label='Full system (Room 1)', linewidth=2, color='black')

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, r in enumerate([4, 3, 2]): # try multiple reduced orders
        A_r, B_r, C_r = balanced_truncation(A, B, C, r)
        y_r = simulate_system(A_r, B_r, C_r, D, u, t)
        error = compute_error(y_full, y_r)
        print(f"Reduction to order {r}: L2 error (Room 1) = {error[0]:.4f}")
        plt.plot(t, y_r[:, 0], '--', label=f'Reduced (order {r})', linewidth=2, color=colors[i])

    plt.title("Balanced Truncation on LTI Thermal System", fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Room 1 Temperature", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("balanced_truncation_lti.png", dpi=300)
    #plt.show() # commented out to avoid blocking execution
    plt.close() # close the figure to allow the script to continue (figure are automatically saved)

# Introduce time-variability
def make_system_time_varying(A, t, freq=0.5):
    A_tv = A.copy()
    variation = 0.5 * np.sin(2 * np.pi * freq * t)
    A_tv[0, 1] += variation
    return A_tv

def run_time_varying_demo():
    A, B, C, D = generate_lti_thermal_system()
    t = np.linspace(0, 10, 1000)
    u = np.zeros((len(t), 2))
    u[:, 0] = 1  

    x = np.zeros((6,))
    y_tv = []

    for i in range(len(t)):
        dt = t[1] - t[0]
        A_tv = make_system_time_varying(A, t[i])
        dx = A_tv @ x + B @ u[i]
        x += dx * dt
        y_tv.append(C @ x)

    y_tv = np.array(y_tv)

    A_r, B_r, C_r = balanced_truncation(A, B, C, 3)
    y_bt = simulate_system(A_r, B_r, C_r, D, u, t)
    error = compute_error(y_tv, y_bt)
    print(f"Time-varying case: L2 error (Room 1) = {error[0]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, y_tv[:, 0], label='Time-Varying System (Room 1)', linewidth=2, color='black')
    plt.plot(t, y_bt[:, 0], '--', label='Reduced LTI Model (Room 1)', linewidth=2, color='tab:red')

    plt.title("Failure of Balanced Truncation on Time-Varying System", fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Room 1 Temperature", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("balanced_truncation_tv.png", dpi=300)
    #plt.show()
    plt.close()

if __name__ == "__main__":
    # Run both demos
    run_balanced_truncation_demo()
    run_time_varying_demo()
