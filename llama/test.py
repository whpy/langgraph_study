import numpy as np

def solve_heat_equation(nx, dx, dt, nt, T_left, T_right, k=0.01):
    r = k * dt / dx**2
    alpha = 1 - r
    beta = 1 - alpha
    T = np.ones(nx) * T_left
    T_new = T.copy()
    
    for _ in range(nt):
        T_new[1:-1] = T[1:-1] + r * (T[:-2] - 2 * T[1:-1] + T[2:]) / dx**2 + beta * (T_right - T[1:-1]) * dt
        T[:] = T_new
        T[0], T[-1] = T_left, T_right  # boundary conditions
    
    return T

nx = 100
dx = 0.01
dt = 0.01
nt = 1000
T_left, T_right = 0.1, 0.5

temperature_profile = solve_heat_equation(nx, dx, dt, nt, T_left, T_right)
print("Temperature at each point along the domain: ", temperature_profile)