import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib
matplotlib.use('TKAgg') 

"""
Simple Model Predictive Control (MPC) Example using CVXPY
- System: Simple car dynamics (position, velocity)
- Goal: Track a reference velocity using MPC
"""

# System dynamics (discrete-time state space model)
# x[k+1] = A*x[k] + B*u[k]
# where x = [position, velocity]' and u = [acceleration]

def car_dynamics(x, u, dt=0.1):
    """Simple car dynamics: position and velocity"""
    # State vector: [position, velocity]
    pos, vel = x
    acc = u  # Control input: acceleration
    
    # Update state
    new_pos = pos + vel * dt
    new_vel = vel + acc * dt
    
    return np.array([new_pos, new_vel])

def run_mpc_simulation():
    # Simulation parameters
    T = 10.0  # Total simulation time (seconds)
    dt = 0.1  # Time step (seconds)
    N = 10    # Prediction horizon (steps)
    
    # System dimensions
    nx = 2  # Number of states (position, velocity)
    nu = 1  # Number of inputs (acceleration)
    
    # System matrices (for linear state space model)
    A = np.array([[1, dt], [0, 1]])  # State transition matrix
    B = np.array([[0], [dt]])        # Control input matrix
    
    # Cost matrices
    Q = np.diag([0.1, 1.0])  # State cost (position, velocity)
    R = np.array([[0.1]])    # Control cost (acceleration)
    
    # Initial state [position, velocity]
    x0 = np.array([0.0, 0.0])
    
    # Reference trajectory (target positions and velocities over time)
    sim_steps = int(T/dt) + 1
    ref_velocity = np.zeros(sim_steps)
    
    # Reference: accelerate to 2 m/s, then to 0 m/s
    ref_velocity[10:50] = 2.0
    ref_velocity[50:] = 0.0
    
    # Calculate reference position by integrating velocity
    ref_position = np.cumsum(ref_velocity) * dt
    
    # Create reference trajectory [position, velocity]
    reference = np.column_stack((ref_position, ref_velocity))
    
    # Storage for results
    states = [x0]
    controls = []
    
    # Current state
    x = x0.copy()
    
    # MPC loop
    for k in range(sim_steps - N):
        # Reference trajectory for the prediction horizon
        ref_horizon = reference[k:k+N]
        
        # Define and solve MPC problem using CVXPY
        
        # Variables
        u = cp.Variable((N, nu))  # Control inputs over horizon
        x_var = cp.Variable((N+1, nx))  # States over horizon
        
        # Cost function
        cost = 0
        for t in range(N):
            # Stage cost: (x_t - ref_t)^T * Q * (x_t - ref_t) + u_t^T * R * u_t
            cost += cp.quad_form(x_var[t] - ref_horizon[t], Q) + cp.quad_form(u[t], R)
            
        # Constraints
        constraints = []
        
        # Initial state constraint
        constraints.append(x_var[0] == x)
        
        # Dynamics constraints
        for t in range(N):
            constraints.append(x_var[t+1] == A @ x_var[t] + B @ u[t])
        
        # Control constraints: -1 <= u <= 1 (acceleration limits)
        for t in range(N):
            constraints.append(u[t] >= -1)
            constraints.append(u[t] <= 1)
        
        # Define the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        # Solve the problem
        problem.solve(solver=cp.OSQP)
        
        if problem.status != "optimal":
            print(f"Warning: Problem not solved to optimality at step {k}, status: {problem.status}")
        
        # Extract the optimal control input for the current time step
        u_optimal = u.value[0, 0]
        controls.append(u_optimal)
        
        # Apply the control and update state
        x = car_dynamics(x, u_optimal, dt)
        states.append(x)
    
    # Convert lists to arrays
    states = np.array(states)
    controls = np.array(controls)
    
    # Plot results
    plot_results(states, controls, reference, dt)

def plot_results(states, controls, reference, dt):
    """Plot the MPC simulation results"""
    time = np.arange(len(states)) * dt
    ref_time = np.arange(len(reference)) * dt
    
    # Create figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot position
    axs[0].plot(time, states[:, 0], 'b-', label='Actual')
    axs[0].plot(ref_time, reference[:, 0], 'r--', label='Reference')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('MPC for Car Velocity Control')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot velocity
    axs[1].plot(time, states[:, 1], 'b-', label='Actual')
    axs[1].plot(ref_time, reference[:, 1], 'r--', label='Reference')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot control input (acceleration)
    control_time = np.arange(len(controls)) * dt
    axs[2].plot(control_time, controls, 'g-')
    axs[2].set_ylabel('Acceleration (m/s²)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('mpc_results_cvxpy.png')
    plt.show()

if __name__ == "__main__":
    run_mpc_simulation()
    print("MPC simulation completed. Results plotted in 'mpc_results_cvxpy.png'")
