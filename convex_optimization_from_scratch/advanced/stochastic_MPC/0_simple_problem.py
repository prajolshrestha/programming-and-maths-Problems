import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib
matplotlib.use('TKAgg')  # For compatibility

"""
Stochastic Model Predictive Control (SMPC) Example
- System: Simple car dynamics (position, velocity) with uncertainty
- Goal: Track a reference velocity using SMPC, accounting for disturbances
"""

def car_dynamics_with_disturbance(x, u, w, dt=0.1):
    """
    Simple car dynamics with disturbance
    Args:
        x: State vector [position, velocity]
        u: Control input (acceleration)
        w: Disturbance vector [position_disturbance, velocity_disturbance]
        dt: Time step
    """
    pos, vel = x
    acc = u
    
    # Update state with disturbance
    new_pos = pos + vel * dt + w[0]
    new_vel = vel + acc * dt + w[1]
    
    return np.array([new_pos, new_vel])

def run_smpc_simulation():
    # Simulation parameters
    T = 10.0          # Total simulation time (seconds)
    dt = 0.1          # Time step (seconds)
    N = 10            # Prediction horizon (steps)
    num_scenarios = 5 # Number of uncertainty scenarios
    
    # System dimensions
    nx = 2  # Number of states (position, velocity)
    nu = 1  # Number of inputs (acceleration)
    nw = 2  # Number of disturbances (position, velocity)
    
    # System matrices
    A = np.array([[1, dt], [0, 1]])  # State transition matrix
    B = np.array([[0], [dt]])        # Control input matrix
    
    # Cost matrices
    Q = np.diag([0.1, 1.0])  # State cost (position, velocity)
    R = np.array([[0.1]])    # Control cost (acceleration)
    
    # Disturbance characteristics
    # We'll simulate random disturbances from normal distributions
    pos_disturbance_std = 0.01   # Position disturbance standard deviation (m)
    vel_disturbance_std = 0.05   # Velocity disturbance standard deviation (m/s)
    
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
    actual_disturbances = []
    
    # Current state
    x = x0.copy()
    
    # MPC loop
    for k in range(sim_steps - N):
        # Reference trajectory for the prediction horizon
        ref_horizon = reference[k:k+N]
        
        # Generate disturbance scenarios for robust prediction
        # Each scenario is a potential sequence of disturbances over the horizon
        disturbance_scenarios = np.zeros((num_scenarios, N, nw))
        for s in range(num_scenarios):
            for t in range(N):
                disturbance_scenarios[s, t, 0] = np.random.normal(0, pos_disturbance_std)
                disturbance_scenarios[s, t, 1] = np.random.normal(0, vel_disturbance_std)
        
        # Define and solve SMPC problem using CVXPY
        
        # Variables
        u = cp.Variable((N, nu))          # Control inputs over horizon
        x_scenarios = []                  # States for each scenario
        
        # Create state variables for each scenario
        for s in range(num_scenarios):
            x_scenarios.append(cp.Variable((N+1, nx)))
        
        # Cost function (expectation over scenarios)
        cost = 0
        for s in range(num_scenarios):
            scenario_cost = 0
            for t in range(N):
                # Stage cost: tracking error + control effort
                scenario_cost += cp.quad_form(x_scenarios[s][t] - ref_horizon[t], Q) + cp.quad_form(u[t], R)
            cost += (1.0/num_scenarios) * scenario_cost  # Equal weight to each scenario
        
        # Constraints
        constraints = []
        
        # Initial state constraint (same for all scenarios)
        for s in range(num_scenarios):
            constraints.append(x_scenarios[s][0] == x)
        
        # Dynamics constraints with disturbances for each scenario
        for s in range(num_scenarios):
            for t in range(N):
                # Apply system dynamics with the specific disturbance sequence for this scenario
                w_st = disturbance_scenarios[s, t]
                constraints.append(x_scenarios[s][t+1] == A @ x_scenarios[s][t] + B @ u[t] + w_st)
        
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
        
        # Generate actual disturbance for this time step
        w_actual = np.array([
            np.random.normal(0, pos_disturbance_std),
            np.random.normal(0, vel_disturbance_std)
        ])
        actual_disturbances.append(w_actual)
        
        # Apply the control and update state with actual disturbance
        x = car_dynamics_with_disturbance(x, u_optimal, w_actual, dt)
        states.append(x)
    
    # Convert lists to arrays
    states = np.array(states)
    controls = np.array(controls)
    actual_disturbances = np.array(actual_disturbances)
    
    # Plot results
    plot_results(states, controls, actual_disturbances, reference, dt)

def plot_results(states, controls, disturbances, reference, dt):
    """Plot the SMPC simulation results"""
    time = np.arange(len(states)) * dt
    ref_time = np.arange(len(reference)) * dt
    dist_time = np.arange(len(disturbances)) * dt
    
    # Create figure and axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    # Plot position
    axs[0].plot(time, states[:, 0], 'b-', label='Actual')
    axs[0].plot(ref_time, reference[:, 0], 'r--', label='Reference')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Stochastic MPC for Car Velocity Control')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot velocity
    axs[1].plot(time, states[:, 1], 'b-', label='Actual')
    axs[1].plot(ref_time, reference[:, 1], 'r--', label='Reference')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot control input (acceleration)
    axs[2].plot(dist_time, controls, 'g-')
    axs[2].set_ylabel('Acceleration (m/s²)')
    axs[2].grid(True)
    
    # Plot disturbances
    axs[3].plot(dist_time, disturbances[:, 0], 'm-')
    axs[3].set_ylabel('Position\nDisturbance (m)')
    axs[3].grid(True)
    
    axs[4].plot(dist_time, disturbances[:, 1], 'c-')
    axs[4].set_ylabel('Velocity\nDisturbance (m/s)')
    axs[4].set_xlabel('Time (s)')
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.savefig('smpc_results.png')
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    run_smpc_simulation()
    print("Stochastic MPC simulation completed. Results plotted in 'smpc_results.png'")
