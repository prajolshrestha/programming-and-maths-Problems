import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib
matplotlib.use('TKAgg')  # For compatibility

"""
Comparison between Deterministic MPC and Stochastic MPC
- Shows how SMPC handles uncertainty better than standard MPC
- Compares performance on a simple car velocity tracking problem
"""

def car_dynamics(x, u, dt=0.1):
    """
    Simple car dynamics without disturbance (deterministic model)
    Args:
        x: State vector [position, velocity]
        u: Control input (acceleration)
        dt: Time step
    """
    pos, vel = x
    acc = u
    
    # Update state (perfect model without disturbances)
    new_pos = pos + vel * dt
    new_vel = vel + acc * dt
    
    return np.array([new_pos, new_vel])

def car_dynamics_with_disturbance(x, u, w, dt=0.1):
    """
    Simple car dynamics with disturbance (actual system behavior)
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

def compare_mpc_smpc():
    """Run both deterministic and stochastic MPC to compare results"""
    # Simulation parameters
    T = 10.0          # Total simulation time (seconds)
    dt = 0.1          # Time step (seconds)
    N = 10            # Prediction horizon (steps)
    num_scenarios = 5 # Number of uncertainty scenarios for SMPC
    
    # System dimensions
    nx = 2  # Number of states (position, velocity)
    nu = 1  # Number of inputs (acceleration)
    nw = 2  # Number of disturbances (position, velocity)
    
    # System matrices (for linear state space model)
    A = np.array([[1, dt], [0, 1]])  # State transition matrix
    B = np.array([[0], [dt]])        # Control input matrix
    
    # Cost matrices
    Q = np.diag([0.1, 1.0])  # State cost (position, velocity)
    R = np.array([[0.1]])    # Control cost (acceleration)
    
    # Disturbance characteristics
    pos_disturbance_std = 0.01   # Position disturbance standard deviation (m)
    vel_disturbance_std = 0.05   # Velocity disturbance standard deviation (m/s)
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
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
    
    # Pre-generate actual disturbances for fair comparison
    actual_disturbances = []
    for k in range(sim_steps - N):
        w_actual = np.array([
            np.random.normal(0, pos_disturbance_std),
            np.random.normal(0, vel_disturbance_std)
        ])
        actual_disturbances.append(w_actual)
    
    # Run deterministic MPC simulation
    mpc_states, mpc_controls = run_deterministic_mpc(x0, reference, A, B, Q, R, N, dt)
    
    # Reset the random seed for consistent scenario generation
    np.random.seed(42)
    
    # Run stochastic MPC simulation
    smpc_states, smpc_controls = run_stochastic_mpc(x0, reference, A, B, Q, R, N, num_scenarios, 
                                                 pos_disturbance_std, vel_disturbance_std, 
                                                 actual_disturbances, dt)
    
    # Calculate performance metrics
    mpc_pos_error = np.mean(np.abs(mpc_states[:, 0] - reference[:len(mpc_states), 0]))
    mpc_vel_error = np.mean(np.abs(mpc_states[:, 1] - reference[:len(mpc_states), 1]))
    mpc_control_effort = np.mean(np.abs(mpc_controls))
    
    smpc_pos_error = np.mean(np.abs(smpc_states[:, 0] - reference[:len(smpc_states), 0]))
    smpc_vel_error = np.mean(np.abs(smpc_states[:, 1] - reference[:len(smpc_states), 1]))
    smpc_control_effort = np.mean(np.abs(smpc_controls))
    
    # Plot comparison results
    plot_comparison(mpc_states, mpc_controls, smpc_states, smpc_controls, 
                   actual_disturbances, reference, dt)
    
    # Print performance metrics
    print("\nPerformance Comparison:")
    print("-----------------------")
    print(f"{'Metric':<20} {'Deterministic MPC':<20} {'Stochastic MPC':<20}")
    print(f"{'Position Error (m)':<20} {mpc_pos_error:<20.4f} {smpc_pos_error:<20.4f}")
    print(f"{'Velocity Error (m/s)':<20} {mpc_vel_error:<20.4f} {smpc_vel_error:<20.4f}")
    print(f"{'Control Effort (m/s²)':<20} {mpc_control_effort:<20.4f} {smpc_control_effort:<20.4f}")
    
    # Calculate and print improvement percentages
    pos_error_improve = ((mpc_pos_error - smpc_pos_error) / mpc_pos_error) * 100
    vel_error_improve = ((mpc_vel_error - smpc_vel_error) / mpc_vel_error) * 100
    control_effort_diff = ((smpc_control_effort - mpc_control_effort) / mpc_control_effort) * 100
    
    print("\nImprovement with SMPC:")
    print(f"Position error reduced by: {pos_error_improve:.2f}%")
    print(f"Velocity error reduced by: {vel_error_improve:.2f}%") 
    print(f"Control effort changed by: {control_effort_diff:.2f}%")

def run_deterministic_mpc(x0, reference, A, B, Q, R, N, dt):
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

    return states, controls
    
def run_stochastic_mpc(x0, reference, A, B, Q, R, N, num_scenarios, 
                     pos_disturbance_std, vel_disturbance_std, actual_disturbances, dt):
    """Run stochastic MPC simulation with predefined disturbances"""
    sim_steps = len(reference) - N
    
    # Storage for results
    states = [x0]
    controls = []
    
    # Current state
    x = x0.copy()
    
    # MPC loop
    for k in range(sim_steps):
        # Reference trajectory for the prediction horizon
        ref_horizon = reference[k:k+N]
        
        # Generate disturbance scenarios for robust prediction
        disturbance_scenarios = np.zeros((num_scenarios, N, 2))
        for s in range(num_scenarios):
            for t in range(N):
                disturbance_scenarios[s, t, 0] = np.random.normal(0, pos_disturbance_std)
                disturbance_scenarios[s, t, 1] = np.random.normal(0, vel_disturbance_std)
        
        # Define and solve SMPC problem using CVXPY
        
        # Variables
        u = cp.Variable((N, 1))  # Control inputs over horizon
        x_scenarios = []         # States for each scenario
        
        # Create state variables for each scenario
        for s in range(num_scenarios):
            x_scenarios.append(cp.Variable((N+1, 2)))
        
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
                # Key difference: SMPC explicitly models disturbances in different scenarios
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
            print(f"Warning: Stochastic MPC not solved to optimality at step {k}, status: {problem.status}")
        
        # Extract the optimal control input for the current time step
        u_optimal = u.value[0, 0]
        controls.append(u_optimal)
        
        # Apply the control and update state with the SAME actual disturbance
        # as used in deterministic MPC (for fair comparison)
        w_actual = actual_disturbances[k]
        x = car_dynamics_with_disturbance(x, u_optimal, w_actual, dt)
        states.append(x)
    
    # Convert lists to arrays
    states = np.array(states)
    controls = np.array(controls)
    
    return states, controls

def plot_comparison(mpc_states, mpc_controls, smpc_states, smpc_controls, disturbances, reference, dt):
    """Plot the comparison between deterministic and stochastic MPC"""
    time = np.arange(len(mpc_states)) * dt
    ref_time = np.arange(len(reference)) * dt
    dist_time = np.arange(len(disturbances)) * dt
    
    # Create figure and axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    
    # Plot position
    axs[0].plot(time, mpc_states[:, 0], 'b-', label='Deterministic MPC')
    axs[0].plot(time, smpc_states[:, 0], 'g-', label='Stochastic MPC')
    axs[0].plot(ref_time, reference[:, 0], 'r--', label='Reference')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Comparison: Deterministic vs Stochastic MPC')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot velocity
    axs[1].plot(time, mpc_states[:, 1], 'b-', label='Deterministic MPC')
    axs[1].plot(time, smpc_states[:, 1], 'g-', label='Stochastic MPC')
    axs[1].plot(ref_time, reference[:, 1], 'r--', label='Reference')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot position error
    pos_error_mpc = np.abs(mpc_states[:, 0] - reference[:len(mpc_states), 0])
    pos_error_smpc = np.abs(smpc_states[:, 0] - reference[:len(smpc_states), 0])
    axs[2].plot(time, pos_error_mpc, 'b-', label='Deterministic MPC')
    axs[2].plot(time, pos_error_smpc, 'g-', label='Stochastic MPC')
    axs[2].set_ylabel('Position Error (m)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot velocity error
    vel_error_mpc = np.abs(mpc_states[:, 1] - reference[:len(mpc_states), 1])
    vel_error_smpc = np.abs(smpc_states[:, 1] - reference[:len(smpc_states), 1])
    axs[3].plot(time, vel_error_mpc, 'b-', label='Deterministic MPC')
    axs[3].plot(time, vel_error_smpc, 'g-', label='Stochastic MPC')
    axs[3].set_ylabel('Velocity Error (m/s)')
    axs[3].legend()
    axs[3].grid(True)
    
    # Plot control input (acceleration)
    axs[4].plot(dist_time, mpc_controls, 'b-', label='Deterministic MPC')
    axs[4].plot(dist_time, smpc_controls, 'g-', label='Stochastic MPC')
    axs[4].set_ylabel('Acceleration (m/s²)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.savefig('mpc_vs_smpc_comparison.png')
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    compare_mpc_smpc()
    print("Comparison completed. Results plotted in 'mpc_vs_smpc_comparison.png'")
