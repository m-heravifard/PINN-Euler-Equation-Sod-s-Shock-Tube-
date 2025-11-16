"""
Supervisor: Prof. K. Hejranfar
Author: Mohammad E. Heravifard
Project #01: Data-Free Physics-Driven Neural Networks Approach
for 1D Euler Equations for Compressible Flows:
Testcase #01: Sod's Shock Tube Problem.
"""
# -----------------------------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler

# ------------------------------------------------------------------------------------------------------------------------------------#

# SET RANDOM SEEDS FOR REPRODUCIBILITY
torch.manual_seed(123456)
np.random.seed(123456)

# -------------------------------------------------------------------------------------------------------------------------------------#

# DEFINING THE NEURAL NETWORK MODEL
class DNN(nn.Module):
    """
    Deep Neural Network (DNN) for approximating the solution to the Euler equations.
    The network maps a 2D input (time, space) to three outputs: density, pressure, and velocity.
    """
    def __init__(self, input_dim=2, hidden_dim=30, output_dim=3, num_hidden_layers=5):
        """
        Initialize the DNN.

        Args:
            input_dim (int): Dimension of input data (default is 2: time and space).
            hidden_dim (int): Number of neurons in the hidden layers.
            output_dim (int): Dimension of output data (density, pressure, velocity).
            num_hidden_layers (int): Number of hidden layers.
        """
        super(DNN, self).__init__()

        # Building the network using nn.Sequential for simplicity
        layers = []
        # First layer: from input dimension to hidden dimension
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers: all with the same dimensions and activation function
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Final output layer: maps hidden dimension to output dimension
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)

    def loss_pde(self, x):
        """
        Compute the loss associated with the PDE residual.

        The PDE is defined by the 1D Euler equations for compressible flows.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2) where the columns are [time, space].

        Returns:
            torch.Tensor: Scalar tensor representing the PDE residual loss.
        """
        # Get the network prediction
        y = self.net(x)
        # Split the output into density (rho), pressure (p), and velocity (u)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        gamma = 1.4  # Specific heat ratio for air

        # Calculating gradients of the network outputs with respect to input coordinates
        drho_dt_dx = compute_gradients(rho, x)  # [rho_t, rho_x]
        rho_t, rho_x = drho_dt_dx[:, :1], drho_dt_dx[:, 1:]

        du_dt_dx = compute_gradients(u, x)      # [u_t, u_x]
        u_t, u_x = du_dt_dx[:, :1], du_dt_dx[:, 1:]

        dp_dt_dx = compute_gradients(p, x)      # [p_t, p_x]
        p_t, p_x = dp_dt_dx[:, :1], dp_dt_dx[:, 1:]

        # Computing the residuals for each Euler equation term
        mass_residual = rho_t + u * rho_x + rho * u_x
        momentum_residual = rho * (u_t + u * u_x) + p_x
        energy_residual = p_t + gamma * p * u_x + u * p_x

        # Mean-squared error for each residual and sum them up with equal weight
        loss = (mass_residual**2).mean() + (momentum_residual**2).mean() + (energy_residual**2).mean()
        return loss

    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        """
        Compute the loss associated with the initial conditions.

        Args:
            x_ic (torch.Tensor): Tensor containing the initial condition coordinates.
            rho_ic (torch.Tensor): Tensor containing the density initial condition.
            u_ic (torch.Tensor): Tensor containing the velocity initial condition.
            p_ic (torch.Tensor): Tensor containing the pressure initial condition.

        Returns:
            torch.Tensor: Scalar tensor representing the initial condition loss.
        """
        # Evaluating the network at the initial condition points
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

        # Computing the mean squared error between predicted and true initial conditions
        loss_ic = ((rho_ic_nn - rho_ic) ** 2).mean() + \
                  ((p_ic_nn - p_ic) ** 2).mean() + \
                  ((u_ic_nn - u_ic) ** 2).mean()
        return loss_ic

# -----------------------------------------------------------------------------------------------------------------------------------#

# UTILITY FUNCTIONS
def compute_gradients(outputs, inputs):
    """
    Compute gradients of outputs with respect to inputs using autograd.

    Args:
        outputs (torch.Tensor): The outputs of the network.
        inputs (torch.Tensor): The inputs to the network.

    Returns:
        torch.Tensor: Gradients of outputs with respect to inputs.
    """
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]

def to_numpy(tensor):
    """
    Convert a torch.Tensor to a NumPy array.

    Args:
        tensor (torch.Tensor or np.ndarray): Input tensor or array.

    Returns:
        np.ndarray: The input converted to a NumPy array.

    Raises:
        TypeError: If the input is neither a torch.Tensor nor a np.ndarray.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f'Expected torch.Tensor or np.ndarray, got {type(tensor)}')

def initial_conditions(x):
    """
    Define the initial conditions for density, velocity, and pressure.

    For the Sod shock tube problem:
        - For x <= 0.5: rho = 1.0, p = 1.0
        - For x > 0.5: rho = 0.125, p = 0.1
        - Velocity is zero everywhere.

    Args:
        x (np.ndarray): 1D array of spatial points.

    Returns:
        tuple: (rho_init, u_init, p_init) as numpy arrays.
    """
    N = x.shape[0]
    rho_init = np.zeros(N)
    u_init = np.zeros(N)
    p_init = np.zeros(N)

    # Loop through each spatial point to assign initial conditions
    for i in range(N):
        if x[i] <= 0.5:
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1

    return rho_init, u_init, p_init

# -----------------------------------------------------------------------------------------------------------------------------------#

# TRAINING AND EVALUATION FUNCTIONS
def train_model(model, optimizer, scheduler, x_int_train, x_ic_train, rho_ic, u_ic, p_ic, epochs, device):
    """
    Train the PINN model using both PDE residuals and initial condition errors.

    Args:
        model (DNN): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        x_int_train (torch.Tensor): Training points in the interior domain.
        x_ic_train (torch.Tensor): Training points on the initial condition.
        rho_ic (torch.Tensor): Density values at the initial condition.
        u_ic (torch.Tensor): Velocity values at the initial condition.
        p_ic (torch.Tensor): Pressure values at the initial condition.
        epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on.
    """
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode

        # Zero out the gradients
        optimizer.zero_grad()

        # Computing the PDE loss on interior points
        loss_pde = model.loss_pde(x_int_train)

        # Computing the initial condition loss on the initial boundary points
        loss_ic = model.loss_ic(x_ic_train, rho_ic, u_ic, p_ic)

        # Total loss: a weighted combination of PDE loss and IC loss
        total_loss = 0.1 * loss_pde + 10 * loss_ic

        # Print losses for monitoring
        print(f"Epoch {epoch:04d}: loss_pde = {loss_pde.item():.8f}, loss_ic = {loss_ic.item():.8f}, total_loss = {total_loss.item():.8f}")

        # Backpropagate the error
        total_loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate if scheduler is used

def evaluate_model(model, x_test, num_x, num_t):
    """
    Evaluate the trained model on the entire spatio-temporal domain.

    Args:
        model (DNN): The trained neural network.
        x_test (torch.Tensor): Test points covering the full domain.
        num_x (int): Number of spatial discretization points.
        num_t (int): Number of temporal discretization points.

    Returns:
        tuple: (rho_pred, p_pred, u_pred) reshaped to grids.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(x_test)

    # Converting predictions to numpy arrays for plotting
    pred_np = to_numpy(predictions)

    # Reshaping the outputs to form grids for density, pressure, and velocity
    rho_pred = pred_np[:, 0].reshape(num_x, num_t)
    p_pred = pred_np[:, 1].reshape(num_x, num_t)
    u_pred = pred_np[:, 2].reshape(num_x, num_t)

    return rho_pred, p_pred, u_pred

def plot_results(t, x, rho_pred, p_pred, u_pred, t_eval=0.2):
    """
    Plot the density, pressure, and velocity evolution and their profiles at a fixed time.

    Args:
        t (np.ndarray): 1D array of time discretization.
        x (np.ndarray): 1D array of spatial discretization.
        rho_pred (np.ndarray): Predicted density grid.
        p_pred (np.ndarray): Predicted pressure grid.
        u_pred (np.ndarray): Predicted velocity grid.
        t_eval (float): The time at which 1D plots are generated.
    """
    # -------------------------------------------------------------------------------------------------------------------------------#

    # Plot 2D Contour Plots for Density, Velocity, and Pressure
    plt.figure(figsize=(10, 6))
    contour1 = plt.contourf(t, x, rho_pred, levels=100, cmap='viridis')
    plt.colorbar(contour1, label=r'Density $\rho$')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title('Density Evolution')
    plt.show()

    plt.figure(figsize=(10, 6))
    contour2 = plt.contourf(t, x, u_pred, levels=100, cmap='viridis')
    plt.colorbar(contour2, label='Velocity (u)')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title('Velocity Evolution')
    plt.show()

    plt.figure(figsize=(10, 6))
    contour3 = plt.contourf(t, x, p_pred, levels=100, cmap='viridis')
    plt.colorbar(contour3, label='Pressure (p)')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title('Pressure Evolution')
    plt.show()

    # ---------------------------------------------------------------------------------------------------------------------------------#

    # Extract and Plot 1D Profiles at a Specific Time (t = t_eval)

    t_idx = np.argmin(np.abs(t - t_eval))  # Find the index closest to t_eval
    rho_at_t = rho_pred[:, t_idx]
    p_at_t = p_pred[:, t_idx]
    u_at_t = u_pred[:, t_idx]

    # Density Profile
    plt.figure(figsize=(10, 6))
    plt.plot(x, rho_at_t, label=r'Density $\rho$', color='blue')
    plt.xlabel('Position (x)')
    plt.ylabel(r'Density $\rho$')
    plt.title(f'Density Profile at t = {t_eval}')
    plt.legend()
    plt.grid()
    plt.show()

    # Velocity Profile
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_at_t, label=r'Velocity $u$', color='green')
    plt.xlabel('Position (x)')
    plt.ylabel(r'Velocity $u$')
    plt.title(f'Velocity Profile at t = {t_eval}')
    plt.legend()
    plt.grid()
    plt.show()

    # Pressure Profile
    plt.figure(figsize=(10, 6))
    plt.plot(x, p_at_t, label=r'Pressure $p$', color='red')
    plt.xlabel('Position (x)')
    plt.ylabel(r'Pressure $p$')
    plt.title(f'Pressure Profile at t = {t_eval}')
    plt.legend()
    plt.grid()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#

# MAIN FUNCTION
def main():
    """
    Main function to set up the problem, train the PINN, and evaluate the results.
    """
    # Device configuration
    device = torch.device('cpu')  # Change to 'cuda' if GPU is available

    # Hyperparameters and discretization settings
    learning_rate = 0.0005        # Learning rate for the optimizer
    epochs = 10000                 # Number of training epochs
    num_x = 1000                  # Number of spatial discretization points
    num_t = 1000                  # Number of temporal discretization points
    num_ic_samples = 1000         # Number of samples from the initial condition
    num_f_samples = 11000         # Number of interior collocation points

    # Create spatio-temporal grid

    x = np.linspace(0, 1, num_x)                # Spatial domain [0,1]
    t = np.linspace(0, 0.2, num_t)              # Temporal domain [0,0.2]
    t_grid, x_grid = np.meshgrid(t, x)          # Create a grid of (t,x) points

    # Flatten the grid to obtain a list of points for training and evaluation
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    # Sample training points for initial conditions and interior points
    # For initial condition: randomly select points from the spatial grid (t=0)
    ic_indices = np.random.choice(num_x, num_ic_samples, replace=False)
    x_ic = x_grid[ic_indices, 0][:, None]
    t_ic = t_grid[ic_indices, 0][:, None]
    x_ic_train = np.hstack((t_ic, x_ic))
    # Evaluate the initial conditions at these points
    rho_ic_train_np, u_ic_train_np, p_ic_train_np = initial_conditions(x_ic.flatten())

    # For the interior points: randomly sample points from the full grid
    total_points = num_x * num_t
    f_indices = np.random.choice(total_points, num_f_samples, replace=False)
    x_int = X[f_indices, None].reshape(-1, 1)  # spatial component
    t_int = T[f_indices, None].reshape(-1, 1)  # temporal component
    x_int_train = np.hstack((t_int, x_int))

    # Prepare the full domain test points for final evaluation
    x_test = np.hstack((T, X))

    # ---------------------------------------------------------------------------------------------------------------------------------#

    # Convert data to torch tensors
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32, device=device)
    x_int_train = torch.tensor(x_int_train, dtype=torch.float32, requires_grad=True, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)

    # Convert initial condition values to tensors
    rho_ic_train = torch.tensor(rho_ic_train_np, dtype=torch.float32, device=device)
    u_ic_train = torch.tensor(u_ic_train_np, dtype=torch.float32, device=device)
    p_ic_train = torch.tensor(p_ic_train_np, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------------------------------------------------------------------#

    # Initializing the model, optimizer, and learning rate scheduler
    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # A scheduler to decrease the learning rate every 1000 epochs
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    # -------------------------------------------------------------------------------------------------------------------------------#

    # Training the model
    print("Starting training...")
    start_time = time.time()
    train_model(model, optimizer, scheduler, x_int_train, x_ic_train,
                rho_ic_train, u_ic_train, p_ic_train, epochs, device)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # --------------------------------------------------------------------------------------------------------------------------------#

    # Evaluating the model on the entire domain

    print("Evaluating model on the full domain...")
    rho_pred, p_pred, u_pred = evaluate_model(model, x_test, num_x, num_t)

    # ---------------------------
    # Plot the results
    # ---------------------------
    plot_results(t, x, rho_pred, p_pred, u_pred, t_eval=0.2)

# --------------------------------------------------------------------------------------------------------------------------------------#

# RUN THE MAIN FUNCTION
if __name__ == '__main__':
    main()
