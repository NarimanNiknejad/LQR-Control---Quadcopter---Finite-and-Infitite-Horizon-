# Quadcopter Finite and Infinite Horizon LQR - 3D Control 

This repository contains a Python script `quadrotor.py` that implements a decentralized LQR controller for a quadrotor. The quadrotor's nonlinear dynamics are linearized into four subsystems and a decentralized LQR controller is designed for each subsystem. The performance of the controller is evaluated using numerical simulations.

## Prerequisites

Before running the code, make sure to install the following packages:
* `numpy`
* `scipy`
* `matplotlib`

## Usage

To run the script, simply run the command `python quadrotor.py` in the terminal. This will generate a 3D plot of the trajectory of the quadrotor.

## Description

The script starts by importing the necessary packages:

```python
from time import time
import nonlinearDynamics
import argparse
import numpy as np
import scipy
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, inv, sqrtm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nonlinearDynamics import g, m, Ix, Iy, Iz
```

It then defines the time instants for the simulation, the initial state, and the time horizon:

```python
# time instants for simulation
t_max = 10
time_step = 0.01
t = np.arange(0., t_max, time_step)

# Inintial state
X0 = np.zeros(12)
tf = 2
t_span = [0, tf]
```

The script then defines a function to solve the Riccati differential equation for a continuous finite time horizon using the terminal value approach:

```python
def solve_riccati_continuous(A, B, Q, R, T, P_T):
    """
    Solves the Riccati differential equation for a continuous finite time horizon using the terminal value approach.

    Args:
    A: ndarray, the system dynamics matrix
    B: ndarray, the input matrix
    Q: ndarray, the state cost matrix
    R: ndarray, the input cost matrix
    T: float, the final time
    P_T: ndarray, the terminal value of the solution

    Returns:
    P: ndarray, the solution to the Riccati differential equation
    """
    def riccati_dynamics(t, P):
        """
        The dynamics of the Riccati differential equation.

        Args:
        t: float, the time variable
        P: ndarray, the solution to the Riccati differential equation at time t

        Returns:
        Pdot: ndarray, the derivative of P with respect to time
        """
        P = P.reshape((len(A), len(A)))
        Pdot = -A.T @ P - P @ A - Q + P @ B @ np.linalg.inv(R) @ B.T @ P
        Pdot = Pdot.reshape(len(A)**2)
        return Pdot
    
    t_span = (0, T)
    P_T = P_T.reshape(len(A)**2)
    sol = solve_ivp(riccati_dynamics, t_span, P_T)
    P = sol.y[:,-1].reshape((len(A), len(A)))
    return P
```

Next, the script defines a function to solve the continuous time LQR controller:

```python
def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with
