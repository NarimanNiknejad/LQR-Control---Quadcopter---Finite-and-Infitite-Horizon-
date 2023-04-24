# Finite/Infinite Time Horizon LQR Control for Quadrotor

This repository contains a Python implementation of a finite time horizon Linear Quadratic Regulator (LQR) control for a quadrotor. The controller aims to make the quadrotor follow given waypoints in a 3D space. The code simulates and compares the performance of both infinite time horizon and finite time horizon LQR controllers.

## Dependencies

To run the code, you will need the following libraries installed:

- `numpy`
- `scipy`
- `matplotlib`

## Usage

To run the simulation and visualize the results, simply execute the main script:

```bash
python LQR_QUAD.py
```

The script will run the simulation for both infinite and finite time horizon LQR controllers and plot the errors in position tracking and the actual trajectories followed by the quadrotor.

## Overview

The main script `LQR_QUAD.py` includes the following functions:

- `solve_riccati_continuous`: Solves the Riccati differential equation for a continuous finite time horizon using the terminal value approach.
- `riccati_dynamics`: The dynamics of the Riccati differential equation.
- `lqr`: Solves the continuous time LQR controller for infite time horizon.


The script also includes the definition of the quadrotor system and the linearized subsystems for the x, y, z, and yaw dynamics.


## Parameters

The following parameters are defined in `djiphantom_params`:

- `B`: Force constant (estimated) [F=b*w^2]
- `D`: Torque constant (estimated) [T=d*w^2]
- `M`: Mass of the drone [kg] (source: https://www.dji.com/phantom-4/info)
- `L`: Arm length of the drone [m] (source: https://www.dji.com/phantom-4/info)
- `Ix`: Moment of inertia along the x-axis (estimated) [kg*m^2]
- `Iy`: Moment of inertia along the y-axis (estimated) [kg*m^2]
- `Iz`: Moment of inertia along the z-axis (estimated) [kg*m^2]
- `Jr`: Inertia of the propellers [kg*m^2] (estimated)
- `maxrpm`: Maximum RPM of the motors

## Usage

To use the `djiphantom_params` in your project, simply import the dictionary into your Python script:

```python
from nonlinearDynamics import djiphantom_params

# Access a specific parameter, e.g. the mass
mass = djiphantom_params['M']
```

## Results

After running the simulation, the script will generate a plot with 4 subplots:

1. Position error for reference tracking.
2. Comparison of x trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).
3. Comparison of y trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).
4. Comparison of z trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).

[Final Results](Results.png)

The comparison of the trajectories will provide insight into the performance of the LQR controllers in tracking the desired waypoints.
