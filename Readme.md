# Finite Time Horizon LQR Control for Quadrotor

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

## Results

After running the simulation, the script will generate a plot with 4 subplots:

1. Position error for reference tracking.
2. Comparison of x trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).
3. Comparison of y trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).
4. Comparison of z trajectories followed by the quadrotor with command and LQR control (both infinite and finite time horizons).

The comparison of the trajectories will provide insight into the performance of the LQR controllers in tracking the desired waypoints.
