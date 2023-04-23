import numpy as np 


djiphantom_params = {

    # Estimated
    'B': 5.E-06,  # force constatnt [F=b*w^2]
    'D': 2.E-06,  # torque constant [T=d*w^2]

    # https:#www.dji.com/phantom-4/info
    'M': 1.380,  # mass [kg]
    'L': 0.350,  # arm length [m]

    # Estimated
    'Ix': 2,       # [kg*m^2]
    'Iy': 2,       # [kg*m^2]
    'Iz': 3,       # [kg*m^2]
    'Jr': 38E-04,  # prop inertial [kg*m^2]

    'maxrpm': 15000
}

g = 9.81
m = djiphantom_params['M']
Ix = djiphantom_params['Ix']
Iy = djiphantom_params['Iy']
Iz = djiphantom_params['Iz']

def f(x, u):
    x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2 = x.reshape(-1).tolist()
    ft, tau_x, tau_y, tau_z = u.reshape(-1).tolist()
    dot_x = np.array([
     x2,
     ft/m*(np.sin(phi1)*np.sin(psi1)+np.cos(phi1)*np.cos(psi1)*np.sin(theta1)),
     y2,
     ft/m*(np.cos(phi1)*np.sin(psi1)*np.sin(theta1)-np.cos(psi1)*np.sin(phi1)),
     z2,
     -g+ft/m*np.cos(phi1)*np.cos(theta1),
     phi2,
     (Iy-Iz)/Ix*theta2*psi2+tau_x/Ix,
     theta2,
     (Iz-Ix)/Iy*phi2*psi2+tau_y/Iy,
     psi2,
     (Ix-Iy)/Iz*phi2*theta2+tau_z/Iz])
    return dot_x