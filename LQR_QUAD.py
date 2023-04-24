from platform import mac_ver
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
from numpy.linalg import norm


# time instants for simulation
t_max = 10
time_step = 0.01
t = np.arange(0., t_max, time_step)

# Inintial state
X0 = np.zeros(12)
tf = 10
t_span = [0, tf]

def solve_riccati_continuous(A, B, Q, R, P_T,T = tf,t_for_eval = t):
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
    sol = solve_ivp(riccati_dynamics, t_span, P_T,t_eval=t_for_eval)
    P = sol.y[:,:]
    # .reshape((len(A), len(A)))
    return P




def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)



seed = 77
np.random.seed(seed)


# The control can be done in a decentralized style
# The linearized system is divided into four decoupled subsystems

# X-subsystem
# The state variables are x, dot_x, pitch, dot_pitch
Ax = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
Bx = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Ix]])



# Y-subsystem
# The state variables are y, dot_y, roll, dot_roll
Ay = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
By = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Iy]])

# Z-subsystem
# The state variables are z, dot_z
Az = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Bz = np.array(
    [[0.0],
     [1 / (m)]])

# Yaw-subsystem
# The state variables are yaw, dot_yaw
Ayaw = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Byaw = np.array(
    [[0.0],
     [1 / Iz]])

# solve LQR 
Ks = []  # feedback gain matrices K for each subsystem
P_IFT_List = []
P_FT_List = []
K_finite = []
P_FT_List_all = []
for A, B in ((Ax, Bx), (Ay, By), (Az, Bz), (Ayaw, Byaw)):
    # infite horizon solution 
    n = A.shape[0]
    m_ = B.shape[1]
    Q = np.eye(n)
    Q[0, 0] = 10.  # The first state variable is the one we care about.
    R = np.diag([1., ])
    K, P_IFT, _ = lqr(A, B, Q, R)
    Ks.append(K)
    P_IFT_List.append(P_IFT)

    # fnite time horizon 
    P_T = np.zeros(A.shape)
    P_FT_List = list()
    P = solve_riccati_continuous(A, B, Q, R, P_T)
    P_FT_List.append(P)
    K_finite_over_time = []
    for j,_ in enumerate(t):
    
        P_ = P[:,len(t)-j-1].reshape((len(A), len(A)))
        # print(P)
        K_ = (inv(R) @ B.T @ P_).reshape(-1)
        # print(K_)
        for i in range(len(K_)):
            if i%2==1:
                K_[i] = -K_[i]
        K = K_.reshape(B.shape[0])
        K_finite_over_time.append(K)

    K_finite.append(np.array(K_finite_over_time))
    P_FT_List_all.append(P_FT_List)


#simulate



def cl_nonlinear(x, t, u):
    x = np.array(x)
    dot_x = nonlinearDynamics.f(x, u(x, t) + np.array([m * g, 0, 0, 0]))
    return dot_x


# follow waypoints
signal = np.zeros([len(t), 4])

signalx = signal[:, 0] + 4
signaly = signal[:, 1] + 8
signalz = signal[:, 2] + 12
signalyaw = signal[:, 3]  # the signal that yaw follows is 0


def u(x, _t):
    # the controller
    dis = _t - t
    dis[dis < 0] = np.inf
    idx = dis.argmin()
    UX = Ks[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0]
    UY = Ks[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
    UZ = Ks[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
    UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
    return np.array([UZ, UY, UX, UYaw])

def u_finite(x, _t):
    # the controller
    dis = _t - t
    dis[dis < 0] = np.inf
    idx = dis.argmin()
    UX = K_finite[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0]
    UY = K_finite[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
    UZ = K_finite[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
    UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
    return np.array([UZ, UY, UX, UYaw])


# simulate


x_nl_finite = odeint(cl_nonlinear, X0, t, args=(u_finite,))
x_nl = odeint(cl_nonlinear, X0, t, args=(u,))


# Plot

fig = plt.figure()


errors = fig.add_subplot(2, 2, 1)
x1_plt = fig.add_subplot(2,2,2)
x2_plt = fig.add_subplot(2,2,3)
x3_plt = fig.add_subplot(2,2,4)



errors.plot(t, signalx - x_nl[:, 0], color="r", label='x error (Infinite Time)')
errors.plot(t, signaly - x_nl[:, 2], color="g", label='y error (Infinite Time)')
errors.plot(t, signalz - x_nl[:, 4], color="b", label='z error (Infinite Time)')
errors.plot(t, signalx - x_nl_finite[:, 0], "r-.", label='x error (Finite Time)')
errors.plot(t, signaly - x_nl_finite[:, 2], "g-.", label='y error (Finite Time)')
errors.plot(t, signalz - x_nl_finite[:, 4], "b-.", label='z error (Finite Time)')
errors.set_title("Position error for reference tracking")
errors.set_xlabel("time {s}")
errors.set_ylabel("error")
errors.legend(loc='lower right', shadow=True, fontsize='small')


x1_plt.plot(t,x_nl[:, 0], color="r", label="Infinite Time")
x1_plt.plot(t,signalx, color="b", label="command")

x2_plt.plot(t,x_nl[:, 2], color="r", label="Infinite Time")
x2_plt.plot(t,signaly, color="b", label="command")

x3_plt.plot(t,x_nl[:, 4], color="r", label="Infinite Time")
x3_plt.plot(t,signalz, color="b", label="command")


x1_plt.plot(t,x_nl_finite[:, 0], "g-.", label="Finite Time")
x2_plt.plot(t,x_nl_finite[:, 2], "g-.", label="Finite Time")
x3_plt.plot(t,x_nl_finite[:, 4], "g-.", label="Finite Time")

x1_plt.set_title("x")
x1_plt.set_xlabel("time {s}")
x1_plt.set_ylabel("x")
x1_plt.legend(loc='lower right', shadow=True, fontsize='small')

x2_plt.set_title("y")
x2_plt.set_xlabel("time {s}")
x2_plt.set_ylabel("y")
x2_plt.legend(loc='lower right', shadow=True, fontsize='small')

x3_plt.set_title("z")
x3_plt.set_xlabel("time {s}")
x3_plt.set_ylabel("z")
x3_plt.legend(loc='lower right', shadow=True, fontsize='small')

plt.show()  


fig1, errors1 = plt.subplots()
errors1.plot(t, signalx - x_nl[:, 0], color="r", label='x error (Infinite Time)')
errors1.plot(t, signaly - x_nl[:, 2], color="g", label='y error (Infinite Time)')
errors1.plot(t, signalz - x_nl[:, 4], color="b", label='z error (Infinite Time)')
errors1.plot(t, signalx - x_nl_finite[:, 0], "r-.", label='x error (Finite Time)')
errors1.plot(t, signaly - x_nl_finite[:, 2], "g-.", label='y error (Finite Time)')
errors1.plot(t, signalz - x_nl_finite[:, 4], "b-.", label='z error (Finite Time)')
errors1.set_title("Position error for reference tracking")
errors1.set_xlabel("time {s}")
errors1.set_ylabel("error")
errors1.legend(loc='lower right', shadow=True, fontsize='small')

fig1.savefig('errors1.png')

fig2, x1_plt2 = plt.subplots()
x1_plt2.plot(t,x_nl[:, 0], color="r", label="Infinite Time")
x1_plt2.plot(t,signalx, color="b", label="command")
x1_plt2.plot(t,x_nl_finite[:, 0], "g-.", label="Finite Time")
x1_plt2.set_title("x")
x1_plt2.set_xlabel("time {s}")
x1_plt2.set_ylabel("x")
x1_plt2.legend(loc='lower right', shadow=True, fontsize='small')

fig2.savefig('x1_plt2.png')

fig3, x2_plt2 = plt.subplots()
x2_plt2.plot(t,x_nl[:, 2], color="r", label="Infinite Time")
x2_plt2.plot(t,signaly, color="b", label="command")
x2_plt2.plot(t,x_nl_finite[:, 2], "g-.", label="Finite Time")
x2_plt2.set_title("x")
x2_plt2.set_xlabel("time {s}")
x2_plt2.set_ylabel("x")
x2_plt2.legend(loc='lower right', shadow=True, fontsize='small')

fig3.savefig('x2_plt2.png')

fig4, x3_plt2 = plt.subplots()
x3_plt2.plot(t,x_nl[:, 4], color="r", label="Infinite Time")
x3_plt2.plot(t,signalz, color="b", label="command")
x3_plt2.plot(t,x_nl_finite[:, 4], "g-.", label="Finite Time")
x3_plt2.set_title("x")
x3_plt2.set_xlabel("time {s}")
x3_plt2.set_ylabel("x")
x3_plt2.legend(loc='lower right', shadow=True, fontsize='small')

fig4.savefig('x3_plt2.png')

L2_k = list()

for i in range(3):
    L2_norm_all_k = list()    

    for j in range(len(t)):


        L2_norm_k = norm(K_finite[i][j]-Ks[i])
        L2_norm_all_k.append(L2_norm_k)

    L2_k.append(L2_norm_all_k)






fig5, K_1_plot = plt.subplots()
K_1_plot.plot(t,L2_k[0], "g-", label="Kx")
K_1_plot.plot(t,L2_k[1], "r-.", label="Ky")
K_1_plot.plot(t,L2_k[2], "b-.", label="Kz")
K_1_plot.set_title("Norm Difference Between Finite and Infinite")
K_1_plot.set_xlabel("time {s}")
K_1_plot.set_ylabel("Norm Difference")
K_1_plot.legend(loc='lower right', shadow=True, fontsize='small')
fig5.savefig('K_Comparison.png')


# fig6, K_z2_plot = plt.subplots()

# K_z2_plot.plot(t,K_finite[2][1], "g-.", label="Finite Time")
# K_z2_plot.plot(t,Ks[2], "g-.", label="Infinite Time")

# fig6.savefig('K_z1_Comparison.png')
   
