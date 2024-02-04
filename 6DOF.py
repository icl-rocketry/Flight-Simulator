# imports
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Atmosphere2 import *
from Rocket2 import *

# set default style
plt.style.use("seaborn-v0_8-bright")


# create a dictionary to store the {time: state} pairs
state_dict = {}
tracked_dict = {}  # stores {time: tracked values} pairs
logger = [1e-6]  # to prevent division by zero errors
eulerLogger = [[0, 0, 0]]


# other constants
m_dry = 9.74  # kg - still with the motor casing
m_wet = 12.155  # kg
Rt_dry = 0.795  # distance from center of mass to thrust vector (when dry)
Isp = 204  # s
Cd_cyl = 1.17  # crossflow drag coefficient for a cylinder. Does not (yet) account for fins and canards
radius = 0.06
length = 2.51
R_tank = 0.195
launchRailLength = 9  # m
launchRailAngle = 5 * np.pi / 180  # rad
launchRailDirection = 30 * np.pi / 180  # rad
separationAngle = 0.3  # rad - angle of attack at flow separation
g0 = 9.81  # m/s^2
sideA = 0.2854
topA = 0.0114
Ilr_dry = np.array([[6.55, 0, 0], [0, 6.55, 0], [0, 0, 0.0219]])  # moment of inertia of the rocket when dry
Ilr_fuel = np.array([[0.0329, 0, 0], [0, 0.0329, 0], [0, 0, 0.0029]])  # moment of inertia of the fuel
canardArea = 0.004  # m^2, per canard
canardNumber = 3  # number of canards
canardArm = 0.11  # distance from CoM to canards, radially
finArm = 0.15  # distance from CoM to fins, radially
sideFinsArea = 0.02  # m^2 (area of fins when viewed from the side, including the canards)
finArea = 0.015  # area of individual fin
finNumber = 3
Cl_max_fins = 1.2  # maximum lift coefficient of the fins, assumed at 45 degrees
Cl_max_canards = 1.2  # maximum lift coefficient of the canards, assumed at 45 degrees

# Rockets
Rocket = None  # Rocket(5, 0.1)


# other functions


def read_thrust_curve(filename, weight):
    time_thrust_data = [(0, weight)]  # Add a point at t=0, F=0 to the beginning of the dat

    # Open the file for reading
    with open(filename, "r") as file:
        lines = file.readlines()

        # Iterate through the lines
        for line in lines:
            # Split the line into columns
            columns = line.split()
            if len(columns) >= 2:  # Make sure there are at least 2 columns, for time and thrust
                try:
                    # Convert the time and thrust columns to float
                    time = float(columns[0])
                    thrust = float(columns[1])
                    if time < 1 and thrust < weight:
                        thrust = weight
                    time_thrust_data.append((time, thrust))  # Add the data point to the list
                except ValueError:
                    pass  # Ignore lines with invalid data

    return time_thrust_data


thrust_data = read_thrust_curve("Cesaroni_4842L610-P.eng", m_wet * g0)


def interpolate_thrust(t, thrust_data):
    # Find the first data point with a time greater than t
    for i in range(len(thrust_data)):
        if thrust_data[i][0] > t:
            break

    # If we are past the end of the data, return 0 thrust
    if i == len(thrust_data) - 1:
        return 0

    # If we are at the beginning of the data, return the first thrust value
    if i == 0:
        return thrust_data[0][1]

    # Otherwise, interpolate between the two data points
    t1, thrust1 = thrust_data[i - 1]
    t2, thrust2 = thrust_data[i]
    return thrust1 + (thrust2 - thrust1) * (t - t1) / (t2 - t1)


def CoP(Rocket, mach, alpha):  # TODO: this can be calculated using barrowman's method
    """This function returns the distance from the thrust vector to the pressure vector."""
    # OLD CODE:
    CoP = length - (1.83 + mach / 20)
    if mach < 0.25:
        CoP = length - 1.8425 - 0.25 * (0.25 - mach)
    return CoP  # crude approximation from openrocket sims
    # NEW CODE (using the Rocket class):


def dragCoef(mach, alpha):  # TODO: with the new sim this can be calculated in a less dodgy way
    """This function returns the drag coefficient at a given mach number."""
    if mach < 0.04:
        Cd1 = 0.624 - (0.1 / 0.04) * mach
    else:
        Cd1 = (
            0.524 + 0.115 * mach**2
        )  # valid for 0-0.8 mach, from openrocket sim TODO: make this more realistic at transonic speeds
    Cd = (
        1.37 * np.sin(alpha) ** 4 + Cd1**2 * np.cos(alpha) ** 4
    ) ** 0.5  # when the rocket is at an angle, there is a side component of drag
    # TODO: use separationAngle to find the angle of attack at which flow separation occurs, and use this to find a higher Cd
    return Cd


# main functions
def recalculate(t, state, dt, turb):
    """This function returns the derivatives of the state vector which is input. This can then be used to solve the ODEs for the next time step.''
    state = [r, dr, q, w, m] --> [dr, ddr, dq, dw, dm]
    r = position vector (3x1)
    dr = velocity vector (3x1)
    q = quaternion (4x1) - but the actual data is stored as a 4x1 array
    w = angular velocity vector (3x1)
    m = mass (scalar)
    initialCall is a boolean which is True if this is the first time the function is called for this time step, and used to find old values
    """
    is_wind = True

    # Unpack the state vector
    r = state[0:3]
    dr = state[3:6]
    q = np.quaternion(state[6], state[7], state[8], state[9]).normalized()  # quaternion representing orientation
    w = state[10:13]  # angular velocity as a quaternion
    m = state[13]

    # centre of mass moves - calculate it
    Rt = (Rt_dry * m_dry + R_tank * (m - m_dry)) / state[
        13
    ]  # distance from center of mass to thrust vector at current time

    """Calculate the derivatives of the state vector"""

    # derivative of the position vector
    diff_r = dr

    # get atmospheric properties for the current altitude
    env.atmosphere(r[2], dr[2] * dt)  # update the atmosphere

    # derivative of the velocity vector requires the forces acting on the rocket
    # Thrust - to get the direction we need to convert the quaternion to a rotation matrix
    direction = quaternion.as_rotation_matrix(q).dot(np.array([0, 0, 1]))  # direction of the rocket
    T = interpolate_thrust(t, thrust_data) * direction / np.linalg.norm(direction)  # thrust vector
    # Drag
    # wind
    # get the index we need in the turbulence data
    i = int(t / dt)
    uWind, vWind = env.getUpperLevelWinds(r[2])  # get the wind at the current time
    totalSpeed = np.sqrt(uWind**2 + vWind**2)  # empirical factor to convert from wind to gust
    uWind = uWind + (turb[0][i] * totalSpeed)  # add turbulence
    vWind = vWind + (turb[1][i] * totalSpeed)  # add turbulence
    wWind = turb[2][i] * totalSpeed * 0.1
    wind = np.array([uWind, vWind, wWind])
    if is_wind == False:
        wind = np.array([0, 0, 0])

    dr_wind = dr - wind
    # cross-sectional area, where alpha is the angle between rocket direction and velocity vector
    if np.dot(dr, direction) == 0:
        alpha = 0
    else:
        alpha = np.arccos(np.dot(dr_wind, direction / np.linalg.norm(dr_wind)))
    mach = np.linalg.norm(dr_wind) / np.sqrt(1.4 * 287 * env.temperature)  # mach number
    Cd = dragCoef(mach, alpha)  # drag coefficient
    rho = env.density
    D_translate = -0.5 * rho * Cd * topA * np.linalg.norm(dr_wind) * dr_wind
    D_rotate = np.array([0, 0, 0])
    D = D_translate + D_rotate
    on_rail = dr[2] > 0 and r[2] < launchRailLength * np.cos(launchRailAngle)
    if on_rail:
        D += 0.1 * D / np.linalg.norm(D)  # add rail friction
    # Lift
    L = np.array([0, 0, 0])  # TODO: add lift, this probably isnt that hard, just use Cl and alpha
    # Gravity
    G = np.array([0, 0, -m * env.g])

    # derivative of the quaternion
    w_quat = np.quaternion(0, w[0], w[1], w[2])  # angular velocity as a quaternion
    diff_q = quaternion.as_float_array(0.5 * w_quat * q)

    # Inertia tensor and moment
    I = (
        Ilr_dry
        + m_dry * np.array([[(Rt - Rt_dry) ** 2, 0, 0], [0, (Rt - Rt_dry) ** 2, 0], [0, 0, 0]])
        + (m - m_dry)
        / (m_wet - m_dry)
        * (Ilr_fuel + (m - m_dry) * np.array([[(Rt - R_tank) ** 2, 0, 0], [0, (Rt - R_tank) ** 2, 0], [0, 0, 0]]))
    )  # TODO: openrocket seems to disagree with this?
    # centre of pressure moves too - but this is more complicated
    Rpt = CoP(Rocket, mach, alpha)  # distance from thrust location to pressure vector
    Rp = Rt - Rpt
    M_forces = np.cross(T, Rt * direction) + np.cross(
        D + L, Rp * direction
    )  # moment - thrust contribution is currently zero but this supports TVC

    # Damping
    # TODO: check the equations here as I derived them myself
    eta = Cd_cyl * rho * np.linalg.norm(w) ** 2 * w * radius
    finAngle = np.pi / finNumber  # fins won't be perpendicular to the side flow
    Cd_fins = abs(2 - 8 / (np.pi**2) * abs((np.pi / 2 - finAngle) + alpha) ** 2)
    pw = (
        w - np.dot(w, direction) * direction
    )  # this is the angular velocity in the plane perpendicular to the direction of the rocket, used for pitching
    # get apparent incidence of fins, due to rocket rotation
    settingAngle = 0  # rad - angle of the canards to the rocket body
    vRocket = np.dot(dr_wind, direction) * direction
    incidence = np.arctan(np.linalg.norm(pw) * finArm / np.linalg.norm(vRocket))
    Cl_fins = Cl_max_fins * np.sin(incidence)  # TODO: find a better way to calculate this, this is awful
    Cl_canards = Cl_max_canards * np.sin(incidence + settingAngle)
    # moment from the rotation of the body
    M_damping_roll = 0.5 * rho * finArea * finArm * Cl_fins * np.linalg.norm(vRocket) * vRocket
    M_damping_pitch = -eta / 4 * ((length - Rt) ** 4 - Rt**4) - (
        0.5 * rho * Cd_fins * sideFinsArea * Rt**3 * np.linalg.norm(pw) * pw
    )
    M_canards = 0.5 * rho * canardArea * canardArm * Cl_canards * np.linalg.norm(vRocket) * vRocket
    M = M_forces + M_damping_roll + M_damping_pitch + M_canards
    # force from the rotation of the body
    D_damping = -eta / 3 * ((length - Rt) ** 3 - Rt**3)
    D = D + D_damping

    # derivative of the velocity vector can now be calculated
    diff_dr = 1 / m * (T + D + L + G)  # acceleration

    diff_w = np.linalg.inv(I).dot(M)

    # derivative of the mass
    diff_m = np.array([-(interpolate_thrust(t, thrust_data) / (env.g * Isp))])

    if on_rail:  # if the rocket is on the launch rail
        diff_w = np.array([0, 0, 0])  # prevent rotation
        diff_q = np.array([0, 0, 0, 0])  # prevent rotation
        diff_dr = (
            np.dot(direction, diff_dr) * direction / np.linalg.norm(direction)
        )  # prevent movement not in the direction of the rail

    # calculate pitch, roll and yaw to put in tracked values
    i2 = len(eulerLogger) - 1
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    newState = np.concatenate((diff_r, diff_dr, diff_q, diff_w, diff_m), axis=0)
    trackedValues = [diff_dr, roll, pitch, yaw, M, D, alpha, Rt, Rp, mach, T, L, I, wind, direction, t]
    return newState, trackedValues


# using my own solver so it goes in order of increasing time


def RK4(state, t, dt, turb):  # other methods can be used but this is a good start
    """This function uses the 4th order Runge-Kutta method to solve the ODEs for the next time step."""
    # calculate the derivatives of the state vector
    k1, trackedValues = recalculate(
        t, state, dt, turb
    )  # dt as the 3rd argument is only used for calculating rates, not part of the RK4 method
    k2, _ = recalculate(t + 0.5 * dt, state + 0.5 * dt * k1, dt, turb)
    k3, _ = recalculate(t + 0.5 * dt, state + 0.5 * dt * k2, dt, turb)
    k4, _ = recalculate(t + dt, state + dt * k3, dt, turb)

    # calculate the next state
    state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # store the state in the dictionary and append tracked values to the tracked list
    eulerLogger.append(trackedValues[1:4])
    rounded = round(t, 2)
    state_dict[rounded] = state
    tracked_dict[rounded] = trackedValues
    return state


def smooth_angles(angles):
    smoothed_angles = [angles[0]]  # Initialize with the first angle
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        # Check for a downward jump
        if diff < -np.pi:
            diff += 2 * np.pi
        # Check for an upward jump
        elif diff > np.pi:
            diff -= 2 * np.pi
        smoothed_angles.append(smoothed_angles[-1] + diff)
    return smoothed_angles


def main():
    """This function runs the simulation."""

    # wind
    t_end = 100  # end time
    global env
    env = Environment()
    env.getForecast()  # this must be done before the simulation starts - could do it in __init__
    turb = env.getTurbulence(1000, t_end + 1)

    # Initial conditions
    r = np.array([0, 0, 0])
    dr = np.array([0, 0, 0])
    alpha = launchRailAngle  # angle of the rocket from the vertical (rad)
    q_quat = quaternion.from_rotation_vector(
        np.array([alpha * np.sin(launchRailDirection), alpha * np.cos(launchRailDirection), 0])
    )
    # make sure roll is zero
    q = quaternion.as_float_array(q_quat)
    w = np.array([0, 0, 0])
    m = np.array([m_wet])  # dry mass

    # instantiate the initial state vector
    state = np.concatenate((r, dr, q, w, m), axis=0)

    # Simulation parameters
    dt = 0.04 # time step
    t = 0  # initial time

    # using RK4
    while t < t_end and state[2] >= 0:
        state = RK4(state, t, dt, turb)
        print(round(t, 2), "seconds", end="\r")
        t += dt

    # extract some values
    t = [t for t in state_dict]
    r = [state_dict[t][0:3] for t in state_dict]
    dr = [state_dict[t][3:6] for t in state_dict]

    # Plotting
    trackedTime = [tracked_dict[t][15] for t in tracked_dict]
    # smooth the euler angles
    roll = smooth_angles([tracked_dict[t][1] for t in tracked_dict])
    pitch = smooth_angles([tracked_dict[t][2] for t in tracked_dict])
    yaw = smooth_angles([tracked_dict[t][3] for t in tracked_dict])
    # calculate euler rates based on smoothed angles
    rollRate = np.gradient(roll, trackedTime)
    pitchRate = np.gradient(pitch, trackedTime)
    yawRate = np.gradient(yaw, trackedTime)
    # create figure with 6 subplots
    fig, axs = plt.subplots(2, 3, constrained_layout=True)

    # plot x,y,z position all in one plot
    axs[0, 0].plot(t, r, label=["x", "y", "z"])
    axs[0, 0].set_xlabel("time (s)")
    axs[0, 0].set_ylabel("position (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(visible=True)

    # plot x,y,z velocity all in one plot and total velocity magnitude
    axs[0, 1].plot(t, dr, label=["x", "y", "z"])
    axs[0, 1].plot(t, [np.linalg.norm(dr[i]) for i in range(len(dr))], label="total", color="black", alpha=0.2)
    axs[0, 1].set_xlabel("time (s)")
    axs[0, 1].set_ylabel("velocity (m/s)")
    axs[0, 1].legend()
    axs[0, 1].grid(visible=True)

    # plot x,y,z acceleration all in one plot and total acceleration magnitude
    axs[0, 2].plot(t, [tracked_dict[t][0] for t in tracked_dict], label=["x", "y", "z"])
    axs[0, 2].plot(
        t, [np.linalg.norm(tracked_dict[t][0]) for t in tracked_dict], label="total", color="black", alpha=0.2
    )
    axs[0, 2].set_xlabel("time (s)")
    axs[0, 2].set_ylabel("acceleration (m/s^2)")
    axs[0, 2].legend()
    axs[0, 2].grid(visible=True)

    # plot thrust magnitude, drag magnitude and lift magnitude all in one plot
    axs[1, 0].plot(t, [np.linalg.norm(tracked_dict[t][10]) for t in tracked_dict], label="thrust")
    axs[1, 0].plot(t, [np.linalg.norm(tracked_dict[t][5]) for t in tracked_dict], label="drag")
    # axs[1, 0].plot(t, [np.linalg.norm(tracked_dict[t][11]) for t in tracked_dict], label="lift")
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("force (N)")
    axs[1, 0].legend()
    axs[1, 0].grid(visible=True)

    # plot pitch, roll and yaw all in one plot and angle of attack
    axs[1, 1].plot(t, roll, label="roll")
    axs[1, 1].plot(t, pitch, label="pitch")
    axs[1, 1].plot(t, yaw, label="yaw")
    axs[1, 1].plot(
        t,
        [tracked_dict[t][6] for t in tracked_dict],
        label="angle of attack",
        color="black",
        alpha=0.2,
    )
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].set_ylabel("angle (rad)")
    axs[1, 1].legend()
    axs[1, 1].grid(visible=True)

    # plot euler rates
    axs[1, 2].plot(t, rollRate, label="roll rate")
    axs[1, 2].plot(t, pitchRate, label="pitch rate")
    axs[1, 2].plot(t, yawRate, label="yaw rate")
    axs[1, 2].set_xlabel("time (s)")
    axs[1, 2].set_ylabel("angular rates (rad/s)")
    axs[1, 2].legend()
    axs[1, 2].grid(visible=True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    # Function to update plot during animation
    def update_plot(num, rocket_line, vel_arrow, dir_arrow, time_text):
        rocket_line.set_data_3d(x[:num+1], y[:num+1], z[:num+1])

        # Scale velocity vector
        vel_magnitude = np.linalg.norm([u[num], v[num], w[num]])
        scaled_uvw = [20 * ui / vel_magnitude for ui in [u[num], v[num], w[num]]]
        vel_arrow.set_segments(
            [[[x[num], y[num], z[num]], [x[num] + scaled_uvw[0], y[num] + scaled_uvw[1], z[num] + scaled_uvw[2]]]]
        )

        # Add direction vector based on pitch, yaw, and roll
        dir_uvw = 20 * directionList[num]
        dir_arrow.set_segments(
            [[[x[num], y[num], z[num]], [x[num] + dir_uvw[0], y[num] + dir_uvw[1], z[num] + dir_uvw[2]]]]
        )

        time_text.set_text("Time: {:.1f} s".format(time[num]))
        speed_text.set_text("Speed: {:.0f} m/s".format(vel_magnitude))
        alt_text.set_text("Altitude: {:.0f} m".format(z[num]))
        wind_text.set_text("Wind: {:.1f} m/s".format(np.linalg.norm([windList[num][0], windList[num][1], windList[num][2]])))

        # Adjust axis limits to make the rocket appear larger
        ax.set_xlim([x[num] - 20, x[num] + 20])
        ax.set_ylim([y[num] - 20, y[num] + 20])
        ax.set_zlim([z[num] - 20, z[num] + 20])

    # Rocket parameters
    time = [t for t in state_dict]
    time_steps = len(time)
    directionList = []
    x = [state_dict[t][0] for t in state_dict]
    y = [state_dict[t][1] for t in state_dict]
    z = [state_dict[t][2] for t in state_dict]
    u = [state_dict[t][3] for t in state_dict]
    v = [state_dict[t][4] for t in state_dict]
    w = [state_dict[t][5] for t in state_dict]
    directionList = [tracked_dict[t][14] for t in tracked_dict]
    windList = [tracked_dict[t][13] for t in tracked_dict]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot rocket position and velocity arrow
    (rocket_line,) = ax.plot(x, y, z, label="Rocket Trajectory", color="y", linewidth=1)
    vel_arrow = ax.quiver(x[0], y[0], z[0], u[0], v[0], w[0], color="r", label="Velocity", linewidth=1)

    # Plot direction vector based on pitch, yaw, and roll
    dir_arrow = ax.quiver(x[0], y[0], z[0], 0, 0, 0, color="k", label="Rocket")

    # Set initial axis limits
    ax.set_xlim([x[0] - 20, x[0] + 20])
    ax.set_ylim([y[0] - 20, y[0] + 20])
    ax.set_zlim([z[0] - 20, z[0] + 20])

    # Set labels
    ax.set_xlabel("Distance north of launch site (m)")
    ax.set_ylabel("Distance east of launch site (m)")
    ax.set_zlabel("Altitude (m)")

    # Add time, speed, and alt display in legend
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    speed_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
    alt_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)
    wind_text = ax.text2D(0.05, 0.80, "", transform=ax.transAxes)

    # Create animation
    ani = FuncAnimation(
        fig,
        update_plot,
        frames=time_steps,
        fargs=(rocket_line, vel_arrow, dir_arrow, time_text),
        interval=0,
        blit=False,
    )

    # Display the animation
    plt.legend(loc="upper right")
    plt.show()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


if __name__ == "__main__":
    main()
