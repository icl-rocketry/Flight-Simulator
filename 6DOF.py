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


# other constants
m_dry = 45.3  # kg - still with the motor casing
m_wet = 55.3  # kg
CG_dry = 2.58  # distance from center of mass to nose when dry
Isp = 200  # s
Cd_cyl = 1.17  # crossflow drag coefficient for a cylinder. Does not (yet) account for fins and canards
radius = 0.095
length = 4.645 # for now, also assumed to be where thrust is applied
CG_tank = 3.24 # distance from nose to centre of mass of the fuel
radius = 0.095
topA = 0.0296  # top area
sideA = np.pi * radius * length  # side area (not including fins)
Ilr_dry = np.array([[59.2, 0, 0], [0, 59.2, 0], [0, 0, 0.227]])  # moment of inertia of the rocket when dry
Ilr_fuel = np.array([[6.1, 0, 0], [0, 6.1, 0], [0, 0, 0.027]])  # moment of inertia of the fuel

# to do with aero surfaces
canardArea = 0  # m^2, per canard
canardNumber = 3  # number of canards
canardArm = 0.11  # distance from CoM to canards, radially
finArm = 0.18  # distance from CoM to fins, radially
sideFinsArea = 0.06  # m^2 (extra side area due to fins)
finArea = 0.0576  # area of individual fin
finNumber = 3
Cl_max_fins = 1.2  # maximum lift coefficient of the fins, assumed at 45 degrees
Cl_max_canards = 1.2  # maximum lift coefficient of the canards, assumed at 45 degrees

# to do with the launch rail
launchRailLength = 9  # m
launchRailAngle = 5 * np.pi / 180  # rad
launchRailDirection = 30 * np.pi / 180  # rad
separationAngle = 0.3  # rad - angle of attack at flow separation

# to do with the parachute
chuteDiameter = 1.5  # m
chuteCd = 1 # coefficient of drag
chuteArea = np.pi * (chuteDiameter / 2) ** 2  # m^2
chuteMargin = 5 # m - the static margin when the chute is deployed

# TODO: multi-stage rockets, put the data like mass etc. in an array with one element per stage


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


def CoP(mach, alpha, powerOn):  # TODO: this can be calculated using barrowman's method
    """This function returns the distance from the thrust vector to the pressure vector."""
    if powerOn:
        CoP = 1.545 - 0.2771 * mach
    else:
        CoP = 1.175 + 0.1687 * mach
    return length - CoP  # crude approximation from openrocket sims


def dragCoef(mach, alpha):  # this is ONLY for the body, not the fins or canards
    """This function returns the drag coefficient at a given mach number."""
    if mach < 0.04:
        Cd1 = 0.59 - 2 * mach
    elif mach > 0.04 and mach < 0.14:
        Cd1 = 0.514 - 0.1 * mach
    else:
        Cd1 = (
            0.1247 * mach ** 2 - 0.02382 * mach + 0.5009
        )  # valid for 0-0.8 mach, from openrocket sim TODO: make this more realistic at transonic speeds
    Cd = (
        Cd_cyl * np.sin(alpha) ** 2 + Cd1**2 * np.cos(alpha) ** 2
    ) ** 0.5  # when the rocket is at an angle, there is a side component of drag
    return Cd


# main functions
def flightUpdate(t, state, dt, turb, thrust_data):
    """This function returns the derivatives of the state vector which is input. This can then be used to solve the ODEs for the next time step.''
    state = [r, dr, q, w, m] --> [dr, ddr, dq, dw, dm]
    r = position vector (3x1)
    dr = velocity vector (3x1)
    q = quaternion (4x1) - but the actual data is stored as a 4x1 array
    w = angular velocity vector (3x1)
    m = mass (scalar)
    """
    is_wind = True
    # Unpack the state vector
    r = state[0:3]
    dr = state[3:6]
    q = np.quaternion(state[6], state[7], state[8], state[9]).normalized()  # quaternion representing orientation
    w = state[10:13]  # angular velocity as a quaternion
    m = state[13]

    # centre of mass moves - calculate it
    CG = (CG_dry * m_dry + CG_tank * (m - m_dry)) / m  # distance from center of mass to nose

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
    if not is_wind:
        wind = np.array([-np.cos(launchRailDirection), np.sin(launchRailDirection), 0]) * 4

    dr_wind = dr - wind
    # get the angle of attack
    if np.dot(dr, direction) == 0:
        alpha = 0
    else:
        alpha = np.arccos(np.dot(dr_wind, direction / np.linalg.norm(dr_wind)))
    mach = np.linalg.norm(dr_wind) / env.a  # mach number
    Cd = dragCoef(mach, alpha)  # drag coefficient
    rho = env.density
    refArea = topA * np.cos(alpha) ** 2 + (sideA + sideFinsArea) * np.sin(alpha) ** 2  # reference area
    D_translate = -0.5 * rho * Cd * topA * np.linalg.norm(dr_wind) * dr_wind # do I use refArea here?
    D_rotate = np.array([0, 0, 0])
    # TODO: add fin drag (this is probably the main reason the openrocket sims are different)
    D = D_translate + D_rotate
    # Lift
    L = np.array([0, 0, 0])  # TODO: add lift, this probably isnt that hard, just use Cl and alpha
    # Gravity
    G = np.array([0, 0, -m * env.g])

    # Inertia tensor and moment
    I = (
        Ilr_dry
        + m_dry * np.array([[(CG - CG_dry) ** 2, 0, 0], [0, (CG - CG_dry) ** 2, 0], [0, 0, 0]])
        + (m - m_dry)
        / (m_wet - m_dry)
        * (Ilr_fuel + (m - m_dry) * np.array([[(CG - CG_tank) ** 2, 0, 0], [0, (CG - CG_tank) ** 2, 0], [0, 0, 0]]))
    )  # TODO: openrocket seems to disagree with this?
    # centre of pressure moves too - but this is more complicated
    if np.linalg.norm(T) <= 1: # power off / power on drag
        CP = CoP(mach, alpha, False)  # distance from thrust location to pressure vector
    else:
        CP = CoP(mach, alpha, True)
    Kn = CP - CG
    # TODO: add fin drag moment (this is probably the main reason the openrocket sims are different)
    M_forces = np.cross(T, (length - CG) * direction) + np.cross(
        D + L, Kn * direction
    )  # moment - thrust contribution is currently zero but this supports TVC

    # Damping
    # TODO: check the equations here as I derived them myself
    pw = (
        w - np.dot(w, direction) * direction
    )  # this is the angular velocity in the plane perpendicular to the direction of the rocket, used for pitching
    eta = Cd_cyl * rho * np.linalg.norm(pw) ** 2 * pw * radius
    finAngle = np.pi / finNumber  # fins won't be perpendicular to the side flow
    Cd_fins = abs(2 - 8 / (np.pi**2) * abs((np.pi / 2 - finAngle) + alpha) ** 2)
    # get apparent incidence of fins, due to rocket rotation
    settingAngle = 0  # rad - angle of the canards to the rocket body
    vRocket = np.dot(dr_wind, direction) * direction
    incidence = np.arctan(np.linalg.norm(pw) * finArm / np.linalg.norm(vRocket))
    Cl_fins = Cl_max_fins * np.sin(incidence)  # TODO: find a better way to calculate this, this is awful
    Cl_canards = Cl_max_canards * np.sin(incidence + settingAngle)
    # moment from the rotation of the body
    M_damping_roll = 0.5 * rho * finArea * finArm * Cl_fins * np.linalg.norm(vRocket) * vRocket
    M_damping_pitch = eta / 4 * ((length - CG) ** 4 - CG**4) - (
        0.5 * rho * Cd_fins * sideFinsArea * (length - CG)**3 * np.linalg.norm(pw) * pw
    )
    M_canards = 0.5 * rho * canardArea * canardArm * Cl_canards * np.linalg.norm(vRocket) * vRocket
    M = M_forces + M_damping_roll + M_damping_pitch + M_canards
    # force from the rotation of the body
    D_damping = -eta / 3 * ((length - CG) ** 3 - CG**3)
    D = D + D_damping

    # derivative of the velocity vector can now be calculated
    diff_dr = 1 / m * (T + D + L + G)  # acceleration

    diff_w = np.linalg.inv(I).dot(M)

    # derivative of the mass
    diff_m = np.array([-(interpolate_thrust(t, thrust_data) / (env.g * Isp))])

    # derivative of the quaternion - used to apply the angular velocity to the direction
    w_quat = np.quaternion(0, w[0], w[1], w[2])  # angular velocity as a quaternion
    diff_q = quaternion.as_float_array(0.5 * w_quat * q)

    #TODO: move this to a separate function
    on_rail = dr[2] > 0 and r[2] < launchRailLength * np.cos(launchRailAngle)
    if on_rail:  # if the rocket is on the launch rail
        diff_w = np.array([0, 0, 0])  # prevent rotation
        diff_dr = (
            np.dot(direction, diff_dr) * direction / np.linalg.norm(direction)
        )  # prevent movement not in the direction of the rail

    # calculate pitch, roll and yaw to put in tracked values
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    newState = np.concatenate((diff_r, diff_dr, diff_q, diff_w, diff_m), axis=0)
    trackedValues = [diff_dr, roll, pitch, yaw, M, D, alpha, CG, CP, mach, T, L, I, wind, direction, t, Kn]
    return newState, trackedValues


# same params as flightUpdate but for the launch rail sim
def railUpdate(t, state, dt, turb):
    pass


# same params as flightUpdate but for the chute sim
def chuteUpdate(t, state, dt, turb):
    pass


# using my own solver so it goes in order of increasing time
def RK4(state, t, dt, turb, thrust_data):  # other methods can be used but this is a good start
    """This function uses the 4th order Runge-Kutta method to solve the ODEs for the next time step."""
    # calculate the derivatives of the state vector
    k1, trackedValues = flightUpdate(
        t, state, dt, turb, thrust_data
    )  # dt as the 3rd argument is only used for calculating rates, not part of the RK4 method
    k2, _ = flightUpdate(t + 0.5 * dt, state + 0.5 * dt * k1, dt, turb, thrust_data)
    k3, _ = flightUpdate(t + 0.5 * dt, state + 0.5 * dt * k2, dt, turb, thrust_data)
    k4, _ = flightUpdate(t + dt, state + dt * k3, dt, turb, thrust_data)

    # calculate the next state
    state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # store the state in the dictionary and append tracked values to the tracked list
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
    turb = env.getTurbulence(1000, t_end * 2)
    thrust_data = read_thrust_curve("BiProp.eng", m_wet * env.g)

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
    dt = 0.03  # time step
    t = 0  # initial time

    # using RK4
    while t < t_end and state[2] >= 0:
        state = RK4(state, t, dt, turb, thrust_data)
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
    axs[1, 1].plot(t, np.degrees(roll), label="roll")
    axs[1, 1].plot(t, np.degrees(pitch), label="pitch")
    axs[1, 1].plot(t, np.degrees(yaw), label="yaw")
    axs[1, 1].plot(
        t,
        [tracked_dict[t][6] * 180/np.pi for t in tracked_dict],
        label="angle of attack",
        color="black",
        alpha=0.2,
    )
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].set_ylabel("angle (°)")
    axs[1, 1].legend()
    axs[1, 1].grid(visible=True)

    # plot euler rates
    axs[1, 2].plot(t, np.degrees(rollRate), label="roll rate")
    axs[1, 2].plot(t, np.degrees(pitchRate), label="pitch rate")
    axs[1, 2].plot(t, np.degrees(yawRate), label="yaw rate")
    axs[1, 2].set_xlabel("time (s)")
    axs[1, 2].set_ylabel("angular rates (°/s)")
    axs[1, 2].legend()
    axs[1, 2].grid(visible=True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    # Function to update plot during animation
    def update_plot(num, rocket_line, vel_arrow, dir_arrow, time_text):
        rocket_line.set_data_3d(x[: num + 1], y[: num + 1], z[: num + 1])

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
        speed_text.set_text("Speed: {:.0f} m/s (M {:.2f})".format(vel_magnitude, machList[num]))
        alt_text.set_text("Altitude: {:.0f} m".format(z[num]))
        wind_text.set_text(
            "Wind: {:.1f} m/s @ {:.0f}°".format(
                np.linalg.norm([windList[num][0], windList[num][1], windList[num][2]]),
                np.degrees(np.arctan2(windList[num][1], windList[num][0]) % (2 * np.pi)),
            )
        )
        aoa_text.set_text("Angle of attack: {:.1f}°".format(np.degrees(aoaList[num])))

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
    machList = [tracked_dict[t][9] for t in tracked_dict]
    aoaList = [tracked_dict[t][6] for t in tracked_dict]

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
    aoa_text = ax.text2D(0.05, 0.75, "", transform=ax.transAxes)

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
    # make it fullscreen
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
