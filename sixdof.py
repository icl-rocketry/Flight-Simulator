# imports
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
import numpy as np
import quaternion
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Atmosphere2 import *
from rocket3 import *
from simulation import *

# set default style
plt.style.use("seaborn-v0_8-bright")


# create a dictionary to store the {time: state} pairs
state_dict = {}
tracked_dict = {}  # stores {time: tracked values} pairs
logger = [1e-6]  # to prevent division by zero errors
eulerLogger = [[0, 0, 0]]


# # other constants
# m_dry = 9.74  # kg - still with the motor casing
# m_wet = 12.155  # kg
# Rt_dry = 0.795  # distance from center of mass to thrust vector (when dry)
# Isp = 204  # s
# Cd_cyl = 1.17  # crossflow drag coefficient for a cylinder. Does not (yet) account for fins and canards
# radius = 0.06
# length = 2.51
# R_tank = 0.195
# launchRailLength = 9  # m
# launchRailAngle = 5 * np.pi / 180  # rad
# launchRailDirection = 30 * np.pi / 180  # rad
# separationAngle = 0.3  # rad - angle of attack at flow separation
# g0 = 9.81  # m/s^2
# sideA = 0.2854
# topA = 0.0114
# Ilr_dry = np.array([[6.55, 0, 0], [0, 6.55, 0], [0, 0, 0.0219]])  # moment of inertia of the rocket when dry
# Ilr_fuel = np.array([[0.0329, 0, 0], [0, 0.0329, 0], [0, 0, 0.0029]])  # moment of inertia of the fuel
# canardArea = 0.004  # m^2, per canard

# canardNumber = 3  # number of canards
# canardArm = 0.11  # distance from CoM to canards, radially
# finArm = 0.15  # distance from CoM to fins, radially
# sideFinsArea = 0.02  # m^2 (area of fins when viewed from the side, including the canards)
# finArea = 0.015  # area of individual fin
# finNumber = 3
# Cl_max_fins = 1.2  # maximum lift coefficient of the fins, assumed at 45 degrees
# Cl_max_canards = 1.2  # maximum lift coefficient of the canards, assumed at 45 degrees

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


def getCoefficients(mach, alpha, Re, filename):
    # Open the file for reading
    with open(filename, "r") as file:
        # first column is mach number so interpolate to find the relevant rows
        lines = file.readlines()
        machs = []
        for line in lines:
            columns = line.split(",")
            machs.append(float(columns[0]))
        # now find the two mach numbers we need (immidiately below and above the current mach number)
        for i in range(len(machs)):
            if machs[i] > mach:
                break
        if i == 0:
            i = 1
        mach1 = machs[i - 1]
        mach2 = machs[i]
        # now search through one of these mach numbers and find the alpha values above and below
        file.seek(0)
        lines = file.readlines()
        alphas = []
        for line in lines:
            columns = line.split(",")
            alphas.append(float(columns[1]))
        for j in range(len(alphas)):
            if alphas[j] > alpha:
                break
        if j == 0:
            j = 1
        alpha1 = alphas[j - 1]
        alpha2 = alphas[j]
        # now do the same for log(Re)
        file.seek(0)
        lines = file.readlines()
        Res = []
        for line in lines:
            columns = line.split(",")
            Res.append(float(columns[2]))
        for k in range(len(Res)):
            if Res[k] > np.log10(Re):
                break
        if k == 0:
            k = 1
        Re1 = Res[k - 1]
        Re2 = Res[k]
        # now we have the 8 values we need to interpolate between
        file.seek(0)
        lines = file.readlines()
        # create a dictionary to store the data
        data = {}
        for line in lines:
            columns = line.split(",")
            key = (float(columns[0]), float(columns[1]), float(columns[2]))
            data[key] = [float(x) for x in columns[3:]]

        keys = [
            (mach1, alpha1, Re1),
            (mach1, alpha1, Re2),
            (mach1, alpha2, Re1),
            (mach1, alpha2, Re2),
            (mach2, alpha1, Re1),
            (mach2, alpha1, Re2),
            (mach2, alpha2, Re1),
            (mach2, alpha2, Re2),
        ]
        values = [data[key] for key in keys]
        # reform the values into a 2x2x2 array
        values = [[[values[0], values[1]], [values[2], values[3]]], [[values[4], values[5]], [values[6], values[7]]]]

        # get 'out' using RegularGridInterpolator
        interp = RegularGridInterpolator(([mach1, mach2], [alpha1, alpha2], [Re1, Re2]), np.array(values))
        out = interp((mach, alpha, np.log10(Re)))
        return out


# OpenRocket CD as this one isnt done yet
def openRocketCD(mach):
    # get Mach (first column) and Cd (second column) from the csv file
    with open("dragCurve.csv", "r") as file:
        # extract the values from the file
        lines = file.readlines()
        machs = []
        Cds = []
        for line in lines:
            columns = line.split(",")
            machs.append(float(columns[0]))
            Cds.append(float(columns[1]))
        # now interpolate to find the Cd at the current mach number
        Cd = np.interp(mach, machs, Cds)
        return Cd


def getCanardCoeffs(alpha):
    # get the coefficients of the canards from the NACA0012.csv file
    with open("NACA0012.csv", "r") as file:
        # extract the values from the file
        lines = file.readlines()
        alphas = []
        CLs = []
        CDs = []
        for line in lines:
            columns = line.split(",")
            alphas.append(float(columns[0]))
            CLs.append(float(columns[1]))
            CDs.append(float(columns[2]))
        # now interpolate to find the Cd at the current mach number
        CL = np.interp(alpha * 180 / np.pi, alphas, CLs)
        CD = np.interp(alpha * 180 / np.pi, alphas, CDs)
        return CL, CD


# main functions
def recalculate(t, state, dt, turb, env, Rocket, Simulation, thrust_data):
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

    # get a load of stuff from the rocket object
    m_dry = Rocket.dryMass
    m_wet = m_dry + Rocket.propMass
    length = Rocket.noseLength + Rocket.bodyLength + Rocket.boattailLength
    Rt_dry = length - Rocket.dryCG
    R_tank = length - Rocket.propCG
    Ilr_dry = Rocket.dryInertia
    Ilr_fuel = Rocket.propInertia
    topA = np.pi * Rocket.rocketDiameter**2 / 4
    sideA = np.pi * Rocket.rocketDiameter * length
    Isp = Rocket.Isp

    # also get the releveant simulation parameters
    launchRailLength = Simulation.launchRailLength
    launchRailAngle = Simulation.launchRailAngle

    # centre of mass moves - calculate it. Can be adjusted
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

    # get the coefficients
    Re = env.density * np.linalg.norm(dr) * length / env.mu
    if Re < 1e5:
        Re = 1e5  # allows calculation of coefficients at low Re, this is just an approximation
    mach = np.linalg.norm(dr) / np.sqrt(1.4 * 287 * env.temperature)
    if np.linalg.norm(dr) == 0:
        alpha = 0  # fixes singularity
    else:
        alpha = np.arccos(np.dot(dr, direction) / np.linalg.norm(dr))
    Cn, Cm, xcp, Mq, Cd = getCoefficients(mach, alpha, Re, "aeroParams.csv")
    #Cd = openRocketCD(mach)
    Cd = Cd + 0.36 # this is how much the calculation is off by for now
    # print("Time: ", t, "Alt. :", r[2], "Mach: ", mach, "Alpha: ", alpha, "C_d: ", Cd)

    # Drag
    # wind
    # get the index we need in the turbulence data
    i = int(r[2] / (env.rocketVelocity * env.deltaTime))
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
    rho = env.density
    D_translate = -0.5 * rho * Cd * topA * np.linalg.norm(dr_wind) * dr_wind
    D_rotate = np.array([0, 0, 0])
    D = D_translate + D_rotate
    on_rail = dr[2] > 0 and r[2] < launchRailLength * np.cos(launchRailAngle)
    if on_rail:
        D += 0.1 * D / np.linalg.norm(D)  # add rail friction
    # Lift - this is where the canards come in (assuming their drag contribution is negligible for now)
    # first calculate the effective area of the canards:
    # clamp roll to between -pi/6 and pi/6 due to symmetry
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))  # used to calculate the canard area

    # this can all go
    canardRoll = min(max(roll, -np.pi / 6), np.pi / 6)
    # get the effective area of the canards
    canardArea1 = np.cos(canardRoll + np.pi / 6) * Rocket.canardArea
    canardArea2 = np.cos(canardRoll - np.pi / 6) * Rocket.canardArea
    # get effective angle of attack for the canards
    canardAlpha1 = alpha * np.cos(canardRoll + np.pi / 6)
    canardAlpha2 = alpha * np.cos(canardRoll - np.pi / 6)
    # use NACA0012.csv to get the lift coefficient of the canards (file is in degrees)
    canardCL1, canardCD1 = getCanardCoeffs(canardAlpha1)
    canardCL2, canardCD2 = getCanardCoeffs(canardAlpha2)
    # get the lmagnitude of the lift force
    magnitudeL = 0.5 * rho * np.linalg.norm(dr_wind) ** 2 * (canardArea1 * canardCL1 + canardArea2 * canardCL2)
    # get the velocity vector's component perpendicular to the direction vector
    dr_perp = np.dot(dr_wind, direction) * direction / np.linalg.norm(direction) - dr_wind
    # this is the direction of the lift force, so normalise it and multiply by the magnitude
    L = magnitudeL * dr_perp / np.linalg.norm(dr_perp)
    D_canard = -0.5 * rho * np.linalg.norm(dr_wind) * (canardArea1 * canardCD1 + canardArea2 * canardCD2) * dr_wind


    # Gravity
    G = np.array([0, 0, -m * env.g])
    # Drag
    D = D_translate + D_rotate + D_canard # also change this when canards are gone

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
    )
    # centre of pressure moves too - but this is more complicated
    Rpt = length - xcp  # assuming the thrust comes out of the bottom of the rocket
    Rp = Rt - Rpt  # distance from center of mass to pressure
    Rc = xcp - Rocket.canardPos  # distance from center of pressure to canards
    M_forces = (
        np.cross(T, Rt * direction) + np.cross(D, Rp * direction) + np.cross(L, Rc * direction)
    )  # moment - thrust contribution is currently zero but this supports TVC

    # Damping stuff
    # get the pitch rate
    M_damping_pitch = np.array([0, 0, 0])  # TODO: do this properly
    M = M_forces + M_damping_pitch

    # derivative of the velocity vector can now be calculated
    diff_dr = 1 / m * (T + D + L + G)  # acceleration

    diff_w = np.linalg.inv(I).dot(M - (np.cross(w, I.dot(w))))  # angular acceleration

    # derivative of the mass
    diff_m = np.array([-(interpolate_thrust(t, thrust_data) / (env.g * Isp))])

    if on_rail:  # if the rocket is on the launch rail
        diff_w = np.array([0, 0, 0])  # prevent rotation
        diff_q = np.array([0, 0, 0, 0])  # prevent rotation
        diff_dr = (
            np.dot(direction, diff_dr) * direction / np.linalg.norm(direction)
        )  # prevent movement not in the direction of the rail

    # calculate pitch, roll and yaw to put in tracked values
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    newState = np.concatenate((diff_r, diff_dr, diff_q, diff_w, diff_m), axis=0)
    trackedValues = [diff_dr, roll, pitch, yaw, M, D, alpha, Rt, Rp, mach, T, L, I, wind, direction, t]
    return newState, trackedValues


# using my own solver so it goes in order of increasing time


def RK4(
    state, t, dt, turb, env, Rocket, Simulation, thrust_data
):  # other met  hods can be used but this is a good start
    """This function uses the 4th order Runge-Kutta method to solve the ODEs for the next time step."""
    # calculate the derivatives of the state vector
    k1, trackedValues = recalculate(
        t, state, dt, turb, env, Rocket, Simulation, thrust_data
    )  # dt as the 3rd argument is only used for calculating rates, not part of the RK4 method
    k2, _ = recalculate(t + 0.5 * dt, state + 0.5 * dt * k1, dt, turb, env, Rocket, Simulation, thrust_data)
    k3, _ = recalculate(t + 0.5 * dt, state + 0.5 * dt * k2, dt, turb, env, Rocket, Simulation, thrust_data)
    k4, _ = recalculate(t + dt, state + dt * k3, dt, turb, env, Rocket, Simulation, thrust_data)

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


def simulate(Rocket, Simulation, env, engineFile):
    """This function runs the simulation."""

    m_wet = Rocket.dryMass + Rocket.propMass
    launchRailAngle = Simulation.launchRailAngle
    launchRailDirection = Simulation.launchRailDirection
    t_end = Simulation.endTime
    dt = Simulation.timeStep
    t = Simulation.startTime

    thrust_data = read_thrust_curve(engineFile, m_wet * env.g)

    env = Environment()
    env.getForecast()  # this must be done before the simulation starts - could do it in __init__
    turb = env.getTurbulence(10000, t_end + 1)

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

    # using RK4
    while t < t_end and state[5] >= -10:
        state = RK4(state, t, dt, turb, env, Rocket, Simulation, thrust_data)
        print("Time:", round(t, 1), "seconds", end="\r")
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

    # print generic flight info
    print("\nFlight info:")
    print("Apogee: {:.0f} m".format(max([r[i][2] for i in range(len(r))])))
    print(
        "Max speed: {:.0f} m/s".format(max([np.linalg.norm([dr[i][0], dr[i][1], dr[i][2]]) for i in range(len(dr))]))
    )
    print("Max acceleration: {:.0f} m/s^2".format(max([np.linalg.norm(tracked_dict[t][0]) for t in tracked_dict])))
    print("Max Mach number: {:.2f}".format(max([tracked_dict[t][9] for t in tracked_dict])))

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
    axs[1, 2].plot(t, yawRate, label="yaw rate", alpha=0.5)
    axs[1, 2].set_xlabel("time (s)")
    axs[1, 2].set_ylabel("angular rates (rad/s)")
    axs[1, 2].legend()
    axs[1, 2].grid(visible=True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
