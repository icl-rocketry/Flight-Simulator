"""Currently the code is my old code - most of this can be changed to fit the new model"""

# big changes to make:
# - add a better geometry system (using OOP) --> use a nose cone, body, and fin set(s) to create a rocket
# - add a better aerodynamics system which uses the geometry system, with calculations for each 'part'
# - make the simulation itself more realistic (e.g. adding better wind)


# imports
import numpy as np
import quaternion
import matplotlib.pyplot as plt

# set default style
plt.style.use("seaborn-v0_8")


# plotting arrays
# create a dictionary to store the {time: state} pairs
state_dict = {}
tracked_dict = {}  # stores {time: tracked values} pairs


# other constants
m_dry = 9.74  # kg - still with the motor casing
m_wet = 12.155  # kg
Rt_dry = 0.795  # distance from center of mass to thrust vector (when dry)
Isp = 204  # s
Cd_cyl = 1.17  # for the drag resulting from the rotation
radius = 0.06
length = 2.51
R_tank = 0.195
windAngle = np.pi / 2  # heading of wind
windSpeed = 4  # m/s
gustAngle = np.pi / 3  # heading of gust (rad)
gustSpeed = 2  # m/s
gustFreq = 0.3  # Hz
launchRailLength = 9  # m
launchRailAngle = 5 * np.pi / 180  # rad
launchRailDirection = np.pi / 2  # rad
separationAngle = 0.3  # rad - angle of attack at flow separation
g0 = 9.81  # m/s^2
sideA = 0.2854
topA = 0.0114
Ilr_dry = np.array([[6.55, 0, 0], [0, 6.55, 0], [0, 0, 0.0219]])  # moment of inertia of the rocket when dry
Ilr_fuel = np.array([[0.0329, 0, 0], [0, 0.0329, 0], [0, 0, 0.0029]])  # moment of inertia of the fuel


# other functions
def pressure(h):
    """This function returns the pressure at a given altitude."""
    return 1.225 * ((288.16 - 0.0065 * h) / 288.16) ** 4.256  # kg/m^3, valid for 0-11km


def temperature(h):
    """This function returns the temperature at a given altitude."""
    return 288.16 - 0.0065 * h  # K, valid for 0-11km


def read_thrust_curve(filename):
    time_thrust_data = [(0, 0)]  # Add a point at t=0, F=0 to the beginning of the dat

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
                    time_thrust_data.append((time, thrust))  # Add the data point to the list
                except ValueError:
                    pass  # Ignore lines with invalid data

    return time_thrust_data


thrust_data = read_thrust_curve("Cesaroni_4842L610-P.eng")


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


def gravity(h):
    """This function returns the acceleration due to gravity at a given altitude."""
    return g0 * ((6371000 / (6371000 + h)) ** 2)


def CoP(mach):  # TODO: with the new sim this can be calculated in a less dodgy way
    """This function returns the distance from the thrust vector to the pressure vector."""
    CoP = length - (1.83 + mach / 20)
    if mach < 0.25:
        CoP = length - 1.8425 - 0.25 * (0.25 - mach)
    return CoP  # crude approximation from openrocket sims


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


def windy(t):
    """This function returns the wind velocity at a given time."""
    return windSpeed * np.array([np.cos(windAngle), np.sin(windAngle), 0]) + gustSpeed * np.array(
        [np.cos(gustAngle), np.sin(gustAngle), 0]
    ) * (0.5 + 0.5 * np.cos(2 * np.pi * t * gustFreq))


# main functions
def recalculate(state, t, dt, initialCall):
    """This function returns the derivatives of the state vector which is input. This can then be used to solve the ODEs for the next time step.''
    state = [r, dr, q, w, m] --> [dr, ddr, dq, dw, dm]
    r = position vector (3x1)
    dr = velocity vector (3x1)
    q = quaternion (4x1) - but the actual data is stored as a 4x1 array
    w = angular velocity vector (3x1)
    m = mass (scalar)
    initialCall is a boolean which is True if this is the first time the function is called for this time step, and used to find old values
    """

    # centre of mass moves - calculate it
    Rt = (Rt_dry * m_dry + R_tank * (state[13] - m_dry)) / state[
        13
    ]  # distance from center of mass to thrust vector at current time

    # Unpack the state vector
    r = state[0:3]
    dr = state[3:6]
    q = np.quaternion(state[6], state[7], state[8], state[9]).normalized()  # quaternion representing orientation
    w = state[10:13]  # angular velocity as a quaternion
    m = state[13]

    """Calculate the derivatives of the state vector"""

    # derivative of the position vector
    diff_r = dr

    # derivative of the velocity vector requires the forces acting on the rocket
    # Thrust
    direction = quaternion.as_rotation_matrix(q).dot(np.array([0, 0, 1]))  # direction of thrust
    direction = direction / np.linalg.norm(direction)  # normalise the direction vector
    T = interpolate_thrust(t, thrust_data) * direction
    # Drag
    # wind
    wind = windy(t)  # the wind velocity TODO: make this more realistic (gusts, direction, magnitude, etc.)
    dr_wind = dr - wind
    # cross-sectional area, where alpha is the angle between rocket direction and velocity vector
    if np.dot(dr, direction) == 0:
        alpha = 0
    else:
        alpha = np.arccos(np.dot(dr_wind, direction / np.linalg.norm(dr_wind)))
    mach = np.linalg.norm(dr_wind) / np.sqrt(1.4 * 287 * temperature(r[2]))  # mach number
    Cd = dragCoef(mach, alpha)  # drag coefficient
    rho = pressure(r[2])
    D_translate = -0.5 * pressure(r[2]) * Cd * topA * np.linalg.norm(dr_wind) * dr_wind
    D_rotate = (
        -np.cross(w, direction) * np.linalg.norm(w) * rho * Cd_cyl * radius / 6 * ((length - Rt) ** 3 - Rt**3)
    )  # drag due to rotation - TODO: add fins
    D = D_translate + D_rotate
    on_rail = dr[2] > 0 and r[2] < launchRailLength * np.cos(launchRailAngle)
    if on_rail:
        D += 0 * D / np.linalg.norm(D)  # add rail friction (zero for now)
    # Lift
    L = np.array([0, 0, 0])  # TODO: add lift, this probably isnt that hard, just use Cl and alpha
    # Gravity
    G = np.array([0, 0, -m * gravity(r[2])])
    # derivative of the velocity vector can now be calculated
    diff_dr = 1 / m * (T + D + L + G)

    # derivative of the quaternion
    w_quat = np.quaternion(0, w[0], w[1], w[2])  # angular velocity as a quaternion
    diff_q = quaternion.as_float_array(0.5 * w_quat * q)
    if on_rail:  # if the rocket is on the launch rail
        diff_q = np.array([0, 0, 0, 0])  # prevent rotation

    # Inertia tensor and moment
    I = (
        Ilr_dry
        + m_dry * np.array([[(Rt - Rt_dry) ** 2, 0, 0], [0, (Rt - Rt_dry) ** 2, 0], [0, 0, 0]])
        + (m - m_dry)
        / (m_wet - m_dry)
        * (Ilr_fuel + (m - m_dry) * np.array([[(Rt - R_tank) ** 2, 0, 0], [0, (Rt - R_tank) ** 2, 0], [0, 0, 0]]))
    )  # TODO: openrocket seems to disagree with this?
    # centre of pressure moves too - but this is more complicated
    Rpt = CoP(mach)  # distance from thrust location to pressure vector
    Rp = Rt - Rpt
    M_forces = np.cross(T, Rt * direction) + np.cross(
        D + L, Rp * direction
    )  # moment - thrust contribution is currently zero but this supports TVC
    M_rotation = (
        -rho * Cd * w * np.linalg.norm(w) * radius / 8 * length**4
    )  # moment due to rotational drag for a cylinder.
    M_rotation_fins = 0  # TODO: take fins into account
    M = M_forces + M_rotation + M_rotation_fins

    diff_w = np.linalg.inv(I).dot(M)
    if on_rail:  # if the rocket is on the launch rail
        diff_w = np.array([0, 0, 0])  # prevent rotation

    # derivative of the mass
    diff_m = np.array([-(interpolate_thrust(t, thrust_data) / (g0 * Isp))])

    # calculate pitch, roll and yaw to put in tracked values
    pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    if t > dt and initialCall:  # wrap around if there is a discontiuity
        old_pitch = tracked_dict[round(t - dt, 2)][1]
        if pitch - old_pitch > np.pi:
            pitch = pitch - 2 * np.pi
        elif pitch - old_pitch < -np.pi:
            pitch = pitch + 2 * np.pi

    roll = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    if t > dt and initialCall:  # wrap around if there is a discontiuity
        old_roll = tracked_dict[round(t - dt, 2)][2]
        if roll - old_roll > np.pi:
            roll = roll - 2 * np.pi
        elif roll - old_roll < -np.pi:
            roll = roll + 2 * np.pi

    yaw = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    if t > dt and initialCall:  # wrap around if there is a discontiuity
        old_yaw = tracked_dict[round(t - dt, 2)][3]
        if yaw - old_yaw > np.pi:  # there must have been a discontiuity
            yaw = yaw - 2 * np.pi
        elif yaw - old_yaw < -np.pi:
            yaw = yaw + 2 * np.pi

    # calculate pitch, roll and yaw rates to put in tracked values
    if t > dt and initialCall:
        rounded = round(t - dt, 2)
        pitch_rate = pitch - tracked_dict[rounded][1]
        roll_rate = roll - tracked_dict[rounded][2]
        yaw_rate = yaw - tracked_dict[rounded][3]
    else:
        pitch_rate = 0
        roll_rate = 0
        yaw_rate = 0

    newState = np.concatenate((diff_r, diff_dr, diff_q, diff_w, diff_m), axis=0)
    trackedValues = [
        diff_dr,
        pitch,
        roll,
        yaw,
        M,
        D,
        alpha,
        pitch_rate,
        roll_rate,
        yaw_rate,
        Rt,
        Rp,
        mach,
        T,
        L,
        I,
    ]  # if we want to plot things not in the state vector, we can add them to this list
    return newState, trackedValues


def RK4(state, dt, t):  # other methods can be used but this is a good start
    """This function uses the 4th order Runge-Kutta method to solve the ODEs for the next time step."""
    # calculate the derivatives of the state vector
    k1, trackedValues = recalculate(
        state, t, dt, True
    )  # dt as the 3rd argument is only used for calculating rates, not part of the RK4 method
    k2, _ = recalculate(state + 0.5 * dt * k1, t + 0.5 * dt, dt, False)
    k3, _ = recalculate(state + 0.5 * dt * k2, t + 0.5 * dt, dt, False)
    k4, _ = recalculate(state + dt * k3, t + dt, dt, False)

    # calculate the next state
    state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # store the state in the dictionary and append tracked values to the tracked list
    rounded = round(t, 2)
    state_dict[rounded] = state
    tracked_dict[rounded] = trackedValues
    return state


def main():
    """This function runs the simulation."""
    # Initial conditions
    r = np.array([0, 0, 1])
    dr = np.array([0, 0, 0])
    alpha = launchRailAngle  # angle of the rocket from the vertical (rad) TODO: add a direction too
    q_quat = quaternion.from_rotation_vector(
        np.array([alpha * np.cos(launchRailDirection), alpha * np.sin(launchRailDirection), 0])
    )
    q = quaternion.as_float_array(q_quat)
    w = np.array([0, 0, 0])
    m = np.array([m_wet])  # dry mass

    # instantiate the initial state vector
    state = np.concatenate((r, dr, q, w, m), axis=0)

    # Simulation parameters
    dt = 0.04  # time step
    t = 0  # initial time
    t_end = 1000  # end time

    # Simulation loop
    while t < t_end and state[2] > 0:
        state = RK4(state, dt, t)
        print(round(t, 2), "seconds", end="\r")
        t += dt

    # Plotting

    # create figure with 6 subplots
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    # plot x,y,z position all in one plot
    axs[0, 0].plot([t for t in state_dict], [state_dict[t][0] for t in state_dict], label="x")
    axs[0, 0].plot([t for t in state_dict], [state_dict[t][1] for t in state_dict], label="y")
    axs[0, 0].plot([t for t in state_dict], [state_dict[t][2] for t in state_dict], label="z")
    axs[0, 0].set_xlabel("time (s)")
    axs[0, 0].set_ylabel("position (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(visible=True)

    # plot x,y,z velocity all in one plot and total velocity magnitude
    axs[0, 1].plot([t for t in state_dict], [state_dict[t][3] for t in state_dict], label="x")
    axs[0, 1].plot([t for t in state_dict], [state_dict[t][4] for t in state_dict], label="y")
    axs[0, 1].plot([t for t in state_dict], [state_dict[t][5] for t in state_dict], label="z")
    axs[0, 1].plot(
        [t for t in state_dict],
        [np.linalg.norm(state_dict[t][3:6]) for t in state_dict],
        label="total",
        color="black",
        alpha=0.2,
    )
    axs[0, 1].set_xlabel("time (s)")
    axs[0, 1].set_ylabel("velocity (m/s)")
    axs[0, 1].legend()
    axs[0, 1].grid(visible=True)

    # plot x,y,z acceleration all in one plot and total acceleration magnitude
    axs[0, 2].plot([t for t in state_dict], [tracked_dict[t][0][0] for t in state_dict], label="x")
    axs[0, 2].plot([t for t in state_dict], [tracked_dict[t][0][1] for t in state_dict], label="y")
    axs[0, 2].plot([t for t in state_dict], [tracked_dict[t][0][2] for t in state_dict], label="z")
    axs[0, 2].plot(
        [t for t in state_dict],
        [np.linalg.norm(tracked_dict[t][0]) for t in state_dict],
        label="total",
        color="black",
        alpha=0.2,
    )
    axs[0, 2].set_xlabel("time (s)")
    axs[0, 2].set_ylabel("acceleration (m/s^2)")
    axs[0, 2].legend()
    axs[0, 2].grid(visible=True)

    # plot thrust magnitude, drag magnitude and lift magnitude all in one plot
    axs[0, 3].plot(
        [t for t in tracked_dict], [np.linalg.norm(tracked_dict[t][13]) for t in tracked_dict], label="thrust"
    )
    axs[0, 3].plot([t for t in tracked_dict], [np.linalg.norm(tracked_dict[t][5]) for t in tracked_dict], label="drag")
    axs[0, 3].plot(
        [t for t in tracked_dict], [np.linalg.norm(tracked_dict[t][14]) for t in tracked_dict], label="lift"
    )
    axs[0, 3].set_xlabel("time (s)")
    axs[0, 3].set_ylabel("force (N)")
    axs[0, 3].legend()
    axs[0, 3].grid(visible=True)

    # plot pitch, roll and yaw all in one plot and angle of attack
    axs[1, 0].plot([t for t in tracked_dict], [tracked_dict[t][1] for t in tracked_dict], label="pitch")
    axs[1, 0].plot([t for t in tracked_dict], [tracked_dict[t][2] for t in tracked_dict], label="roll")
    axs[1, 0].plot([t for t in tracked_dict], [tracked_dict[t][3] for t in tracked_dict], label="yaw")
    axs[1, 0].plot(
        [t for t in tracked_dict],
        [tracked_dict[t][6] for t in tracked_dict],
        label="angle of attack",
        color="black",
        alpha=0.2,
    )
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("angle (rad)")
    axs[1, 0].legend()
    axs[1, 0].grid(visible=True)

    # plot pitch, roll and yaw rates all in one plot
    axs[1, 1].plot([t for t in tracked_dict], [tracked_dict[t][7] for t in tracked_dict], label="pitch rate")
    axs[1, 1].plot([t for t in tracked_dict], [tracked_dict[t][8] for t in tracked_dict], label="roll rate")
    axs[1, 1].plot([t for t in tracked_dict], [tracked_dict[t][9] for t in tracked_dict], label="yaw rate")
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].set_ylabel("angular velocity (rad/s)")
    axs[1, 1].legend()
    axs[1, 1].grid(visible=True)

    # plot Centre of Pressure and Centre of Mass
    axs[1, 2].plot([t for t in tracked_dict], [tracked_dict[t][10] for t in tracked_dict], label="CoP")
    axs[1, 2].plot([t for t in tracked_dict], [tracked_dict[t][11] for t in tracked_dict], label="CoM")
    axs[1, 2].set_xlabel("time (s)")
    axs[1, 2].set_ylabel("distance from base (m)")
    axs[1, 2].legend()

    # plot inertia tensor Ixx then Izz on a twin y axis
    axs[1, 3].plot([t for t in tracked_dict], [tracked_dict[t][15][0][0] for t in tracked_dict], label="Ixx/Iyy")
    axs[1, 3].set_xlabel("time (s)")
    axs[1, 3].set_ylabel("Ixx (kg m^2)")
    axs[1, 3].grid(visible=True)
    axs2 = axs[1, 3].twinx()
    axs2.plot(
        [t for t in tracked_dict], [tracked_dict[t][15][2][2] for t in tracked_dict], label="Izz", color="orange"
    )
    axs2.set_ylabel("Izz (kg m^2)")
    axs2.legend()

    # plot the 3d trajectory
    # ax = fig.add_subplot(2, 3, 6, projection='3d')
    # ax.plot([state_dict[t][0] for t in state_dict],
    #         [state_dict[t][1] for t in state_dict],
    #         [state_dict[t][2] for t in state_dict])
    # ax.plot(np.zeros_like([state_dict[t][0] for t in state_dict]),
    #         [state_dict[t][1] for t in state_dict],
    #         [state_dict[t][2] for t in state_dict], "k--", alpha=0.2)
    # ax.plot([state_dict[t][0] for t in state_dict],
    #         np.zeros_like([state_dict[t][0] for t in state_dict]),
    #         [state_dict[t][2] for t in state_dict], "k--", alpha=0.2)
    # ax.plot([state_dict[t][0] for t in state_dict],
    #         [state_dict[t][1] for t in state_dict],
    #         np.zeros_like([state_dict[t][0] for t in state_dict]), "k--", alpha=0.2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    # animation! - TODO

    return True


if __name__ == "__main__":
    main()
