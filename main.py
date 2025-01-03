# import Atmosphere2 as a
from rocket3 import Rocket
from numpy import arange
import sixdof as sim
import AeroCalculator
from numpy import pi

# MAIN SCRIPT #
if __name__ == "__main__":
    # Initialise rocket
    # get the aero parameters
    Nimbus = Rocket(
        0,  # noseType
        0.35,  # noseLength
        3.93,  # bodyLength
        0.302,  # boattailLength
        0.783,  # boattailDRatio
        0.194,  # rocketDiameter
        16,  # finSweep
        0.28,  # finRoot
        0.13,  # finTip
        0.235,  # finSpan
        0.194,  # finGap
        3.93 - 0.28,  # finPos
        2.5,  # dryCG from nose
        3.93 + 0.35 - 1.0824,  # propCG from nose
        51.431,  # dryMass
        11.5,  # propMass
        [[58.1, 0, 0], [0, 58.1, 0], [0, 0, 0.231]],  # dryInertia
        [[0.605, 0, 0], [0, 0.6094, 0], [0, 0, 0.1004]],  # propInertia
        180,  # Isp
        0.0051,  # canardArea (per canard)
        1.21,  # canardPos
    )
    """The table is created using the getAeroParams function and the results are stored in a csv file"""
    alphaList = [0, 1, 2, 5, 10, 15, 20, 30, 40, 60, 90]
    with open("aeroParams.csv", "w") as f:
        for M in arange(0, 1.1, 0.1):
            # print the mach number considered to 2 decimal places
            print(f"Calculating the rocket's aerodyamic parameters at Mach {M:.1f}", end="\r")
            for alpha in alphaList:
                for logR in arange(5, 9.5, 0.5):
                    Cn, Cm, xcp, Mq, Cd, Cdp, Cdw, Cdwv, Cdbase, Cf = AeroCalculator.getAeroParams(
                        M, alpha, logR, Nimbus
                    )
                    f.write(f"{M},{alpha},{logR},{Cn},{Cm},{xcp},{Mq},{Cd}\n")
    # Initialise simulation
    print("\nInitialising simulation")
    Simulation = sim.Simulator(
        12,  # launchRailLength
        4 * pi / 180,  # launchRailAngle (rad)
        0,  # launchRailDirection (rad)
        0,  # windSpeed
        0,  # windDirection
        0.1,  # timeStep
        0,  # startTime
        100,  # endTime
    )
    # Initialise environment
    print("Initialising environment")
    env = sim.Environment()  # TODO: allow parameters to go in here, incuding moving windSpeed and windDirection
    # Run simulation
    # sim.simulate(Nimbus, Simulation, env, "NimbusThrustCurve.eng")

    # also plot Cd vs mach number for Mach 0 to 2 in 0.1 increments at alpha = 0 and logR = 8

    CdList = []
    CdpList = []
    CdwList = []
    CdwvList = []
    CdbaseList = []
    CfList = []
    machList = arange(0, 3.1, 0.01)
    for M in machList:
        Cn, Cm, xcp, Mq, Cd, Cdp, Cdw, Cdwv, Cdbase, Cf = AeroCalculator.getAeroParams(M, 0, 8, Nimbus)
        CdList.append(Cd)
        CdpList.append(Cdp)
        CdwList.append(Cdw)
        CdwvList.append(Cdwv)
        CdbaseList.append(Cdbase)
        CfList.append(Cf)

    import matplotlib.pyplot as plt

    plt.plot(machList, CdList)
    plt.plot(machList, CdpList)
    plt.plot(machList, CdwList)
    plt.plot(machList, CdwvList)
    plt.plot(machList, CdbaseList)
    plt.plot(machList, CfList)
    plt.xlabel("Mach number")
    plt.ylabel("Coefficient")
    plt.legend(["Cd", "Cdp", "Cdw", "Cdwv", "Cdbase", "Cdf"])
    plt.grid()
    plt.show()
