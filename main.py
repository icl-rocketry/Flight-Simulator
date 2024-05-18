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
        0.6,  # noseLength
        3.5,  # bodyLength
        0.4,  # boattailLength
        0.8,  # boattailDRatio
        0.2,  # rocketDiameter
        12,  # finSweep
        0.22,  # finRoot
        0.08,  # finTip
        0.5,  # finSpan
        0.15,  # finGap
        3.8,  # finPos
        2.7,  # dryCG from nose
        4,  # propCG from nose
        44.796,  # dryMass
        54.126 - 44.796,  # propMass
        [[59.2, 0, 0],
         [0, 0.226, 0],
         [0, 0, 0.226]],  # dryInertia
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],  # propInertia
        190,  # Isp
    )
    """The table is created using the getAeroParams function and the results are stored in a csv file"""
    alphaList = [0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 70, 90]
    with open("aeroParams.csv", "w") as f:
        for M in arange(0, 2.05, 0.05):
            # print the mach number considered to 2 decimal places
            print(f"Mach = {M:.2f}", end="\r")
            for alpha in alphaList:
                for logR in arange(5, 9.5, 0.5):
                    Cn, Cm, xcp, Mq, Cd = AeroCalculator.getAeroParams(M, alpha, logR, Nimbus)
                    f.write(f"{M},{alpha},{logR},{Cn},{Cm},{xcp},{Mq},{Cd}\n")
                    # TODO: Mq doesnt work at M=1, Cd doesnt work at M=0
    # Initialise simulation
    Simulation = sim.Simulator(
        5,  # launchRailLength
        5*pi/180,  # launchRailAngle (rad)
        45*pi/180,  # launchRailDirection (rad)
        0,  # windSpeed
        0,  # windDirection
        0.02,  # timeStep
        0,  # startTime
        100,  # endTime
    )
    # Initialise environment
    env = sim.Environment() # TODO: allow parameters to go in here, incuding moving windSpeed and windDirection
    # Run simulation
    sim.simulate(Nimbus, Simulation, env, "NimbusThrustCurve.eng")
