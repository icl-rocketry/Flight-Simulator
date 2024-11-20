# this file defines the simulation parameters (like launch rail stuff etc)
# it doesnt actually run the simulation, that is done in sixdof.py

class Simulator:
    def __init__(self, launchRailLength, launchRailAngle, launchRailDirection, windSpeed, windDirection, timeStep, startTime, endTime):
        self.launchRailLength = launchRailLength
        self.launchRailAngle = launchRailAngle
        self.launchRailDirection = launchRailDirection
        self.windSpeed = windSpeed
        self.windDirection = windDirection
        self.timeStep = timeStep
        self.startTime = startTime
        self.endTime = endTime