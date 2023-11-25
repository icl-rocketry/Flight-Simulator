import numpy as np
from AeroSurfaces import NoseCone, BodyTube, Boattail, Fins


class Rocket:
    def __init__(self, rocketLength, rocketRadius):
        self.rocketLength = rocketLength  # Say rocket is the body tube for now...
        self.rocketRadius = rocketRadius
        self.cpPosition = 0  # Position of CP

        self.aerodynamicSurfaces = []  # Empty tuple that stores all components

        # Empty tuple that stores all components

        # Evaluate static margin without aero surfaces
        self.evaluateStaticMargin()

        return None

    # def addSurfaces(self, surfaces, positions):
    #     """
    #     Adds aero surfaces to rocket. However, the surface must be initialised first by using the commands below

    #     surfaces [list]
    #     position [list, noseToTail coordinate system]
    #     """
    #     try:
    #         for surface, position in zip(surfaces, positions):
    #             self.aerodynamicSurfaces.append(surface, position)
    #     except TypeError:
    #         self.aerodynamicSurfaces.append(surfaces, positions)

    #     # Re-evalute static margin with nose cone
    #     self.evaluateStaticMargin()

    #     return None
    
    def evaluateCG(self):
        pass #this must be done before evaluating the static margin as the cg is required

    def evaluateStaticMargin(self):
        # For each cp (which is calculated within the component class and position argument for each component, calcalte total cp and its final positon)
        total = 0
        for surface in self.aerodynamicSurfaces:
            Cna = surface.Cna
            Xn = surface.Xn
            total += Cna * Xn
            totalCna += Cna
        self.cpPosition = total / totalCna
        self.staticMargin = (self.cpPosition - self.cgPosition) / (2 * self.rocketRadius)

        pass

    def addNose(self, coneType, length, radius, material):
        nose = NoseCone(self, coneType, length, radius, self.rocketRadius, material)  # Set parameters for nose
        self.addSurfaces(
            nose, position=0
        )  # Add nose cone into rocket, position = 0 as nose is forced to be put at the top
        self.aerodynamicSurfaces.append(nose)
        return nose

    def addBoattail(self, upperRadius, lowerRadius, length, topLocation):
        boattail = Boattail(self, upperRadius, lowerRadius, rocketRadius, length, topLocation)
        self.addSurfaces(boattail, position=0)
        self.aerodynamicSurfaces.append(boattail)
        return boattail

    def addFins(self):
        pass
