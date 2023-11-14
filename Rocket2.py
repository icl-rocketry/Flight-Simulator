import numpy as np
from AeroSurfaces import NoseCone, BoatTail

class Rocket: 
    def __init__(self, rocketLength, rocketRadius):
        self.rocketLength = rocketLength # Say rocket is the body tube for now...
        self.rocketRadius = rocketRadius
        self.cpPosition = 0 # Position of CP
        self.cgPosition = 0 # Position of CG

        # Empty tuple that stores all components
        self.aerodynamicSurfaces = []

        # Evaluate static margin without aero surfaces
        self.evaluateStaticMargin() 


        self.drawPoint = 0 # Reference point for drawing to be done later
        return None

    def addSurfaces(self, surfaces, positions):
        """
        Adds aero surfaces to rocket. However, the surface must be initialised first by using the commands below

        surfaces [list]
        position [list, noseToTail coordinate system]
        """
        try:
            for surface, position in zip(surfaces, positions):
                self.aerodynamicSurfaces.append(surface, position)
        except TypeError:
            self.aerodynamicSurfaces.append(surfaces, positions)

        # Re-evalute static margin with nose cone
        self.evaluateStaticMargin() 

        return None
    
    def evaluateStaticMargin():
        # For each cp (which is calculated within the component class and position argument for each component, calcalte total cp and its final positon)
        # Maybe the same for cg?

        #Use Extended Barrowman here to evaluate CP

        pass
    
    def evaluateCG():
        pass

    def addNose(self, type, length, radius, material):
        nose = NoseCone(self, type, length, radius, self.rocketRadius, material) # Set parameters for nose
        self.addSurfaces(nose, position=0) # Add nose cone into rocket, position = 0 as nose is forced to be put at the top
        return nose
    
    def addBoatTail(self,upperRadius,lowerRadius,pos):
        if pos < 0:
            ValueError("BROOOOO") #Force thing to be placed in correct position

        boatTail = BoatTail(self,upperRadius,lowerRadius,)
        self.addSurfaces(boatTail,position=pos)
        return boatTail

    def addFins():
        pass