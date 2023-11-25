import numpy as np
from AeroSurfaces import NoseCone, BoatTail, Fins
import Materials

class Rocket: 
    def __init__(self, rocketLength, rocketRadius):
        """
        rocketLength: 
        rocketRadius
        rocketCG: position of CG
        rocketCP: position of CP
        """

        # Physical properties of rocket
        self.rocketLength = rocketLength # Say rocket is the body tube for now... (on second thought, nahhhhhhhh)
        self.rocketRadius = rocketRadius

        self.rocketCP = 0
        self.rocketCG = 0

        self.rocketCGPos = 0 # Position of CG
        self.rocketCPPos = 0 # Position of CP
        self.staticMargin = 0 # Static Margin

        # Initialise list that stores aerodynamic coefficients for stability analysis
        self.surfaceCN = []
        self.surfaceCPPos = []

        return None

    def addSurface(self, cp, cpPos):

        try:
            self.surfaceCN.append(cp)
            self.surfaceCPPos.append(cpPos)

        except TypeError:
            self.surfaceCN.append(cp)
            self.surfaceCPPos.append(cpPos)

        # Re-evalute static margin with nose cone
        self.evaluateStaticMargin() 

        return None
    
    def evaluateStaticMargin(self):
        """
        This shit not working
        """
        # For each cp (which is calculated within the component class and position argument for each component, calcalte total cp and its final positon)
        # Maybe the same for cg?

        #Use Extended Barrowman here to evaluate CP
        self.evaluateRocketCP()
        self.evaluateRocketCG()

        self.staticMargin = (self.rocketCPPos - self.rocketCG)/(2*self.rocketRadius) # in calibers

        return None
    
    def evaluateRocketCP(self):
        try:
            for coeff, pos in zip(self.surfaceCN,self.surfaceCPPos):
                cpTop += coeff*pos

        except TypeError:
            for coeff, pos in zip(self.surfaceCN,self.surfaceCPPos):
                cpTop += coeff*pos

        self.rocketCN = sum(self.surfaceCPPos)
        self.rocketCPPos = cpTop/self.rocketCP


    def evaluateRocketCG(self):
        pass


    # ADD AERODYNAMIC SURFACES #

    def addNose(self, type, length, noseRadius, material):
        """
        Adds nose cone to rocket

        """
        nose = NoseCone(type, length, noseRadius, self.rocketRadius, material) # Set parameters for nose
        nose.add() # Initialise design parameters associated with nose
        self.addSurface(nose.cn, nose.cnPos) # Add nose cone into rocket, position = 0 as nose is forced to be put at the top
        return nose
    
    def addBodyTube(self):
        pass

    def addBoatTail(self, upperRadius, lowerRadius, length, radius, pos):
        if pos < 0:
            ValueError("BROOOOO") #Force thing to be placed in correct position

        boatTail = BoatTail(upperRadius, lowerRadius, length, radius, pos)
        boatTail.add() # Initialise design parameters associated with boat tail
        #self.addSurface(boatTail.cn,boatTail.cnPos)
        return boatTail

    def addFins():

        pass

    def clear(self):
        # Clear all aerodynamic surfaces on rocket, could potentially be used for housekeeping
        self.aerodynamicSurfaces.clear()
