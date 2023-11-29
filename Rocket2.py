import numpy as np
from AeroSurfaces import NoseCone, BodyTube, BoatTail, Fins

class Rocket: 
    def __init__(self, rocketLength, rocketRadius):
        """
        ----------------------------------------------------------------------
        rocketLength: Total length of rocket (m)
        rocketRadius: Radius of rocket (m)
        ----------------------------------------------------------------------
        rocketCL: Lift Coefficient of rocket ()
        rocketCD: Drag coefficinet of rocket ()
        ----------------------------------------------------------------------
        rocketCGPos: Position of CG on rocket @ alpha = 0 (m)
        rocketCPPos: Position of CP on rocket @ alpha = 0 (m)
        staticMargin: Static Margin of rocket (calibers)
        ----------------------------------------------------------------------
        surfaceCN []: Array of Normal Coefficients of each surface
        surfaceCNPos []: Array of Normal Coefficients of each surface
        surfaceMass []: Array of Normal Coefficients of each surface
        surfaceCG []: Array of Normal Coefficients of each surface
        ----------------------------------------------------------------------
        """

        # Physical properties of rocket
        self.rocketLength = rocketLength # Rocket length is for entirety of rocket
        self.rocketRadius = rocketRadius

        # Aerodynamic Coefficients
        self.rocketCL = 0
        self.rocketCD = 0

        # Aerodynamic Stability Analysis
        self.rocketCN = 0
        self.rocketCGPos = 0 
        self.rocketCPPos = 0 
        self.staticMargin = 0 

        # Initialise list that stores aerodynamic coefficients for stability analysis
        self.surfaceCN = []
        self.surfaceCNPos = []
        self.surfaceMass = []
        self.surfaceCG = []

        # Booleans to check if one surface has been inputted?
        self.noseAdded = False
        self.bodyTubeAdded = False
        self.boatTailAdded = False

        # Checkers to check whether input is valid
        isTooLong = 0 # Variable that computes total length of components

        return None

    def addSurface(self, cp, cpPos, mass, cgPos):
        """
        Appends cp, cpPos, mass and cgPos of each surface into respective arrays
        """
        try:
            self.surfaceCN.append(cp)
            self.surfaceCNPos.append(cpPos)
            self.surfaceMass.append(mass) # Assume constant g for now, we can write a function of cg with gravity
            self.surfaceCG.append(cgPos)

        except TypeError:
            self.surfaceCN.append(cp)
            self.surfaceCNPos.append(cpPos)
            self.surfaceMass.append(mass) # Assume constant g for now, we can write a function of cg with gravity
            self.surfaceCG.append(cgPos)

        # Re-evalute static margin with newly added aero surface
        self.evaluateStaticMargin() 

        return None
    
    def evaluateStaticMargin(self):
        """
        Stability analysis by first evaluating the overall position of CG and CP, then evaluate static margin
        """
        # For each cp (which is calculated within the component class and position argument for each component, calcalte total cp and its final positon)
        # Maybe the same for cg?

        #Use Extended Barrowman here to evaluate CP
        self.evaluateRocketCP()
        self.evaluateRocketCG()

        self.staticMargin = (self.rocketCPPos - self.rocketCGPos)/(2*self.rocketRadius)

        return None
    
    def evaluateRocketCP(self):
        """
        Evaluates centre of pressure of rocket
        """

        cpTop = 0 # Initialise cpTop variable

        try:
            for coeff, pos in zip(self.surfaceCG,self.surfaceCNPos):
                cpTop += coeff*pos

        except TypeError:
            for coeff, pos in zip(self.surfaceCN,self.surfaceCNPos):
                cpTop += coeff*pos

        self.rocketCN = sum(self.surfaceCN)

        # In the odd case where there are no existing aero surfaces initialsied on rocket...
        if self.rocketCN == None:
            self.rocketCPPos = 0
        else:
            self.rocketCPPos = cpTop/self.rocketCN


    def evaluateRocketCG(self):
        """
        Evaluates centre of gravity of rocket
        """

        cgTop = 0

        try:
            for m, pos in zip(self.surfaceMass,self.surfaceCG):
                cgTop += m*9.81*pos

        except TypeError:
            for coeff, pos in zip(self.surfaceMass,self.surfaceCG):
                cgTop += m*9.81*pos
        
        self.rocketCGPos = cgTop/sum(self.surfaceMass)

    def evaluateRocketCL(self):
        pass

    def evaluateRocketCD(self):
        pass


    ### ADD AERODYNAMIC SURFACES ###

    def addNose(self, type, length, noseRadius, material, thickness,mass=0):
        """
        Adds nose cone to rocket
        """

        nose = NoseCone(type, length, noseRadius, self.rocketRadius, material, thickness,mass) # Pass parameters into NoseCone Class
        nose.add() # Add nose cone to rocket
        self.addSurface(nose.cn, nose.cnPos, nose.mass, nose.cgPos) # Add nose cone into rocket, position = 0 as nose is forced to be put at the top
        self.noseAdded = True

        #self.checkTotalLength(length)

        return nose
    

    def addBodyTube(self, length, radius, thickness, material,mass=0):
        """
        Adds body tube to rocket
        """

        if self.noseAdded == True:
            bodyTube = BodyTube(length, radius, thickness, material,mass) # Pass parameters into BodyTube Class
            bodyTube.add() # Add body tube to rocket
            self.addSurface(bodyTube.cn, bodyTube.cnPos, bodyTube.mass, bodyTube.cgPos) # Technically contributes to nothing
            self.bodyTubeAdded = True
        else:
            raise Exception("Nose Cone Not Added!")

        #self.checkTotalLength(length)

        return bodyTube


    def addBoatTail(self, upperRadius, lowerRadius, length, thickness, boatTailPos, material, mass=0):
        """
        Adds boat tail to rocket
        """

        if boatTailPos < 0:
            ValueError("BROOOOO") #Force thing to be placed in correct position

        boatTail = BoatTail(upperRadius, lowerRadius, length, self.rocketRadius, thickness, boatTailPos, material, mass)
        boatTail.add() # Initialise design parameters associated with boat tail
        self.addSurface(boatTail.cn,boatTail.cnPos, boatTail.mass, boatTail.cgPos)

        #self.checkTotalLength(length)

        return boatTail


    def addFins():
        """
        Adds fins to rocket
        """

        fins = Fins()
        
        return fins


    ### OTHER FUNKY FUNCTIONS ###

    def clear(self):
        """
        The odd case where the user wants to delete all surfaces
        """
        # Clear all aerodynamic surfaces on rocket, could potentially be used for housekeeping
        self.aerodynamicSurfaces.clear()

        # Reset all centres
        self.rocketCGPos = 0 
        self.rocketCPPos = 0 

        return None


    ### Data Validation ###

    def checkTotalLength(self, length):
        self.isTooLong += length
        if (self.isTooLong < self.rocketLength):
            pass
        else:
            raise Exception("Physically not possible")
