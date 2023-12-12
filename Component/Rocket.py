import numpy as np
from Component.AeroSurfaces import NoseCone, BodyTube, BoatTail, TrapezoidalFins
from Component.MoreComponents import MassComponent

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
        self.aerodynamicSurfacesList = []
        self.aerodynamicPositionsList = []

        # Booleans to check if one surface has been inputted?
        self.noseAdded = False
        self.bodyTubeAdded = False
        self.boatTailAdded = False
        self.finsAdded = False

        # Checkers to check whether input is valid
        self.isTooLong = 0 # Variable that computes total length of components
        
        self.nosePos = 0
        self.bodyTubePos = 0
        self.boatTailPos = 0
        self.finsPos = 0
        
        self.noseLength = 0
        self.bodyTubeLength = 0
        self.boatTailLength = 0
        self.finsLength = 0

        return None
 
    def addSurface(self, surfaces):
        """
        Updated add surface, note cpPos is relative to nose cone tip
        surfaces is a list also cp POs 
        """
        try:
            for surface in surfaces:
                self.aerodynamicSurfacesList.append(surface)
        except TypeError:
            self.aerodynamicSurfacesList.append(surfaces)
            
        # Re-evalute static margin with newly added aero surface
        self.evaluateRocketCP()
        self.evaluateRocketCG()
        self.staticMargin = self.evaluateStaticMargin() 

        return None
    
    def evaluateStaticMargin(self):
        """
        Stability analysis by first evaluating the overall position of CG and CP, then evaluate static margin
        """
        #Use Extended Barrowman here to evaluate CP
        return (self.rocketCPPos - self.rocketCGPos)/(2*self.rocketRadius)
    
    def evaluateRocketCP(self):
        """
        Evaluates centre of pressure of rocket
        """

        cpTop = 0 # Initialise cpTop variable

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                self.rocketCN += surface.cnAlpha

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
        cgBottom = 0

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                cgTop += surface.mass*9.81*surface.cgPos
                cgBottom += surface.mass
    
        self.rocketCGPos = cgTop/cgBottom
        


#---------------------------------------------------------------------- ADD AERODYNAMIC SURFACES ----------------------------------------------------------------------#

    def addNose(self, type, length, noseRadius, material, thickness,mass=0):
        """
        Adds nose cone to rocket
        """

        nose = NoseCone(type, length, noseRadius, self.rocketRadius, material, thickness, mass) # Pass parameters into NoseCone Class
        self.addSurface(nose) # Add nose cone into rocket, position = 0 as nose is forced to be put at the top
        
        self.noseAdded = True
        self.noseLength = length
        self.checkTotalLength(length)

        return nose
    

    def addBodyTube(self, length, radius, thickness, material, position, mass=0):
        """
        Adds body tube to rocket
        [FUTURE WORK] MAKE MULTIPLE BODY TUBES
        """

        if self.noseAdded == True:
            if position != self.noseLength:
                postition = self.noseLength
            bodyTube = BodyTube(length, radius, thickness, position, material,mass) # Pass parameters into BodyTube Class
            self.addSurface(bodyTube) # Technically contributes to nothing
            self.bodyTubeAdded = True
        else:
            raise Exception("Nose Cone Not Added!")
        
        self.bodyTubeLength = length
        self.checkTotalLength(length)

        return bodyTube


    def addBoatTail(self, upperRadius, lowerRadius, length, thickness, position, material, mass=0):
        """
        Adds boat tail to rocket
        """

        self.checkTotalLength(length)

        if self.bodyTubeAdded == True:
            if position != (self.noseLength + self.bodyTubeLength):
                position = self.noseLength + self.bodyTubeLength
            boatTail = BoatTail(upperRadius, lowerRadius, length, self.rocketRadius, thickness, position, material, mass)
            self.addSurface(boatTail)
        else:
            raise Exception("Body Tube Not Added!")

        return boatTail


    def addTrapezoidalFins(self,numberOfFins, finSpan, finRootChord, finMidChord, finTipChord, sweepLength, sweepAngle, rocketRadius, pos, mass=0):
        """
        Adds trapezoidal fins to rocket
        """

        trapFins = TrapezoidalFins(numberOfFins, finSpan, finRootChord, finMidChord, finTipChord, sweepLength, sweepAngle, rocketRadius, pos, mass)
        
        return trapFins


#---------------------------------------------------------------------- ADD OTHER COMPONENTS ----------------------------------------------------------------------#

    def addMassComponet(self,mass,pos):
        """
        Adds mass component (to shift cgPos)
        mass in kg and pos in m
        """
    # Check if position is valid
        if pos < 0:
            raise Exception("Position is invalid! Must be greater than 0")
        elif pos > self.rocketLength:
            raise Exception("Mass component is literally not on rocket")

        massComponent = MassComponent(mass,pos)
        self.addsurface(0,0,massComponent.mass*9.81, massComponent.pos)

        return massComponent

#------------------------------------------------------------------- AERODYNAMIC PARAMETERS -------------------------------------------------------------------#
# Do this in dynamics model 
# #for aero_surface, position in self.rocket.aerodynamic_surfaces: 
#   c_lift = aero_surface.cl(comp_attack_angle, comp_stream_mach)
#---------------------------------------------------------------------- FUNKY FUNCTIONS ----------------------------------------------------------------------#

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


#---------------------------------------------------------------------- DATA VALIDATION ----------------------------------------------------------------------#

    def checkTotalLength(self, length):
        self.isTooLong += length
        if (self.isTooLong < self.rocketLength):
            pass
        else:
            raise Exception("Physically not possible")
        