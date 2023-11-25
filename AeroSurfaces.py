# General Imports, method is same as RocketPy coz it is neat code
import numpy as np
from function import Function

# NOSE CONE #
class NoseCone():
    def __init__(self, coneType, length, noseRadius, rocketRadius, material):
        """
        type: type of nose cone
        noseLength: length of nose cone (m)
        """   

        # Nose Cone Design Parameters
        self.type = coneType
        self.noseLength = length
        self.noseRadius = noseRadius
        self.rocketRadius = rocketRadius

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.k = None # Cp Factor
        self.k2 = None # Cp Position Factor, check cambridge doc for this (end of section 3.2.1)
        self.cn = None
        self.cl = None
        self.cd = None

        self.cnPos = None

    def add(self):
    # Note the position of the nose is @ top of rocket coordinate system

        if self.type.lower() == "vonkarman":    
            self.k = 0.5
            self.k2 = 0.6
        elif self.type.lower() == "haack":
            self.k = 0.437
            self.k2 = 0.6
        else:
            raise ValueError("Cannot find nose type?")
        
        self.geometricalParameters() # Evaluate geometrical parameters of nose
        
        self.evaluateCN()
        self.evaluateCL()
        
        # Drawing for later broooooooooooooooooo

    def geometricalParameters(self): # Finding rho, the radius ratio of nose cone
        if self.noseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.noseRadius/self.rocketRadius
        
    def evaluateCN(self):
        self.cn = Function(lambda alpha: 2*alpha) # Note cnAlpha = 2 for all nose cones because I said so
        self.cnPos = self.k * self.noseLength
    
    def evaluateCL(self):
        # Calculate clalpha
        # clalpha is currently a constant, meaning it is independent of Mach
        # number. This is only valid for subsonic speeds.
        # It must be set as a Function because it will be called and treated
        # as a function of mach in the simulation.

        self.clAlpha = Function(
            lambda mach: 2 * self.radius_ratio**2
            )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha
            )
        
        return None

#     def evaluateCD(self):

# BODY TUBE #
class bodyTube():
    def __init__(self, bodyTubeLength, bodyTubeRadius):
        self.bodyTubeLength = bodyTubeLength
        self.bodyTubeRadius = bodyTubeRadius

        self.cn = None
        self.cnPos = None

    def add(self):
        pass


# BOAT TAIL #
class BoatTail():
    def __init__(self, upperRadius, lowerRadius, length, rocketRadius, boatTailPos):

        # Initial Arguments
        self.startRadius = upperRadius
        self.endRadius = lowerRadius # Assume big to small boat tail, how else are you going to do it?
        self.rocketRadius = rocketRadius
        self.length = length
        self.boatTailPos = boatTailPos
        
        # Aerodynamic parameters
        self.cn = None
        self.cl = None
        self.cd = None

        self.cnPos = None

    
    def add(self):

        # Evaluate Cn and its position
        self.evaluateCN()



    def evaluateCN(self):
        # Sanity check, a boat tail should have a negative CP, hence pushing the total CP forward
        # No CP solution found yet, so for now just find CN and then ue barrowman to find overall cp position, isnt this the usual way?
        # Note only trapezoidal/linear boattail for now

        dRatio = self.rocketRadius/self.endRadius

        self.cn = 2*((self.endRadius/self.rocketRadius)**2 - (self.startRadius/self.rocketRadius)**2)
        self.cnPos = self.boatTailPos + self.length/3 * (1+(1-dRatio)/(1-dRatio**2))



    def drawBoatTail(self):
        n = 127 # number of points used to plot boat tail
        ymax = self.startRadius / 2 
        ymin = self.endRadius / 2
        m = (ymax-ymin)/ self.length    # gradient of line

        return [xBTPoints,yBTPoints]
    

# FINS # 
class Fins():
    def __init__(self, finType, n, finSpan, finRootChord, finMidChord, finTipChord, rocketRadius, pos):
        # [IMPORTANT] See how there are geometric parameters defined for trapezoidal fins ONLY, this should be moved to a separate function in the near future
        # i.e. addTrapezoidalFins(self, finSpanm finRootChord...) instead of initialising these geometric parameters in the fins class itself 
        """
        ----------------------------------------------------------------------
        numberOfFins: number of fins ()
        finType: geometry of fins ()
        finSpan: span of fins ()

        """
        # Physical parameters
        self.numberOfFins = n
        self.finType = finType
        self.finSpan = finSpan
        self.finRootChord = finRootChord
        self.finMidChord = finMidChord
        self.finTipChord = finTipChord

        self.rocketRadius = rocketRadius

        # Aerodynamic parameters
        self.kfb = None # Factor for interference between fin and body
        self.cn = None
        self.cpPos = None
        self.cl = None
        self.cd = None

        self.pos = pos


    def addFins(self):

        if self.finType.lower() == "trapezoidal":
            self.kfb = 1 + self.rocketRadius/(2*self.finSpan+self.rocketRadius)
        else:
            raise ValueError("Bruh no fins")
        
        self.evaluateCN()

    def evaluateCN(self):

        if NoseCone.noseRadius is not None:
            self.cnAlpha =  (self.kfb * 
                            (4*self.numberOfFins*(self.finSpan/NoseCone.noseRadius)**2)
                            /(1+np.sqrt(1+(2*self.finMidChord/(self.finRootChord+self.finTipChord))**2)
                            )
                                )
        else:
            raise ValueError("Nose cone not defined yet")
        
        self.cn = Function(lambda alpha: self.cnAlpha * alpha) #Assume small angles
        self.cpPos = (self.pos + 
                      self.finMidChord*(self.finRootChord+2*self.finTipChord)
                        /(3*(self.finRootChord+self.finTipChord))
                          +(self.finRootChord+self.finTipChord
                          -(self.finRootChord*self.finTipChord
                        /(self.finRootChord+self.finTipChord)))
                      /6
                    )


    def evaluateCD(self):
        self.cd = 0



