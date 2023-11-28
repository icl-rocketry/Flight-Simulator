# General Imports, method is same as RocketPy coz it is neat code
import numpy as np
import Materials as mat
from function import Function

# ADD MASS OVERRIDE OPTION 
# METHOD 1: ARGUMENT = 0 OR 1 FOR MASS MODE (MASSOVERRIDE = 1)
# METHOD 2: KEYWORD ARGUMENT (DEFAULT VALUE I.E. MASS=100, IF NO VALUE GIVEN THEN USE MASS FUNCTION)

# POSITION, FORCE FOR NOW, SO USE ISTOOLONG VARIABLE



# NOSE CONE #
class NoseCone():
    def __init__(self, coneType, length, noseRadius, rocketRadius, material, thickness):
        """
        ----------------------------------------------------------------------
        type: Type of nose cone (string)
        noseLength: Length of nose cone (m)
        noseRadius: Radius of nose cone (m)
        thickness: Thickness of nose cone (m)
        rocketRadius: Radius of rocket (m)
        noseMaterial: Material of nose cone (string)
        ----------------------------------------------------------------------
        """   

        # Nose Cone Design Parameters
        self.type = coneType
        self.noseLength = length
        self.noseRadius = noseRadius
        self.thickness = thickness
        self.rocketRadius = rocketRadius

        # Physical Parameters
        self.noseMaterial = material
        self.volume = None 
        self.mass = None

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.k = None # Cp Factor
        self.k2 = None # Cp Position Factor, check cambridge doc for this (end of section 3.2.1)
        self.cn = None
        self.cl = None
        self.cd = None

        self.cnPos = 0
        self.cgPos = 0

    def add(self):
    # Note the position of the nose is @ top of rocket coordinate system

        if self.type.lower() == "vonkarman":    
            self.k = 0.5
            self.volume = self.noseLength * np.pi/ 2 * (9/8) * (self.noseRadius**2-(self.noseRadius-self.thickness)**2)
        elif self.type.lower() == "haack":
            self.k = 0.437
            self.volume = (np.pi*self.noseLength/2) * ((self.noseRadius**2) - (self.noseRadius-self.thickness)**2)
        else:
            raise ValueError("Cannot find nose type?")
        
        self.geometricalParameters() # Evaluate geometrical parameters of nose
        
        self.evaluateCN()
        self.evaluateCL()
        
        # Delegate drawing to 1st year

    def geometricalParameters(self): # Finding rho, the radius ratio of nose cone
        if self.noseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.noseRadius/self.rocketRadius

        self.mass = mat.Materials(self.noseMaterial).density*1000*self.volume # Finding mass
        

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
            lambda alpha, mach: self.clAlpha(mach) * alpha
            )
        
        return None
    

    def evaluateCD(self):
        pass



# BODY TUBE #
class BodyTube():
    def __init__(self, bodyTubeLength, bodyTubeRadius, thickness, material):
        """
        ----------------------------------------------------------------------
        bodyTubeLength: Length of body tube (m)
        bodyTubeRadius: Radius of body tube (m)
        bodyTubeThickness: Thickness of body tube (m)
        ----------------------------------------------------------------------

        ----------------------------------------------------------------------
        """
        self.bodyTubeLength = bodyTubeLength
        self.bodyTubeRadius = bodyTubeRadius # Assume cylindrical tube
        self.bodyTubeThickness = thickness

        # Physical parameters of body tube
        self.bodyTubeMaterial = material
        self.volume = 0
        self.mass = 0

        # Aerodynamic parameters
        self.cn = 0 # Cn of body tube = 0 (someone verify this please)
        self.cnPos = 0
        self.cgPos = 0 # In the future, use Fusion API (have wet and dry CG)

    def add(self):
        self.evaluateMass()
        self.evaluateCN()

    def evaluateMass(self):
        self.volume = np.pi*(self.bodyTubeRadius**2-(self.bodyTubeRadius-self.bodyTubeThickness)**2)*self.bodyTubeLength
        self.mass = mat.Materials(self.bodyTubeMaterial).density*1000*self.volume

    def evaluateCN(self):
        self.cnPos = 0.5 * self.bodyTubeLength # note this is relative to the body tube, not the entire rocket, need to change this!

    def drawBodyTube(self):
        # FIRST YEAR JOB
        pass


# BOAT TAIL #
class BoatTail():
    def __init__(self, upperRadius, lowerRadius, length, rocketRadius, thickness, boatTailPos, material):

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

        self.cnPos = 0
        self.cgPos = 0

        # Physical parameters of boat tail
        self.boatTailMaterial = material
        self.boatTailThickness = thickness
        self.volume = None
        self.mass = None
    
    def add(self):
        self.evaluateCN() # Evaluate Cn and its position
        self.evaluateMass() # Evaluate mass of boat tail

    def evaluateCN(self):
        # Sanity check, a boat tail should have a negative CP, hence pushing the total CP forward
        dRatio = self.rocketRadius/self.endRadius

        self.cn = 2*((self.endRadius/self.rocketRadius)**2 - (self.startRadius/self.rocketRadius)**2) # Modelling boat tail as body tube transition
        self.cnPos = self.boatTailPos + self.length/3 * (1+(1-dRatio)/(1-dRatio**2))

    def evaluateMass(self):
        self.volume = np.pi*self.length/(3*self.startRadius) * (self.startRadius**3-self.endRadius**3) - np.pi*self.length/(3*(self.startRadius-self.boatTailThickness)) * ((self.startRadius-self.boatTailThickness)**3-(self.endRadius-self.boatTailThickness)**3)
        self.mass = self.volume * mat.Materials(self.boatTailMaterial)

    def drawBoatTail(self):
        # FIRST YEAR JOB
        pass
    

# FINS # 
class Fins():
    def __init__(self, finType, n, finSpan, finRootChord, finMidChord, finTipChord, rocketRadius, pos):
        # [IMPORTANT] See how there are geometric parameters defined for trapezoidal fins ONLY, this should be moved to a separate function in the near future
        # i.e. addTrapezoidalFins(self, finSpanm finRootChord...) instead of initialising these geometric parameters in the fins class itself 
        """
        ----------------------------------------------------------------------
        numberOfFins: number of fins ()
        finType: geometry of fins ()
        finSpan: span of fins (m)
        ----------------------------------------------------------------------
        """
        # Physical parameters
        self.numberOfFins = n
        self.finType = finType
        self.finSpan = finSpan
        self.finRootChord = finRootChord
        self.finMidChord = finMidChord
        self.finTipChord = finTipChord

        # Rocket parameters
        self.rocketRadius = rocketRadius

        # Aerodynamic parameters
        self.kfb = None # Factor for interference between fin and body
        self.cn = None
        self.cnPos = None
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

        self.cnPos = (self.pos + 
                      self.finMidChord*(self.finRootChord+2*self.finTipChord)
                        /(3*(self.finRootChord+self.finTipChord))
                          +(self.finRootChord+self.finTipChord
                          -(self.finRootChord*self.finTipChord
                        /(self.finRootChord+self.finTipChord)))
                      /6
                    )


    def evaluateCD(self):
        self.cd = 0



