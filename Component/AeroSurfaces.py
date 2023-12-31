# General Imports, method is similar to RocketPy coz it is neat code
import numpy as np
import Component.Materials as mat
from utils.function import Function


# CHECKLIST # 
# STATICS: MASS
# STABILITY: CENTRE OF GRAVITY, CENTRE OF PRESSURE
# DYANMICS: CL, CD, CN


# ADD MASS OVERRIDE OPTION 
# METHOD 1: ARGUMENT = 0 OR 1 FOR MASS MODE (MASSOVERRIDE = 1)
# METHOD 2: KEYWORD ARGUMENT (DEFAULT VALUE I.E. MASS=100, IF NO VALUE GIVEN THEN USE MASS FUNCTION) [THIS]

# POSITION, FORCE FOR NOW, SO USE ISTOOLONG VARIABLE

# DRAG CALCULATOR????
# Maybe seperate module for this, where we pass geometrical parameters of all surfaces

### NOSE CONE ###
class NoseCone():
    def __init__(self, coneType, length, noseRadius, rocketRadius, material, thickness, mass):
        """
        ----------------------------------------------------------------------
        type: Type of nose cone (string)
        noseLength: Length of nose cone (m)
        noseRadius: Radius of nose cone (m)
        thickness: Thickness of nose cone (m)
        rocketRadius: Radius of rocket (m)
        ----------------------------------------------------------------------
        noseMaterial: Material of nose cone (string)
        volume: Volume of nose (m^3)
        mass: Mass of nose (kg)
        ----------------------------------------------------------------------
        k: Length factor used to determine the position of CP of nose cone ()
        (k2): Not sure what this does yet... 
        cn: Normal coefficient of nose ()
        cl: Lift coefficient of nose ()
        cd: Drag coefficient of nose ()
        ----------------------------------------------------------------------
        cnPos: Position of CP of nose, relative to entire rocket body (m)
        cgPos: Position of CG of nose, relative to entire rocket body (m)
        ----------------------------------------------------------------------
        """   

        #NOTE: Nose cone is automatically positioned @ top of rocket (i.e. pos = 0)

        # Nose Cone Design Parameters
        self.type = coneType
        self.noseLength = length
        self.noseRadius = noseRadius
        self.thickness = thickness
        self.rocketRadius = rocketRadius

        # Physical Parameters
        self.noseMaterial = material
        self.volume = None 
        self.mass = mass

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.k = None # Cp Factor
        self.k2 = None # Cp Position Factor, check cambridge doc for this (end of section 3.2.1)
        self.cnAlpha = 0
        self.cn = 0
        self.cl = 0
        self.cd = 0

        # Stability Parameters
        self.cnPos = 0
        self.cgPos = 0

        if self.type.lower() == "vonkarman":    
            self.k = 0.5
            self.volume = self.noseLength * np.pi/ 2 * (9/8) * (self.noseRadius**2-(self.noseRadius-self.thickness)**2)
        elif self.type.lower() == "haack":
            self.k = 0.437
            self.volume = (np.pi*self.noseLength/2) * ((self.noseRadius**2) - (self.noseRadius-self.thickness)**2)
        else:
            raise ValueError("Cannot find nose type?")
        
        if self.mass == 0:
            self.evaluateMass() # Evaluate mass if mass is not given  
        else:
            # self.mass stays as user input
            pass 

        self.geometricalParameters() # Evaluate geometrical parameters of nose
        self.evaluateCN()
        self.evaluateCG()
        self.evaluateCL()
    
    
    def geometricalParameters(self): # Finding rho, the radius ratio of nose cone
        if self.noseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.noseRadius/self.rocketRadius
        

    def evaluateCN(self):
        self.cnAlpha = 2
        self.cn = Function(lambda alpha: 2*alpha) # Note cnAlpha = 2 for all nose cones
        self.cnPos = self.k * self.noseLength

    def evaluateCG(self):
        self.cgPos = self.k * self.noseLength # Wrong expression but yeah

    def evaluateCL(self):
        self.clAlpha = Function(
            lambda mach: 2 * self.radius_ratio**2
            )
        self.cl = Function(
            lambda alpha, mach: self.clAlpha(mach) * alpha
            )
        
        return None

    def evaluateCD(self):
        # Cd from nose cone accounted for in body tube! From reference!
        pass

    def evaluateMass(self):
        self.mass = mat.Materials(self.noseMaterial).density*1000*self.volume
        return self.mass

    def drawNoseCone(self):
        # FIRST YEAR JOB
        pass



### BODY TUBE ###
class BodyTube():
    def __init__(self, bodyTubeLength, bodyTubeRadius, thickness, position, material, mass):
        """
        ----------------------------------------------------------------------
        bodyTubeLength: Length of body tube (m)
        bodyTubeRadius: Radius of body tube (m)
        bodyTubeThickness: Thickness of body tube (m)
        ----------------------------------------------------------------------
        bodyTubeMaterial: Material of body tube ()
        volume: Volume of body tube (m^3)
        mass: Mass of body tube (kg)
        ----------------------------------------------------------------------
        cd: Drag coefficient of body tube ()
        cn: Normal coefficient of body tube ()
        ----------------------------------------------------------------------
        cnPos: Position of CP of nose, relative to entire rocket body (m)
        cgPos: Position of CG of nose, relative to entire rocket body (m)
        ----------------------------------------------------------------------
        """
        self.bodyTubeLength = bodyTubeLength
        self.bodyTubeRadius = bodyTubeRadius # Assume cylindrical tube
        self.bodyTubeThickness = thickness

        # Physical parameters of body tube
        self.bodyTubeMaterial = material
        self.bodyTubePos = position
        self.volume = 0
        self.mass = mass

        # Aerodynamic parameters
        self.cd = 0
        self.clAlpha = 0
        self.cl = 0
        
        # Stability parameters
        self.cnAlpha = 2
        self.cn = 0 # Cn of body tube = 0 (someone verify this please)
        self.cnPos = 0
        self.cgPos = 0 # In the future, use Fusion API (have wet and dry CG)

        if self.mass == 0:
            self.evaluateMass() # Evaluate mass if mass is not given  
        else:
            # self.mass stays as user input
            pass 
        
        self.evaluateCN()
        self.evaluateCG()


    def evaluateMass(self):
        self.volume = np.pi*(self.bodyTubeRadius**2-(self.bodyTubeRadius-self.bodyTubeThickness)**2)*self.bodyTubeLength
        self.mass = mat.Materials(self.bodyTubeMaterial).density*1000*self.volume

    def evaluateCN(self):
        self.cnPos = 0.5 * self.bodyTubeLength # note this is relative to the body tube, not the entire rocket, need to change this!

    def evaluateCG(self):
        self.cgPos = 0.5 * self.bodyTubeLength + self.bodyTubePos # Relative to nose cone tip
           
    def evaluateCL(self):
        self.clAlpha = 2*np.pi
        self.cl = Function(lambda alpha: 2*np.pi * alpha)
        pass

    def evaluateCD(self):
        self.cd = 1.02
        pass

    def drawBodyTube(self):
        # FIRST YEAR JOB
        pass


### BOAT TAIL ###
class BoatTail():
    def __init__(self, upperRadius, lowerRadius, length, rocketRadius, thickness, position, material, mass):
        """
        ----------------------------------------------------------------------
        startRadius: Starting radius of boat tail, r1 (m)
        endRadius: End radius of boat tail, r2 (m)
        rocketRadius: Radius of rocket (m)
        length: Length of boat tail (m)
        boatTailPos: Position of boat tail, relative to whole rocket body (m)
        ----------------------------------------------------------------------
        cn: Normal coefficient of boat tail ()
        cl: Lift coefficient of boat tail ()
        cd: Drag coefficient of boat tail ()
        ----------------------------------------------------------------------
        cnPos: Position of CP of nose, relative to entire rocket body (m)
        cgPos: Position of CG of nose, relative to entire rocket body (m)
        ----------------------------------------------------------------------
        boatTailMaterial: Material of boat tail ()
        boatTailThickness: Thickness of boat tail (m)
        volume: Volume of boat tail (m^3)
        mass: Mass of boat tail (kg)
        ----------------------------------------------------------------------
        """

        #NOTE: Boattail is modelled as a conical transition

        # Initial Arguments
        self.startRadius = upperRadius
        self.endRadius = lowerRadius # Assume big to small boat tail, how else are you going to do it?
        self.rocketRadius = rocketRadius
        self.length = length
        self.boatTailPos = position
        
        # Aerodynamic parameters
        self.cnAlpha = 0
        self.cn = None
        self.cl = None
        self.cd = None

        # Stability parameters
        self.cnPos = 0
        self.cgPos = 0

        # Physical parameters of boat tail
        self.boatTailMaterial = material
        self.boatTailThickness = thickness
        self.volume = 0
        self.mass = mass
    
    def add(self):
        if self.mass == 0:
            self.evaluateMass() # Evaluate mass if mass is not given  
        else:
            # self.mass stays as user input
            pass 

        self.evaluateCN() # Evaluate Cn and its position

    def evaluateMass(self):
        # [Note Boat Tail MAss is kinda wrong]
        self.volume = np.pi*self.length/(3*self.startRadius) * (self.startRadius**3-self.endRadius**3) - np.pi*self.length/(3*(self.startRadius-self.boatTailThickness)) * ((self.startRadius-self.boatTailThickness)**3-(self.endRadius-self.boatTailThickness)**3)
        self.mass = self.volume * mat.Materials(self.boatTailMaterial).density
        
        # Modelling the boat tail as a 3D trapezoidal component... The CG position can be given by a common equation
        self.cgPos = (2*self.lowerRadius + self.upperRadius) * self.length / (3*(self.upperRadius + self.lowerRadius)) + self.boatTailPos


    def evaluateCN(self):
        # Sanity check, a boat tail should have a negative CP, hence pushing the total CP forward
        dRatio = self.rocketRadius/self.endRadius
        self.cnAlpha = 2*((self.endRadius/self.rocketRadius)**2 - (self.startRadius/self.rocketRadius)**2)
        self.cn = Function(lambda alpha: self.cnAlpha * alpha) # Modelling boat tail as body tube transition
        self.cnPos = self.boatTailPos + self.length/3 * (1+(1-dRatio)/(1-dRatio**2))

    def drawBoatTail(self):
        # FIRST YEAR JOB
        pass
    

### FINS ### 
class TrapezoidalFins():
    def __init__(self, numberOfFins, finSpan, finRootChord, finMidChord, finTipChord, sweepLength, sweepAngle, rocketRadius, pos, mass):
        """
        ----------------------------------------------------------------------
        numberOfFins: Number of fins ()
        finSpan: span of fins (m)
        finRootChord: Root chord of trapezoidal fins (m)
        finMidChord: Mid chord of trapezoidal fins (m)
        finTipChord: Tip chord of trapezoidal fins (m)
        ----------------------------------------------------------------------
        """
        # Physical parameters
        self.numberOfFins = numberOfFins
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

        # Physical parameters
        self.pos = pos
        self.volume = None
        self.mass = mass

    def addTrapezoidalFins(self):
        if self.mass == 0:
            self.evaluateMass() # Evaluate mass if mass is not given  
        else:
            # self.mass stays as user input
            pass 
        
        self.evaluateCN()

    def evaluateMass(self):
        pass

    def evaluateCN(self):

        self.kfb = 1 + self.rocketRadius/(2*self.finSpan+self.rocketRadius) # Calculate interference factor

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
                      self.finMidChord*(self.finRootChord+2*self.finTipChord)/(3*(self.finRootChord+self.finTipChord))
                          +(self.finRootChord+self.finTipChord
                          -(self.finRootChord*self.finTipChord/(self.finRootChord+self.finTipChord)))/6
                      )

    def evaluateCL(self):
        pass

    def evaluateCD(self):
        self.cd = 0

    def drawFins(self):
        # FIRST YEAR JOB
        pass



