<<<<<<< Updated upstream
# General Imports, method is same as RocketPy coz it is neat code
import numpy as np
from function import Function


class NoseCone:
    def __init__(self, coneType, length, noseRadius, rocketRadius, shapeParameter):
        self.type = coneType
        self.noseLength = length
        self.noseRadius = noseRadius
        self.rocketRadius = rocketRadius
        self.shapeParameter = shapeParameter

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.Cna = None
        self.Xn = None
        self.cp = None
        self.cl = None
        self.cd = None

    def addNose(self):
        # Note the position of the nose is @ top of rocket coordinate system

        # if self.type == "VonKarman":
        #     self.Xn = 0.5
        self.Cna = 2 * (self.noseRadius / self.rocketRadius) ** 2 # is this correct?
        if self.type == "Haack":
            self.Xn = self.length / 2 * (1 - (3 * self.shapeParameter / 8))
        elif self.type == "Ogive":
            f = 1 / self.shapeParameter  # to be consistent with literature
            lam = np.sqrt(2 * f - 1)
            realOgiveXn = self.radius * ((f**2 - lam**2 / 3) * lam - f**2 * (f - 1) * np.arcsin(lam / f))
            # the actual ogive might be squashed since the length can be changed, so we need to scale it
            realOgiveLength = np.sqrt((2 * lam + 1) * self.radius**2)
            self.Xn = realOgiveXn * self.length / realOgiveLength
        else:
            raise ValueError("Cannot find nose type")

        self.geometricalParameters()  # Evaluate geometrical parameters of nose

        self.evaluateCP()
        self.evaluateCL()

        # Drawing for later broooooooooooooooooo

=======
# General Imports, method is similar to RocketPy coz it is neat code
import numpy as np
import Materials as mat
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
class NoseCone:
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

        # NOTE: Nose cone is automatically positioned @ top of rocket (i.e. pos = 0)

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
        self.k = None  # Cp Factor
        self.k2 = None  # Cp Position Factor, check cambridge doc for this (end of section 3.2.1)
        self.cnAlpha = 0
        self.cn = 0
        self.cl = 0
        self.cd = 0

        # Stability Parameters
        self.cnPos = 0
        self.cgPos = 0

        if self.type.lower() == "vonkarman":
            self.k = 0.5
            self.volume = (
                self.noseLength * np.pi / 2 * (9 / 8) * (self.noseRadius**2 - (self.noseRadius - self.thickness) ** 2)
            )
        elif self.type.lower() == "haack":
            self.k = 0.4375
            self.volume = (np.pi * self.noseLength / 2) * (
                (self.noseRadius**2) - (self.noseRadius - self.thickness) ** 2
            )
        else:
            raise ValueError("Cannot find nose type?")

        if self.mass == 0:
            self.evaluateMass()  # Evaluate mass if mass is not given
        else:
            # self.mass stays as user input
            pass

        self.geometricalParameters()  # Evaluate geometrical parameters of nose
        self.evaluateCN()
        self.evaluateCG()
        self.evaluateCL()

>>>>>>> Stashed changes
    def geometricalParameters(self):  # Finding rho, the radius ratio of nose cone
        if self.noseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.noseRadius / self.rocketRadius

<<<<<<< Updated upstream
    def evaluateCP(self):
        self.cpx = self.Xn
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def evaluateCL(self):
        # Calculate clalpha
        # clalpha is currently a constant, meaning it is independent of Mach
        # number. This is only valid for subsonic speeds.
        # It must be set as a Function because it will be called and treated
        # as a function of mach in the simulation.
        self.clalpha = Function(
            lambda mach: 2 * self.radius_ratio**2,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        return None

    def evaluateCD(self):
        # Calculate clalpha
        # clalpha is currently a constant, meaning it is independent of Mach
        # number. This is only valid for subsonic speeds.
        # It must be set as a Function because it will be called and treated
        # as a function of mach in the simulation.
        self.cdalpha = Function(
            lambda mach: 2 * self.radius_ratio**2,
            "Mach",
            f"Drag coefficient derivative for {self.name}",
        )
        self.cd = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cd",
        )
        return None


class Fins:
    def __init__(self, span, rootChord, tipChord, sweep, rootLocation, number, rocketRadius):
        self.span = span
        self.rootChord = rootChord
        self.tipChord = tipChord
        self.sweep = sweep  # sweep on root chord only
        self.rootLocation = rootLocation
        self.number = number
        self.rocketRadius = rocketRadius  # used in fin interference calcs

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.Cna = None
        self.Xn = None
        self.cp = None
        self.cl = None
        self.cd = None

    def addFins(self):
        # Note the position of the nose is @ top of rocket coordinate system

        l = np.sqrt(self.span**2 + (abs(self.rootChord - self.tipChord) + self.sweep) ** 2)
        self.Cna = (
            (self.number * self.span / (2 * self.rocketRadius)) ** 2
            / (1 + np.sqrt(1 + (2 * l / (self.rootChord + self.tipChord)) ** 2))
            * (1 + self.rocketRadius / (self.span + self.rocketRadius))
        )
        self.Xn = (
            self.rootLocation
            + (self.sweep * (self.rootChord + 2 * self.tipChord) / (3 * (self.rootChord + self.tipChord)))
            + (self.rootChord + self.tipChord - (self.rootChord * self.tipChord) / (self.rootChord + self.tipChord))
            / 6
        )

        self.geometricalParameters()  # Evaluate geometrical parameters of nose

        self.evaluateCP()
        self.evaluateCL()

        # Drawing for later broooooooooooooooooo

    def evaluateCP(self):
        self.cpx = self.Xn
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

=======
    def evaluateCN(self):
        self.cnAlpha = 2
        self.cn = Function(lambda alpha: 2 * alpha)  # Note cnAlpha = 2 for all nose cones
        self.cnPos = self.k * self.noseLength

    def evaluateCG(self):
        self.cgPos = self.k * self.noseLength  # Wrong expression but yeah

    def evaluateCL(self):
        self.clAlpha = Function(lambda mach: 2 * self.radius_ratio**2)
        self.cl = Function(lambda alpha, mach: self.clAlpha(mach) * alpha)
        return None

    def evaluateCD(self):
        # Cd from nose cone accounted for in body tube! From reference!
        pass

    def evaluateMass(self):
        self.mass = mat.Materials(self.noseMaterial).density * 1000 * self.volume
        return self.mass

    def evaluateInertia(self):
        if self.type.lower() == "vonkarman":
            pass
        elif self.type.lower() == "haack":
            pass
        else:
            raise ValueError("Cannot find nose type?")

    def drawNoseCone(self):
        # FIRST YEAR JOB
        pass


### BODY TUBE ###
class BodyTube:
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
        self.bodyTubeRadius = bodyTubeRadius  # Assume cylindrical tube
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
        self.cn = 0  # Cn of body tube = 0 (someone verify this please)
        self.cnPos = 0
        self.cgPos = 0  # In the future, use Fusion API (have wet and dry CG)

        if self.mass == 0:
            self.evaluateMass()  # Evaluate mass if mass is not given
        else:
            # self.mass stays as user input
            pass

        self.evaluateCN()
        self.evaluateCG()

    def evaluateMass(self):
        self.volume = (
            np.pi
            * (self.bodyTubeRadius**2 - (self.bodyTubeRadius - self.bodyTubeThickness) ** 2)
            * self.bodyTubeLength
        )
        self.mass = mat.Materials(self.bodyTubeMaterial).density * 1000 * self.volume

    def evaluateCN(self):
        self.cnPos = (
            0.5 * self.bodyTubeLength
        )  # note this is relative to the body tube, not the entire rocket, need to change this!

    def evaluateCG(self):
        self.cgPos = 0.5 * self.bodyTubeLength + self.bodyTubePos  # Relative to nose cone tip

    def evaluateCL(self):
        self.clAlpha = 2 * np.pi
        self.cl = Function(lambda alpha: self.clAlpha * alpha)
        pass

    def evaluateCD(self):
        self.cd = 1.02 # from the side
        
        pass

    def evaluateInertia(self):
        r1 = self.bodyTubeRadius
        r2 = self.bodyTubeRadius - self.bodyTubeThickness
        h = self.bodyTubeLength
        m = self.mass
        Ixx = (1 / 12) * m * (3 * r1**2 + 3 * r2**2 + h**2)
        Izz = (1 / 12) * m * (r1**2 + r2**2)
        self.I = np.array([[Ixx, 0, 0], [0, Ixx, 0], [0, 0, Izz]])


    def drawBodyTube(self):
        # FIRST YEAR JOB
        pass


### BOAT TAIL ###
class BoatTail:
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

        # NOTE: Boattail is modelled as a conical transition

        # Initial Arguments
        self.startRadius = upperRadius
        self.endRadius = lowerRadius  # Assume big to small boat tail, how else are you going to do it?
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
            self.evaluateMass()  # Evaluate mass if mass is not given
        else:
            # self.mass stays as user input
            pass

        self.evaluateCN()  # Evaluate Cn and its position

    def evaluateMass(self):
        # [Note Boat Tail MAss is kinda wrong]
        self.volume = np.pi * self.length / (3 * self.startRadius) * (
            self.startRadius**3 - self.endRadius**3
        ) - np.pi * self.length / (3 * (self.startRadius - self.boatTailThickness)) * (
            (self.startRadius - self.boatTailThickness) ** 3 - (self.endRadius - self.boatTailThickness) ** 3
        )
        self.mass = self.volume * mat.Materials(self.boatTailMaterial).density

        # Modelling the boat tail as a 3D trapezoidal component... The CG position can be given by a common equation
        self.cgPos = (2 * self.lowerRadius + self.upperRadius) * self.length / (
            3 * (self.upperRadius + self.lowerRadius)
        ) + self.boatTailPos

    def evaluateCN(self):
        # Sanity check, a boat tail should have a negative CP, hence pushing the total CP forward
        dRatio = self.rocketRadius / self.endRadius
        self.cnAlpha = 2 * ((self.endRadius / self.rocketRadius) ** 2 - (self.startRadius / self.rocketRadius) ** 2)
        self.cn = Function(lambda alpha: self.cnAlpha * alpha)  # Modelling boat tail as body tube transition
        self.cnPos = self.boatTailPos + self.length / 3 * (1 + (1 - dRatio) / (1 - dRatio**2))

    def evaluateInertia(self):
        r1t = self.startRadius - self.boatTailThickness
        r2t = self.endRadius - self.boatTailThickness
        t = self.boatTailThickness
        r1 = self.startRadius
        r2 = self.endRadius
        m = self.mass
        Ixx = 0 # TODO: actually do the maths for this
        Izz1 = (m / (20 * t * (r1 + r2))) * (4 * r2**5 + 8 * r2t ** 5 + 5 * r2t**4) # from inner section
        Izz2 = (1 / 12) * m * ((r1-t)**2 + r2**2) # from middle section
        Izz3 = (m / (20 * t * (r1 + r2))) * (4 * r1**5 + 8 * r1t ** 5 + 5 * r1t**4) # from outer section
        Izz = Izz1 + Izz2 + Izz3
        I = np.array([[Ixx, 0, 0], [0, Ixx, 0], [0, 0, Izz]])

    def drawBoatTail(self):
        # FIRST YEAR JOB
        pass


### FINS ###
class TrapezoidalFins:
    def __init__(
        self,
        numberOfFins,
        finSpan,
        finRootChord,
        finMidChord,
        finTipChord,
        sweepLength,
        sweepAngle,
        rocketRadius,
        pos,
        mass,
        thickness
    ):
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
        self.sweepLength = sweepLength
        self.thickness = thickness

        # Rocket parameters
        self.rocketRadius = rocketRadius

        # Aerodynamic parameters
        self.kfb = None  # Factor for interference between fin and body
        self.cn = None
        self.cnPos = None
        self.cgPos = None
        self.cl = None
        self.cd = None

        # Physical parameters
        self.pos = pos
        self.volume = None
        self.mass = mass

    def addTrapezoidalFins(self):
        if self.mass == 0:
            self.evaluateMass()  # Evaluate mass if mass is not given
        else:
            # self.mass stays as user input
            pass
        self.evaluateCN()

    def evaluateMass(self):
        centroidTop = self.finRootChord/3
        centroidMid = self.finRootChord/2 + self.sweepLength
        centroidBot = (1 / 3) * (2 * self.finRootChord + self.finTipChord + 2 * self.sweepLength) # ignore this part for now (its generally not that big)
        weightTop = self.finRootChord * self.finSpan / 4
        weightMid = self.finTipChord * self.finSpan
        weightBot = ((self.finRootChord - self.finTipChord) / 2 - self.sweepLength) * self.finSpan / 2 # again, ignore this part for now
        self.cgPos = centroidTop * weightTop + centroidMid * weightMid + centroidBot * weightBot + self.pos

    def evaluateCN(self, NoseCone):

        self.kfb = 1 + self.rocketRadius / (2 * self.finSpan + self.rocketRadius)  # Calculate interference factor

        if NoseCone.noseRadius is not None:
            self.cnAlpha = (
                self.kfb
                * (4 * self.numberOfFins * (self.finSpan / NoseCone.noseRadius) ** 2)
                / (1 + np.sqrt(1 + (2 * self.finMidChord / (self.finRootChord + self.finTipChord)) ** 2))
            )
        else:
            raise ValueError("Nose cone not defined yet")

        self.cn = Function(lambda alpha: self.cnAlpha * alpha)  # Assume small angles

        self.cnPos = (
            self.pos
            + self.finMidChord
            * (self.finRootChord + 2 * self.finTipChord)
            / (3 * (self.finRootChord + self.finTipChord))
            + (
                self.finRootChord
                + self.finTipChord
                - (self.finRootChord * self.finTipChord / (self.finRootChord + self.finTipChord))
            )
            / 6
        )

>>>>>>> Stashed changes
    def evaluateCL(self):
        pass

    def evaluateCD(self):
<<<<<<< Updated upstream
        pass


class BodyTube:
    def __init__(self, length):
        self.length = length

    # aero is not used here, only the length is used to determine the position of boattail


class Boattail:
    def __init__(self, upperRadius, lowerRadius, rocketRadius, length, topLocation):  # conical only
        self.upperRadius = upperRadius
        self.lowerRadius = lowerRadius
        self.rocketRadius = rocketRadius
        self.length = length
        self.topLocation

        # Aerodynamic Parameters
        # reference: https://offwegorocketry.com/userfiles/file/Nose%20Cone%20&%20Fin%20Optimization.pdf
        self.Cna = None
        self.Xn = None
        self.cp = None
        self.cl = None
        self.cd = None

    def addBoattail(self):
        # Note the position of the nose is @ top of rocket coordinate system

        self.Cna = 2 * (self.upperRadius / self.rocketRadius) ** 2 * ((self.lowerRadius / self.upperRadius) ** 2 - 1)
        self.Xn = (
            (1 / 3)
            * (1 + (1 - (self.upperRadius / self.lowerRadius)) / (1 - (self.upperRadius / self.lowerRadius) ** 2))
        ) * self.length + self.topLocation

        self.geometricalParameters()  # Evaluate geometrical parameters of nose

        self.evaluateCP()
        self.evaluateCL()

        # Drawing for later broooooooooooooooooo

    def evaluateCP(self):
        self.cpx = self.Xn
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def evaluateCL(self):
        pass

    def evaluateCD(self):
        pass


#
=======
        self.cd = 0

    def evaluateInertia(self):
        pass # do we need this?

    def drawFins(self):
        # FIRST YEAR JOB
        pass
>>>>>>> Stashed changes
