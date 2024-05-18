# General Imports, method is same as RocketPy coz it is neat code
import numpy as np
from function import Function


class NoseCone:
    def __init__(self, coneType, length, noseRadius, material, thickness, mass):
        self.coneType = coneType
        self.noseLength = length
        self.noseRadius = noseRadius
        self.material = material
        self.thickness = thickness
        self.mass = mass

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
        self.Cna = 2 * (self.noseRadius / self.rocketRadius) ** 2  # is this correct?
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

    def geometricalParameters(self):  # Finding rho, the radius ratio of nose cone
        if self.noseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.noseRadius / self.rocketRadius

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
        thickness,
    ):
        self.span = finSpan
        self.rootChord = finRootChord
        self.tipChord = finTipChord
        self.midChord = finMidChord
        self.sweepLength = sweepLength  # sweep on root chord only
        self.sweepAngle = sweepAngle
        self.rootLocation = pos
        self.number = numberOfFins
        self.rocketRadius = rocketRadius  # used in fin interference calcs
        self.mass = mass
        self.thickness = thickness

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

    def evaluateCL(self):
        pass

    def evaluateCD(self):
        pass


class BodyTube:
    def __init__(self, length, radius, thickness, position, material, mass):
        self.length = length
        self.radius = radius
        self.thickness = thickness
        self.position = position
        self.material = material
        self.mass = mass

    # aero is not used here, only the length is used to determine the position of boattail


class Boattail:
    def __init__(self, upperRadius, lowerRadius, length, thickness, position, material, mass):  # conical only
        self.upperRadius = upperRadius
        self.lowerRadius = lowerRadius
        self.upperRadius = upperRadius
        self.length = length
        self.thickness = thickness
        self.position = position
        self.material = material
        self.mass = mass

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
