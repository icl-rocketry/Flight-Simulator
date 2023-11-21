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
        self.Cna = 2  # doesn't need evaluating, it is 2 by default for a nosecone
        self.Xn = None
        self.cp = None
        self.cl = None
        self.cd = None

    def addNose(self):
        # Note the position of the nose is @ top of rocket coordinate system

        # if self.type == "VonKarman":
        #     self.Xn = 0.5
        if self.type == "Haack":
            self.Xn = self.length / 2 * (1 - (3 * self.shapeParameter / 8))
        elif self.type == "Ogive":
            f = 1 / self.shapeParameter  # to be consistent with literature
            lam = np.sqrt(2 * f - 1)
            realOgiveXn = self.radius * ((f**2 - lam**2 / 3) * lam - f**2 * (f - 1) * np.arcsin(lam / f))
            # the actual ogive might be squashed, so we need to scale it
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

    def evaluateCL(self):
        pass

    def evaluateCD(self):
        pass


class BodyTube:
    def __init__(self, length):
        self.length = length

    # aero is not used here, only the length is used to determine the position of boattail


class Boattail:
    def __init__(self, upperRadius, lowerRadius, rocketRadius, length, topLocation): #conical only
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
        self.Xn = ()*self.length + self.topLocation

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
