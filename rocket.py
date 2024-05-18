import numpy as np
from AeroSurfaces import NoseCone, BodyTube, Boattail, TrapezoidalFins
from MoreComponents import MassComponent


class Rocket:
    def __init__(
        self,
        rocketLength,
        rocketRadius,
        finNumber,
        finThickness,
        Cl_max_fins,
        Cl_max_canards,
        launchRailLength,
        launchRailAngle,
        launchRailDirection,
        chuteArea,
        chuteCD,
        chuteMargin,
        m_dry,
        m_wet,
        I_dry,
        I_fuel,
        CG_dry,
        Isp,
        Cd_cyl,
        CG_tank,
    ):
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
        m_dry: Dry mass of rocket (kg)
        m_wet: Wet mass of rocket (kg)
        CG_dry: CG of rocket when dry (from nose) (m)
        Isp: Specific Impulse of rocket (s) - this may not be needed as it is already defined via masses
        Cd_cyl: Drag coefficient of rocket body from the side
        CG_tank: CG of the tank(s) (from nose) (m)
        topA: Area of rocket when viewed from the top (m^2)
        sideA: Area of rocket when viewed from the side (m^2)
        I_dry: Inertia tensor of rocket when dry (kg*m^2)
        I_fuel: Inertia tensor of the fuel (kg*m^2)
        finArm: Moment arm of fin (m)
        sideFinsArea: Area added by fins when the rocket is viewed from the side (m^2)
        finArea: Area of one fin (m^2)
        Cl_max_fins: Max lift coefficient of fins
        launchRailLength: Length of launch rail (m)
        launchRailAngle: Angle of launch rail from vertical (rad)
        launchRailDirection: Heading of launch rail (rad)
        thrust_data: Thrust data of rocket (time, thrust) (s, N)
        ----------------------------------------------------------------------
        """

        # Physical properties of rocket
        self.rocketLength = rocketLength  # Rocket length is for entirety of rocket
        self.rocketRadius = rocketRadius
        self.finNumber = finNumber  # Number of fins
        self.finThickness = finThickness
        self.Cl_max_fins = Cl_max_fins  # Default max lift coefficient of fins
        self.Cl_max_canards = Cl_max_canards  # Default max lift coefficient of canards
        self.launchRailLength = launchRailLength  # Default length of launch rail
        self.launchRailAngle = launchRailAngle  # Default angle of launch rail from vertical
        self.launchRailDirection = launchRailDirection  # Default heading of launch rail
        self.chuteArea = chuteArea  # Default area of parachute (m^2)
        self.chuteCD = chuteCD
        self.chuteMargin = chuteMargin  # Static margin under parachute (m)

        # parameters which need to be evaluated but for now are just placeholders
        self.m_dry = m_dry
        self.m_wet = m_wet
        self.CG_dry = CG_dry
        self.Isp = Isp
        self.Cd_cyl = Cd_cyl
        self.CG_tank = CG_tank
        self.I_dry = I_dry
        self.I_fuel = I_fuel
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
        self.isTooLong = 0  # Variable that computes total length of components

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

        return None

    def evaluateStaticMargin(self):
        """
        Stability analysis by first evaluating the overall position of CG and CP, then evaluate static margin
        """
        # Use Extended Barrowman here to evaluate CP
        return (self.rocketCPPos - self.rocketCGPos) / (2 * self.rocketRadius)
    
    def evaluateRocketVolume(self):
        """
        Evaluates volume of rocket
        """
        self.volume = np.pi * self.rocketRadius ** 2 * self.rocketLength # TODO: this can be better
        return self.volume

    def evaluateRocketCP(self):
        """
        Evaluates centre of pressure of rocket
        """
        rocketCN = 0  # Initialise rocketCN variable (sums the Cn * Xn for each surface)
        alphaSum = 0  # Initialise posSum variable (sums the Xn for each surface)

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                rocketCN += surface.cnAlpha * surface.cnPos
                alphaSum += surface.cnAlpha

        # In the odd case where there are no existing aero surfaces initialsied on rocket...
        if self.rocketCN == None:
            self.rocketCPPos = 0
        else:
            self.rocketCPPos = rocketCN / alphaSum

    def evaluateRocketCG(self):
        """
        Evaluates centre of gravity of rocket
        """

        cgTop = 0
        cgBottom = 0

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                cgTop += surface.mass * 9.81 * surface.cgPos
                cgBottom += surface.mass

        self.rocketCGPos = cgTop / cgBottom

    def evaluateRocketMass(self):
        """
        Evaluates mass of rocket
        """

        mass = 0

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                mass += surface.mass
        return mass

    def evaluateFinProperties(self):
        """
        Evaluates fin area of rocket
        """

        # get the fin object from the surface list
        for surface in self.aerodynamicSurfacesList:
            if isinstance(surface, TrapezoidalFins):
                self.finArea = 0.5 * surface.finSpan * (surface.finRootChord + surface.finTipChord)
                self.finArm = surface.finSpan / 3 * self.rocketRadius
                self.topA = np.pi * self.rocketRadius**2 + surface.numberOfFins * surface.finSpan * surface.thickness
                self.sideA = 2 * np.pi * self.rocketRadius * self.rocketLength + self.finArea

    def evaluateInertia(self):
        """
        Evaluates inertia tensor of rocket
        """

        I = 0

        if len(self.aerodynamicSurfacesList) > 0:
            for surface in self.aerodynamicSurfacesList:
                I += surface.I
        return I
    
    def evaluateRocketCL(self):
        pass

    def evaluateRocketGeometry(self):
        # use ESDU 77028
        dOverL = self.noseLength / (2 * self.rocketRadius)
        lfOverL = self.noseLength / self.rocketLength
        self.dOverL = dOverL
        self.lfOverL = lfOverL


    def evaluateRocketCDV(self, mach, Re):

        self.evalulateRocketGeometry()
        dOverL = self.dOverL
        Cvf = self.Cvf
        lfOverL = self.lfOverL
        
        # calculation of Cd*, based on volume^(2/3) instead of surface area
        # all equations from ESDU 78019

        Fm1 = 0.18 * mach ** 2 / (np.arctan(0.4219 * mach)) ** 2
        Fm2 = (1 + 0.178 * mach ** 2) ** -0.702 / Fm1
        B = 2.62105 - 0.0042167 * np.log10(Fm2 * Re)
        Cf0 = 0.455 / (Fm1 * (np.log10(Fm2 * Re)) ** B)

        # skin friction stuff
        xtrOverL = 0.95 * lfOverL # transition to turbulent flow, distance from nose divided by rocket length
        F1 = 41.1463 * Re ** -0.377849
        g = 0.71916 - 0.0164 * np.log10(Re)
        h = 0.66584 + 0.02307 * np.log10(Re)
        F2 = 1.1669 * np.log10(Re) ** -3.0336 - 0.001487
        Cf = Cf0 * (1 - xtrOverL + F1 * xtrOverL ** g) ** h - F2

        # other conversion factors
        Ktr = 1 + 0.36 * xtrOverL - 3.3 * xtrOverL ** 3
        b = lfOverL * (1 - Cvf)
        Fm = 1.5 * mach ** 2 * (1 + 1.5 * mach ** 4)
        if b < 0.03:
            Fb = 0
        elif b < 0.15:
            Fb = 0.0924 / b + 0.725 * b + 12.2 * b ** 2
        else:
            Fb = 1
        Km = 1 + Fm * Fb * dOverL ** 2

        # final calculations
        CdV_nose = Cf * Ktr * Km * 3.764 * (dOverL ** (-1/3) + 1.75 * dOverL ** (7/6) + 3.48 * dOverL ** (8/3))
        CdV = CdV_nose
        return CdV

    # ---------------------------------------------------------------------- ADD AERODYNAMIC SURFACES ----------------------------------------------------------------------#

    def addNose(self, coneType, length, noseRadius, material, thickness, mass=0):
        """
        Adds nose cone to rocket
        """

        nose = NoseCone(
            coneType, length, self.rocketRadius, material, thickness, mass
        )  # Pass parameters into NoseCone Class
        self.addSurface(nose)  # Add nose cone into rocket, position = 0 as nose is forced to be put at the top

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
            bodyTube = BodyTube(
                length, radius, thickness, position, material, mass
            )  # Pass parameters into BodyTube Class
            self.addSurface(bodyTube)  # Technically contributes to nothing
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
            boatTail = Boattail(
                upperRadius, lowerRadius, length, thickness, position, material, mass
            )
            self.addSurface(boatTail)
        else:
            raise Exception("Body Tube Not Added!")

        return boatTail

    def addTrapezoidalFins(
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
        """
        Adds trapezoidal fins to rocket
        """

        trapFins = TrapezoidalFins(
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
        )

        NoseCone = self.aerodynamicSurfacesList[
            0
        ]  # NOTE: this assumes we add the nose cone first which I think is always the case

        # evaluate fin properties
        trapFins.evaluateCD()
        trapFins.evaluateCN(NoseCone)
        trapFins.evaluateCL()
        trapFins.evaluateMass()
        trapFins.evaluateInertia()

        # add the fins to the rocket
        self.addSurface(trapFins)

        return trapFins

    # ---------------------------------------------------------------------- ADD OTHER COMPONENTS ----------------------------------------------------------------------#

    def addMassComponent(self, mass, pos):
        """
        Adds mass component (to shift cgPos)
        mass in kg and pos in m
        """
        # Check if position is valid
        if pos < 0:
            raise Exception("Position is invalid! Must be greater than 0")
        elif pos > self.rocketLength:
            raise Exception("Mass component is literally not on rocket")

        massComponent = MassComponent(mass, pos)
        self.addsurface(0, 0, massComponent.mass * 9.81, massComponent.pos)

        return massComponent

    # ------------------------------------------------------------------- AERODYNAMIC PARAMETERS -------------------------------------------------------------------#
    # Do this in dynamics model
    # #for aero_surface, position in self.rocket.aerodynamic_surfaces:
    #   c_lift = aero_surface.cl(comp_attack_angle, comp_stream_mach)
    # ---------------------------------------------------------------------- FUNKY FUNCTIONS ----------------------------------------------------------------------#

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

    # ---------------------------------------------------------------------- DATA VALIDATION ----------------------------------------------------------------------#

    def checkTotalLength(self, length):
        self.isTooLong += length
        if self.isTooLong < self.rocketLength + 0.001:  # avoid floating point errors
            pass
        else:
            print(self.isTooLong, self.rocketLength)
            raise Exception("Physically not possible")
