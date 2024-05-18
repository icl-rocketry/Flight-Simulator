# a class for the rocket. Parameters are those used in AeroCalculator.py


class Rocket:
    def __init__(
        self,
        noseType,
        noseLength,
        bodyLength,
        boattailLength,
        boattailDRatio,
        rocketDiameter,
        finSweep,
        finRoot,
        finTip,
        finSpan,
        finGap,
        finPos,
        # take the next ones from openrocket or similar, at least it can be used for these
        dryCG,
        propCG,
        dryMass,
        propMass,
        dryInertia,
        propInertia,
        Isp, # TODO: calculate this from propellant mass and thrust etc.
    ):
        self.noseType = noseType  # 0 for Haack, others not implemented yet
        self.noseLength = noseLength  # Length of nose cone (m)
        self.bodyLength = bodyLength  # Length of body tube (m)
        self.boattailLength = boattailLength  # Length of boat tail (m)
        self.boattailDRatio = boattailDRatio  # Ratio of smaller to larger diameter of boat tail
        self.rocketDiameter = rocketDiameter  # Diameter of rocket (m)
        self.finSweep = finSweep  # Sweep angle of fins at midchord (degrees)
        self.finRoot = finRoot  # Root chord of fin (m)
        self.finTip = finTip  # Tip chord of fin (m)
        self.finSpan = finSpan  # tip-to-tip span of fin (m)
        self.finGap = finGap  # gap between fins at tip
        self.finPos = finPos  # distance from nose to leading edge of fin (m)
        self.dryCG = dryCG  # dry mass CG location from nose (m)
        self.propCG = propCG  # propellant mass CG location from nose (m)
        self.dryMass = dryMass  # dry mass of rocket (kg)
        self.propMass = propMass  # propellant mass (kg)
        self.dryInertia = dryInertia  # dry mass inertia tensor (kg*m^2)
        self.propInertia = propInertia  # wet mass inertia tensor (kg*m^2)
        self.Isp = Isp

    def getGeoParams(self):
        # get nose cone related parameters
        if self.noseType == 0:
            Cvf = 0.5  # Volume coefficient for Haack series
            Cpl = 0.651  # planform area coefficient
            Ccl = 1.239  # I can't remember what this is lol
        else:
            raise NotImplementedError("Only Haack series is implemented")

        # non-dimensionalise other parameters
        lfd = self.noseLength / self.rocketDiameter
        lcd = self.bodyLength / self.rocketDiameter
        lad = self.boattailLength / self.rocketDiameter
        L = self.noseLength + self.bodyLength + self.boattailLength
        ld = lfd + lcd  # total length of nose and body, ignore boattail
        rcd = self.finRoot / self.rocketDiameter
        tcd = self.finTip / self.rocketDiameter
        spand = self.finSpan / self.rocketDiameter
        gapd = self.finGap / self.rocketDiameter
        led = self.finPos / self.rocketDiameter
        xm = L / 2  # pitching axis? TODO: check if this is correct
        xml = xm / L  # so this should be 0.5

        return (
            Cvf,
            Cpl,
            Ccl,
            lfd,
            lcd,
            lad,
            self.boattailDRatio,
            L,
            self.rocketDiameter,
            self.finSweep,
            rcd,
            tcd,
            spand,
            gapd,
            led,
            ld,
            xm,
            xml,
        )

    def drawRocket(self):
        pass  # I can do this when im bored
