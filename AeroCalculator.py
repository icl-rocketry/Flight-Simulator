"""This file computes the relevant aerodynamic parameters 
from the geometrical characterisation of the rocket"""

# imports
from numpy import *
from scipy.interpolate import RegularGridInterpolator


# functions
def getValue2D(x, y, filename):
    """This function gets the value from the data using interpolation
    with the given x and y values, by accessing the csv file given"""
    data = genfromtxt(filename, delimiter=",")
    # extract x and y values
    xValues = data[1, 1:]
    yValues = data[2:, 0]
    dataValues = data[2:, 1:]
    # interpolate
    interp = RegularGridInterpolator((yValues, xValues), dataValues)
    return interp([x, y])[0]


def getValue3D(x, y, z, filename):
    data = genfromtxt(filename, delimiter=",")
    nanIndices = [i for i, row in enumerate(data) if row[0] != row[0]]
    jumpIndices = [0] + [nanIndices[i] for i in range(1, len(nanIndices)) if nanIndices[i] - nanIndices[i - 1] > 1]
    dataArrays = [
        data[jumpIndices[i] : jumpIndices[i + 1] if i + 1 < len(jumpIndices) else None, :]
        for i in range(len(jumpIndices) - 1)
    ] + [data[jumpIndices[-1] :, :]]
    zValues = [dataArray[0, 1] for dataArray in dataArrays]

    aboveIndex = next((i for i, val in enumerate(zValues) if val > z), None)
    belowIndex = aboveIndex - 1 if aboveIndex is not None else None
    if z in zValues:
        aboveIndex = belowIndex = zValues.index(z)

    interpAbove = RegularGridInterpolator(
        (dataArrays[aboveIndex][3:, 0], dataArrays[aboveIndex][2, 1:]), dataArrays[aboveIndex][3:, 1:]
    )
    interpBelow = RegularGridInterpolator(
        (dataArrays[belowIndex][3:, 0], dataArrays[belowIndex][2, 1:]), dataArrays[belowIndex][3:, 1:]
    )

    if z in zValues:
        return interpAbove([x, y])[0]

    val = (interpBelow([x, y]) * (zValues[aboveIndex] - z) + interpAbove([x, y]) * (z - zValues[belowIndex])) / (
        zValues[aboveIndex] - zValues[belowIndex]
    )
    return val[0]


def getAeroParams(M, alpha, logR, Rocket):
    """Gets aero parameters of the rocket (force and moment)
    M:      M number
    alpha:  angle of attack in degrees
    T:      temperature in K
    a:      speed of sound in m/s
    xm:     distance from the nose to pitch axis
    Rocket: rocket object used to get the geometry

    outputs:
    Cn:     normal force coefficient
    Cm:     pitching moment coefficient
    xcp:    center of pressure in m from the nose cone tip
    Mq:     pitch damping derivative"""

    (
        Cvf,
        Cpl,
        Ccl,
        lfd,
        lcd,
        lad,
        dbd,
        L,
        D,
        sweep,
        rcd,
        tcd,
        spand,
        gapd,
        led,
        ld,
        xm,
        xml,
    ) = Rocket.getGeoParams()

    Re = 10**logR
    beta = abs(M**2 - 1) ** 0.5

    if M >= 0.8 or M <= 1.36**0.5:
        beta = 0.64  # avoid singularities
    if beta >= 1 and beta <= 1.001:
        beta = 1.001  # avoid singularities

    # interpolating from ESDU 89008
    Cna_l = getValue2D(M, lfd, "inviscid.csv")
    k = getValue3D(lfd, lcd, M, "overflowFactor.csv")
    CmCn = getValue3D(lfd, lcd, M, "momentRatio.csv")
    dStarL = 0.001 * getValue2D(M, logR, "displacementThickness.csv")
    Cf = 0.001 * getValue2D(M, logR, "skinFriction.csv")

    # force and moment coefficient calculations from ESDU 89008
    delta_CmCn = (Cvf - 0.54) * lfd
    delta_Cna_d = 8 * dStarL * ld
    delta_Cna_f = 4 * Cf * ld
    Cna = k * Cna_l + delta_Cna_d + delta_Cna_f
    delta_Cm0a_d = -4 * dStarL * ld**2
    delta_Cm0a_f = -2 * Cf * ld**2
    Cm0a = k * Cna_l * (CmCn + delta_CmCn) + delta_Cm0a_d + delta_Cm0a_f  # about the nose

    # Cnc from ESDU 89014
    Cnc = getValue2D(M, alpha, "vortexGeneration.csv")

    # convert coefficients to forces and moments using ESDU 89014
    alpha = radians(alpha)
    Cn = Cna * sin(alpha) * cos(alpha) + 4 / pi * ld * Cpl * Cnc
    Cm0 = Cm0a * sin(alpha) * cos(alpha) - 2 / pi * ld**2 * Cpl * Ccl * Cnc
    Cma = Cna * (xml * ld) + Cm0a
    Cm = Cn * (xml * ld) + Cm0  # pitching moment about midpoint

    # modifications due to boattail, from ESDU 87033
    A = -1.3 * lad + 6.35 * lad - 7.85
    if M >= 0 and M < 1:
        delta_Cna = -2 * (1 - dbd**2)
    else:
        delta_Cna = -(0.6 + 1.4 * exp(A * (M - 1) ** 0.8)) * (1 - dbd**2)
    delta_Cdc = -(2 + tanh(1.5 * M * sin(alpha) - 2.4)) * lad * (1 - dbd)
    delta_Cn = delta_Cna * sin(alpha) * cos(alpha) * (1 - (sin(alpha) ** 0.6)) + delta_Cdc * (sin(alpha) ** 2)
    Cn += delta_Cn
    # we don't modify Cm as the fins affect it instead (trust ESDU)
    if alpha == 0:  # avoid division by zero
        Cm = 0
        xcp = -Cm0a / Cna
    else:
        xcp = xml * ld - Cm / Cn

    # get fin parameters (ESDU 91004 Appendix)
    lam = tcd / rcd
    sf = 0.5 * (tcd + rcd) * (spand - gapd)  # * D^2
    AR = (spand - gapd) ** 2 / sf
    back = AR * tan(radians(sweep))  # distance swept back at mid-chord
    betaAR = beta**0.5 * AR

    # get fin lift-curve slope etc. from ESDU 70011/70012
    if M <= 1:
        Cna_fin = AR * getValue3D(back, betaAR, lam, "finSubsonic.csv")
        xOverC = getValue3D(lam, betaAR, back, "finCentreSubsonic.csv")
    else:
        Cna_fin = AR * getValue3D(back, betaAR, lam, "finSupersonic.csv")
        xOverC = getValue3D(lam, betaAR - back, back, "finCentreSupersonic.csv")
    xOverD = xOverC * rcd / (2 * (1 + lam + lam**2)) * (3 * (1 + lam))  # TODO: check this?
    xfl = (led + xOverD) / ld  # distance from nose to x_ac

    # fin interference (this bit is really annoying - see ESDU S.01.03.01)
    r = 0.5 * D
    s = spand * D / 2
    Kwb = (
        (2 / pi)
        * (
            (1 + r**4 / s**4) * (0.5 * arctan(0.5 * (s / r - r / s)) + pi / 4)
            - (r**2 / s**2) * ((s / r - r / s) + 2 * arctan(r / s))
        )
        / (1 - r / s) ** 2
    )
    # the following is used to determine the shocks on the fin
    shockParam = AR * beta**0.5 * (1 + lam) * (tan(radians(sweep)) + 1)
    if shockParam < 4:
        # this is the subsonic case, there is only one equation
        Kbw = (1 + r / s) ** 2 - Kwb
    else:
        # define a load of dimensionless parameters to make the equation simpler
        pB = beta**-0.5 / tan(radians(sweep))  # B
        pD = 2 * r * beta**0.5 / (rcd * D)  # D
        pP = (ld - rcd - led) * D / (2 * rcd * beta**0.5)  # P
        pP = min(pP, 1)  # P is limited to 1
        pR = pP + 1 / pD  # R
        # subsonic LE (B < 1)
        if pB < 1:
            if pR < 1:
                Kbw = ((16 * pB**0.5 * pD) / (pi * (pB + 1))) * (
                    ((pB**1.5 / (pD**2 * (1 + pB))) * (((pB + (1 + pB) * pP * pD) / pB) ** 0.5) - 2)
                    - (pB / (1 + pB)) * pD**-0.5 * (pB * pR + pP) ** 1.5
                    + pB * (1 + pB) * pR**2 * arctan(1 / (pD * (pB * pR + pP))) ** 0.5
                )
            else:  # what even is this lol
                Kbw = ((16 * pB**0.5 * pD) / (pi * (pB + 1))) * (
                    ((pB**1.5 / (pD**2 * (1 + pB))) * (((pB + (1 + pB) * pP * pD) / pB) ** 0.5) - 2)
                    - (pB / (1 + pB)) * pD**-0.5 * (pB * pR + pP) ** 1.5
                    + pB * (1 + pB) * pR**2 * arctan(1 / (pD * (pB * pR + pP))) ** 0.5
                ) + ((16 * pB**0.5 * pD) / (pi * (pB + 1))) * (
                    ((pB * pR + 1) * ((pR - 1) * (pB * pR + 1)) ** 0.5)
                    - (pB + 1) / (pB**0.5) * arctanh((pB * pR - pB) / (pB * pR + 1)) ** 0.5
                    - pB * (1 + pB) * pR**2 * arctan((pR - 1) / (pB * pR + 1)) ** 0.5
                )
        else:  # supersonic LE (B > 1)
            if pR < 1:
                Kbw = (
                    (8 * pD)
                    / (pi * (pB**2 - 1) ** 0.5)
                    * (
                        (-pB / (1 + pB)) * (pB * pR + pP) ** 2 * arccos((pR + pB * pP) / (pB * pR + pP))
                        + ((pB * (pB**2 - 1) ** 0.5) / (pD**2 * (1 + pB))) * ((1 + 2 * pP * pD) ** 0.5 - 1)
                        - (pB**2) / (pD**2 * (1 + pB)) * arccos(1 / pB)
                        + (pB * pR**2 * (pB**2 - 1) ** 0.5 * arccos(pP / pR))
                    )
                )
            else:
                Kbw = (
                    (8 * pD)
                    / (pi * (pB**2 - 1) ** 0.5)
                    * (
                        (-pB / (1 + pB)) * (pB * pR + pP) ** 2 * arccos((pR + pB * pP) / (pB * pR + pP))
                        + ((pB * (pB**2 - 1) ** 0.5) / (pD**2 * (1 + pB))) * ((1 + 2 * pP * pD) ** 0.5 - 1)
                        - (pB**2) / (pD**2 * (1 + pB)) * arccos(1 / pB)
                        + (pB * pR**2 * (pB**2 - 1) ** 0.5 * arccos(pP / pR))
                    )
                ) + ((8 * pD) / (pi * (pB**2 - 1) ** 0.5)) * (
                    (pB * pR + 1) ** 2 * arccos((pR + pB) / (pB * pR + 1))
                    - (pB**2 - 1) ** 0.5 * arccosh(pR)
                    + pB * pR**2 * (pB**2 - 1) ** 0.5 * (arcsin(1 / pR) - pi / 2)
                )
    # convert to the correct form
    Kbw /= beta * Cna_fin * (1 + lam) * (s / r - 1)

    # now we need pitch damping derivatives, from ESDU 91004
    xcl = xml  # TODO: distance of volume centroid from the nose tip
    # F1 = getValue2D(M, xml, "f1.csv")  # TODO: I literally made the subsonic numbers up
    # F2 = (1.045 * ld**2 - 0.438 * ld + 8.726) / (ld**2 - 1.009 * ld + 12.71)
    MqPlusMwdot = -Cna * (1 - xml) ** 2 * ld**2  # use F1*F2-xml if we consider boattail effects
    Mwdot = Cma * ((Cvf * (xcl - xml) * ld) / ((1 - xml) * dbd**2 - Cvf))

    # now incorporate the fin effects
    Sf = (tcd + rcd) * (spand - gapd) * D**2 / 4
    Sref = pi * D**2 / 4
    Mq = MqPlusMwdot - Mwdot - Cna_fin * (Kwb + Kbw) * (Sf / Sref) * (xfl - xml) ** 2 * ld**2

    # drag
    # calculation of Cdp (profile drag coefficient) - applies for small incidence too (<3deg)
    # profile drag equations from ESDU 78019

    # some nice geometry stuff
    # Cv is the ratio of the volumes of the rocket and its enclosing cylinder
    # Cs is the ratio of the surface area of the rocket to the surface area of its enclosing cylinder
    Cva = 1 / 3 * (1 + dbd + dbd**2)
    Cv = lcd / (ld + lad) + Cva * (lad / (ld + lad)) + Cvf * (lfd / (ld + lad))
    Csa = 0.5 * (1 + dbd) * (1 + 0.25 * ((1 - dbd) / lad) ** 2) ** 0.5
    Csf = (0.2642 * lfd**-2 + 0.6343 * lfd**-1 + 2.214) / (lfd**-1 + 3.402)
    Cs = lcd / (ld + lad) + Csa * (lad / (ld + lad)) + Csf * (lfd / (ld + lad))

    if M == 0:
        Fm1 = 0.1011
    else:
        Fm1 = 0.18 * M**2 / (arctan(0.4219 * M)) ** 2
    Fm2 = (1 + 0.178 * M**2) ** -0.702 / Fm1
    B = 2.62105 - 0.0042167 * log10(Fm2 * Re)
    Cf0 = 0.455 / (Fm1 * (log10(Fm2 * Re)) ** B)
    lfOverL = lfd / ld

    # skin friction stuff
    xtrOverL = 0.95 * lfOverL  # transition to turbulent flow, distance from nose divided by rocket length
    F1 = 41.1463 * Re**-0.377849
    g = 0.71916 - 0.0164 * log10(Re)
    h = 0.66584 + 0.02307 * log10(Re)
    F2 = 1.1669 * log10(Re) ** -3.0336 - 0.001487
    Cf = Cf0 * ((1 - xtrOverL + F1 * xtrOverL**g) ** h - F2)

    # other conversion factors
    Ktr = 1 + 0.36 * xtrOverL - 3.3 * xtrOverL**3
    b = lfOverL * (1 - Cvf)
    Fm = 1.5 * M**2 * (1 + 1.5 * M**4)
    if b < 0.03:
        Fb = 0
    elif b < 0.15:
        Fb = 0.0924 / b + 0.725 * b + 12.2 * b**2
    else:
        Fb = 1
    Km = 1 + Fm * Fb * (1 / ld) ** 2

    # final calculations
    CdV_fins = 0
    CdV_nose = Cf * Ktr * Km * 3.764 * ((1 / ld) ** (-1 / 3) + 1.75 * (1 / ld) ** (7 / 6) + 3.48 * (1 / ld) ** (8 / 3))
    CdV = CdV_nose + CdV_fins
    Cdp = CdV * (Cv ** (2 / 3) / (2 * (2 * pi * ld) ** (1 / 3) * Cs))

    # look at B.S.02.03.01 (very important!)

    # TODO: look at ESDU 76033/78041/79022 to add base drag
    boattailAngle = degrees(arctan2(1 - dbd, 2))
    if M <= 0.8:
        # Subsonic base drag - ESDU 76033
        Cdb = getValue2D(dbd, boattailAngle, "baseDragSubsonic.csv")
        Cd_beta = getValue3D(dbd, boattailAngle, M, "boattailDragSubsonic.csv")
    elif M <= 1.3:
        # Transonic base drag - ESDU 78041
        Cdb = getValue3D(dbd, boattailAngle, M, "baseDragTransonic.csv")
        Cd_beta = getValue3D(dbd, boattailAngle, M, "boattailDragTransonic.csv")
    else:
        # Supersonic base drag - ESDU 79022
        Cdb = getValue3D(dbd**2, boattailAngle, M, "baseDragSupersonic.csv")
        # ESDU B.S.02.03.02 - boattail drag coefficient (wave drag)
        #Cd_beta = (D / (2 * lad)) ** 2 * getValue3D(dbd**2, boattailAngle, M, "boattailDragSupersonic.csv")
        Cd_beta = 0

    F = getValue2D(M, alpha, "angleDrag.csv")  # angle of attack effect on base drag
    # TODO: Use ESDU 02012 for the effect of the jet - ignore this until we can get exhaust temperature and pressure

    Cdw = 0  # wave drag - use ESDU B.S.02.03.01
    Cdwv = 0  # viscous form drag?
    Cd = Cdp + Cdw + Cdwv + F * (Cdb + Cd_beta) + Cf

    return Cn, Cm, xcp, Mq, Cd, Cdp, Cdw, Cdwv, F * (Cdb + Cd_beta), Cf
