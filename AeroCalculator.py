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

    val = (interpAbove([x, y]) * (zValues[aboveIndex] - z) + interpBelow([x, y]) * (z - zValues[belowIndex])) / (
        zValues[aboveIndex] - zValues[belowIndex]
    )
    return val[0]

def getAeroParams(M, alpha, T, a, Rocket):
    '''Gets aero parameters of the rocket (force and moment)
    M:      Mach number
    alpha:  angle of attack in degrees
    T:      temperature in K
    a:      speed of sound in m/s
    Rocket: rocket object used to get the geometry
    
    outputs:
    Cn:     normal force coefficient
    Cm:     pitching moment coefficient
    xcp:    center of pressure in m from the nose cone tip
    ...other derivatives?'''

    # nose cone geometry from ESDU 77028
    Cvf = 0.5  # for haack
    Cpl = 0.651  # for haack
    Ccl = 1.239  # for haack

    lfd = 3  # lf/D, length of nose cone to rocket diameter ratio
    lcd = 10  # lc/D, length of rocket body to rocket diameter ratio
    lad = 1  # la/D, length of boattail to rocket diameter ratio
    dbd = 0.8  # db/D, boattail diameter ratio
    L = 4.7  # length of rocket in m

    # get flow parameters
    ld = lfd + lcd + lad
    mu = 1.458 * 10**-6 * (T**1.5) / (T + 110.4)
    Re = 1.225 * a * (L/ld) / mu
    logR = log10(Re)  # used as the ESDU graph is logaritmic

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
    delta_Cm0a_d = -4 * dStarL * ld ** 2
    delta_Cm0a_f = -2 * Cf * ld ** 2
    Cm0a = k * Cna_l * (CmCn + delta_CmCn) + delta_Cm0a_d + delta_Cm0a_f  # about the nose

    # Cnc from ESDU 89014
    Cnc = getValue2D(M, alpha, "vortexGeneration.csv")

    # convert coefficients to forces and moments using ESDU 89014
    alpha = radians(alpha)
    Cn = Cna * sin(alpha) * cos(alpha) + 4 / pi * ld * Cpl * Cnc
    Cm0 = Cm0a * sin(alpha) * cos(alpha) - 2 / pi * ld ** 2 * Cpl * Ccl * Cnc

    # modifications due to boattail, from ESDU 87033
    A = -1.3 * lad + 6.35 * lad - 7.85
    if M >= 0 and M < 1:
        delta_Cna = -2 * (1 - dbd**2)
    else:
        delta_Cna = -(0.6 + 1.4 * exp(A * (M - 1) ** 0.8)) * (1 - dbd**2)
    delta_Cdc = -(2 + tanh(1.5 * M * sin(alpha) - 2.4)) * lad * (1 - dbd)
    delta_Cn = delta_Cna * sin(alpha) * cos(alpha) * (1 - (sin(alpha) ** 0.6)) + delta_Cdc * (sin(alpha) ** 2)
    Cn += delta_Cn
    if alpha == 0: # avoid division by zero
        Cm = 0
        xcp = 0
    else:
        Cm = Cn * (ld/2 + Cm0/Cn) # pitching moment about midpoint
        Cm -= 0.5 * delta_Cn * (ld - lad) # again about the midpoint
        xcp = ld/2 - Cm/Cn
    return Cn, Cm, xcp