import numpy as np

class Environment:
    def __init__(self):
        '''
        ----------------------------------------------------------------------
        a: Speed of sound (m/s)
        g: Acceleration due to gravity (m/s^2)
        R: Air gas constant (J/kg.K)
        gamma: Air constant ()
        pressure: Atmospheric pressure (Pa)
        density: Air density (kg/m^3)
        temperature: Atmospheric temperature (K)
        ----------------------------------------------------------------------
        earthMass: Mass of Earth (kg)
        earthRadius: Radius of Earth (m)
        G: Universal Gravitational Constant (forgor units)
        ----------------------------------------------------------------------
        altitudeLast: Placeholder for storing last altitude (m)
        ----------------------------------------------------------------------
        '''
        #Atmospheric Parameters
        self.a = 343
        self.g = 9.80665 #Note: NOT Negative
        self.R = 287 
        self.gamma = 1.4
        self.pressure = 101325
        self.density = 1.225
        self.temperature = 288.15

        #Planet Parameters
        self.earthMass = 5.97e24
        self.earthRadius = 6.36e6
        self.G = 6.67e-11

        #Rocket Parameters
        self.altitudeLast = 0

    def atmosphere(self, altitude):
        #Basic ISA Atmospheric Model for now, assume constant g (geopotential)
        #Model ONLY applicable for troposphere 
        
        #Finding Lapse Rate (K/m)
        if altitude < 0:
            raise Exception("Rocket disappeared.")
        elif (0 <= altitude <= 11000):
            lapseRate = 0.0065

        d_altitude = altitude - self.altitudeLast #Compute altitude difference
        self.altitudeLast = altitude #Store previous altitude

        self.temperature -= lapseRate * d_altitude #Temperature variation
        self.pressure = self.pressure * (1-lapseRate/288.15)**(self.g/(self.R*lapseRate)) #Pressure variation
        self.density = self.density * (1-lapseRate/288.15)**(self.g/(self.R*lapseRate)-1) #Density variation


    def gust(self):
        #Some form of sussy gust model here, prob from OpenRocket Documentation rn
        amongus = 1