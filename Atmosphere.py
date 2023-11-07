import numpy as np
from matplotlib import pyplot as plt

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

        #Wind Parameters
        self.windSpeed = 8
        self.windDirection = np.pi/4
        self.windTurbulence = 0.15 #std dev / mean
        self.deltaTime = 0.01
        self.sampleLength = 100 #if this is small (<10), the normalisation fails
        self.frequency = 20 #Hz
        self.U = self.windSpeed * np.cos(self.windDirection)
        self.V = self.windSpeed * np.sin(self.windDirection)
        self.sigma_u = self.U * self.windTurbulence
        self.sigma_v = self.V * self.windTurbulence
        self.sigma_w = 0.5
        self.time_series = np.arange(0, self.sampleLength-self.deltaTime, self.deltaTime)
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

    # the spectral energy density of the wind velocity fluctuations - von Karman
    def Su(self, h, f): #vertical
        self.Lu = h * 3.281 / (0.177 + 0.000823 * h * 3.281) ** 1.2
        return 4 * self.sigma_u ** 2 * self.Lu / self.U * (1 + 1.339 * (self.Lu * 2 * np.pi * f / self.U)) ** -5/3
    
    def Sv(self, h, f): #lateral
        self.Lv = h * 1.6405 / (0.177 + 0.000823 * h * 3.281) ** 1.2
        return 4 * self.sigma_v ** 2 * self.Lv / self.V * (1 + 8/3 * (2.678 * self.Lv * 2 * np.pi * f / self.U)) / ((1 + 2.678 * (self.Lv * 2 * np.pi * f / self.V)) ** 11/3)
    
    def Sw(self, h, f): #lateral
        self.Lw = h * 1.6405
        return 4 * self.sigma_w ** 2 * self.Lw / self.windSpeed * (1 + 8/3 * (2.678 * self.Lw * 2 * np.pi * f / self.U)) / ((1 + 2.678 * (self.Lw * 2 * np.pi * f / self.windSpeed)) ** 11/3)

    def wind(self, altitude):

        #setup parameters
        samples = int(self.sampleLength // self.deltaTime)
        i = np.arange(samples)
        f = i / (self.deltaTime * samples) / 20 #openRocket uses 20 Hz as the frequency

        #compute wind speeds using inverse FFT
        magnitude_u = samples * np.sqrt(self.Su(altitude, f))
        magnitude_v = samples * np.sqrt(self.Sv(altitude, f))
        magnitude_w = samples * np.sqrt(self.Sw(altitude, f))
        phase = 2 * np.pi * np.random.randn(samples)
        FFT_u = magnitude_u * np.exp(1j * phase)
        FFT_v = magnitude_v * np.exp(1j * phase)
        FFT_w = magnitude_w * np.exp(1j * phase)
        self.Unu = np.real(np.fft.ifft(FFT_u))
        self.Unv = np.real(np.fft.ifft(FFT_v))
        self.Unw = np.real(np.fft.ifft(FFT_w))

        # scale to match the mean and standard deviation
        self.Unu = self.Unu * self.sigma_u / np.std(self.Unu) + self.U
        self.Unv = self.Unv * self.sigma_v / np.std(self.Unv) + self.V
        self.Unw = self.Unw * self.sigma_w / np.std(self.Unw)

