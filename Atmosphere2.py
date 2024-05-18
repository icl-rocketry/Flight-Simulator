# general imports
import numpy as np
from matplotlib import pyplot as plt

# forecasting API imports, download for new users
import requests_cache
import openmeteo_requests
from retry_requests import retry


class Environment:
    def __init__(self):
        """
        ----------------------------------------------------------------------
        a: Speed of sound (m/s)
        g: Acceleration due to gravity (m/s^2)
        R: Air gas constant (J/kg.K)
        gamma: Air constant (dimensionless)
        pressure: Atmospheric pressure (Pa)
        density: Air density (kg/m^3)
        temperature: Atmospheric temperature (K)
        ----------------------------------------------------------------------
        deltaTime: Time step (s). Affects turbulence as a lower timestep allows for higher frequencies in the PSD
        sampleLength: Length of time series (s). The gust spectra are then recalculated when this time elapses as they are altitude-dependent
        totalLength: Total length of time series (s). This is the time of the wind model above the boundary layer
        userWind: Whether to use the user-defined wind model or the Open-Meteo API
        modelAtmo: Whether to use the ISA atmospheric model or the Open-Meteo API
        ----------------------------------------------------------------------
        windSpeed: Wind speed (m/s) at 10m altitude. ONLY USED IF USER-DEFINED WIND MODEL IS USED
        windDirection: Wind direction (degrees clockwise from North). ONLY USED IF USER-DEFINED WIND MODEL IS USED
        turbulenceIntensity: Turbulence intensity (std dev / mean) of the wind speed
        z1: Height of boundary layer (m)
        ----------------------------------------------------------------------
        earthMass: Mass of Earth (kg)
        earthRadius: Radius of Earth (m)
        G: Universal Gravitational Constant (m^3 kg^-1 s^-2)
        latitude: Latitude of launch site (degrees)
        longitude: Longitude of launch site (degrees)
        ----------------------------------------------------------------------
        rocketVelocity: Rocket velocity (m/s). Modelled as constant to assess gusts at various altitudes
        altitudeLast: Placeholder for storing last altitude (m)
        ----------------------------------------------------------------------
        """
        # Atmospheric Parameters
        self.g = 9.80665  # Note: NOT Negative
        self.R = 287
        self.gamma = 1.4
        self.pressure = 101325
        self.density = 1.225
        self.temperature = 288.15
        self.a = np.sqrt(self.gamma * self.R * self.temperature)
        self.mu = 1.458 * 10**-6 * (self.temperature**1.5) / (self.temperature + 110.4)

        # Wind Simulation Parameters
        self.deltaTime = 0.025  # openRocket recommends 0.05s. Can be different from the simulation timestep
        self.sampleLength = 20
        self.totalLength = 20
        self.userWind = False
        self.modelAtmo = False

        # Wind Parameters
        self.windSpeed = 0  # at 10m altitude. TODO: allow for more complex user-defined wind models
        self.windDirection = np.pi / 3
        self.turbulenceIntensity = 0.15  # typically 0.1-0.2
        self.z0 = 0.001
        self.z1 = 1000 * (self.z0**0.18)

        # Planet Parameters
        self.earthMass = 5.97e24
        self.earthRadius = 6.371e6
        self.G = 6.67e-11
        self.latitude = 55.05
        self.longitude = -2.55

        # Rocket Parameters
        self.altitudeLast = 0
        self.rocketVelocity = 100  # this should come from the rocket class but there isn't one yet

        # pressure list
        self.pressureList = [
            1000,
            975,
            950,
            925,
            900,
            850,
            800,
            700,
            600,
            500,
            400,
            300,
            250,
            200,
            150,
            100,
            70,
            50,
            30,
        ]  # hPa

        self.params = {
            "latitude": self.latitude,  # this is near EuRoC
            "longitude": self.longitude,
            "hourly": [
                "temperature_1000hPa",
                "temperature_975hPa",
                "temperature_950hPa",
                "temperature_925hPa",
                "temperature_900hPa",
                "temperature_850hPa",
                "temperature_800hPa",
                "temperature_700hPa",
                "temperature_600hPa",
                "temperature_500hPa",
                "temperature_400hPa",
                "temperature_300hPa",
                "temperature_250hPa",
                "temperature_200hPa",
                "temperature_150hPa",
                "temperature_100hPa",
                "temperature_70hPa",
                "temperature_50hPa",
                "temperature_30hPa",
                "windspeed_1000hPa",
                "windspeed_975hPa",
                "windspeed_950hPa",
                "windspeed_925hPa",
                "windspeed_900hPa",
                "windspeed_850hPa",
                "windspeed_800hPa",
                "windspeed_700hPa",
                "windspeed_600hPa",
                "windspeed_500hPa",
                "windspeed_400hPa",
                "windspeed_300hPa",
                "windspeed_250hPa",
                "windspeed_200hPa",
                "windspeed_150hPa",
                "windspeed_100hPa",
                "windspeed_70hPa",
                "windspeed_50hPa",
                "windspeed_30hPa",
                "winddirection_1000hPa",
                "winddirection_975hPa",
                "winddirection_950hPa",
                "winddirection_925hPa",
                "winddirection_900hPa",
                "winddirection_850hPa",
                "winddirection_800hPa",
                "winddirection_700hPa",
                "winddirection_600hPa",
                "winddirection_500hPa",
                "winddirection_400hPa",
                "winddirection_300hPa",
                "winddirection_250hPa",
                "winddirection_200hPa",
                "winddirection_150hPa",
                "winddirection_100hPa",
                "winddirection_70hPa",
                "winddirection_50hPa",
                "winddirection_30hPa",
                "geopotential_height_1000hPa",
                "geopotential_height_975hPa",
                "geopotential_height_950hPa",
                "geopotential_height_925hPa",
                "geopotential_height_900hPa",
                "geopotential_height_850hPa",
                "geopotential_height_800hPa",
                "geopotential_height_700hPa",
                "geopotential_height_600hPa",
                "geopotential_height_500hPa",
                "geopotential_height_400hPa",
                "geopotential_height_300hPa",
                "geopotential_height_250hPa",
                "geopotential_height_200hPa",
                "geopotential_height_150hPa",
                "geopotential_height_100hPa",
                "geopotential_height_70hPa",
                "geopotential_height_50hPa",
                "geopotential_height_30hPa",
                "wind_speed_10m",
                "wind_direction_10m",
                "temperature_2m",
            ],
            "wind_speed_unit": "ms",
            "timezone": "auto",
            "forecast_days": 1,
        }

    def atmosphere(self, altitude, rocketHeightDifference):
        # Basic ISA Atmospheric Model for now, assume constant g (geopotential)

        # Finding Lapse Rate (K/m)
        if altitude < -1:
            lapseRate = 0
            # raise Exception("Rocket disappeared.")
        elif 0 <= altitude <= 11000:
            lapseRate = 0.0065
        elif 11000 < altitude <= 20000:
            lapseRate = 0
        elif 20000 < altitude <= 32000:
            lapseRate = -0.001
        else:
            lapseRate = 0

        if self.modelAtmo:
            self.temperature -= lapseRate * rocketHeightDifference  # Temperature variation
            if 0 <= altitude <= 32000:
                self.pressure = self.pressure * (1 - lapseRate / 288.15) ** (
                    self.g / (self.R * lapseRate)
                )  # Pressure variation
                self.density = self.density * (1 - lapseRate / 288.15) ** (
                    self.g / (self.R * lapseRate) - 1
                )  # Density variation
            else:
                self.pressure = 0
                self.density = 0

        else:
            # interpolate temperature and pressure from the forecast
            self.temperature, self.pressure = self.getForecastProperties(altitude)
            self.density = self.pressure / (self.R * self.temperature)

    def getTurbulence(self, altitude, sampleLength):
        # uses the Kaimal spectrum, returns a time series of turbulence. Only needs to switch at z=30m
        # setup parameters (IEC 1999)
        freq = 20  # max. frequency (Hz)
        maxFreq = freq**2 / self.deltaTime
        uBar = []  # set up IFFT inputs
        vBar = []
        wBar = []

        # calculate the length scales
        if altitude < self.z1:  # in boundary layer
            Lu = 280 * (altitude / self.z1) ** 0.35
            Lv = 140 * (altitude / self.z1) ** 0.38
            Lw = 140 * (altitude / self.z1) ** 0.45
        else:  # when z1 is exceeded, we must recalculate the entire time series
            Lu = 280
            Lv = 140
            Lw = 140

        # calculate the PSD and set it up for IFFT
        self.timeSeries = np.arange(0, sampleLength, self.deltaTime)
        for f in np.linspace(0, maxFreq, len(self.timeSeries) // 2):
            PSDu = (4 * Lu / self.rocketVelocity) / ((1 + 70.8 * (f * Lu / self.rocketVelocity) ** 2) ** (5 / 6))
            PSDv = (
                (4 * Lv / self.rocketVelocity)
                * (1 + 755.2 * (f * Lv / self.rocketVelocity) ** 2)
                / ((1 + 283.2 * (f * Lv / self.rocketVelocity) ** 2) ** (11 / 6))
            )
            PSDw = (
                (4 * Lw / self.rocketVelocity)
                * (1 + 755.2 * (f * Lw / self.rocketVelocity) ** 2)
                / ((1 + 283.2 * (f * Lw / self.rocketVelocity) ** 2) ** (11 / 6))
            )
            uBar.append(np.sqrt(PSDu))
            vBar.append(np.sqrt(PSDv))
            wBar.append(np.sqrt(PSDw))

        # duplicate and flip the PSDs for the negative frequencies (IFFT requires this)
        uBar = np.concatenate((uBar, np.flip(uBar)))
        vBar = np.concatenate((vBar, np.flip(vBar)))
        wBar = np.concatenate((wBar, np.flip(wBar)))

        # add a random phase to each frequency
        uBar = uBar * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(uBar)))
        vBar = vBar * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(vBar)))
        wBar = wBar * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(wBar)))

        # generate the time series
        u = np.real(np.fft.ifft(uBar))
        v = np.real(np.fft.ifft(vBar))
        w = np.real(np.fft.ifft(wBar))

        # scale to match the standard deviation of 1
        if np.std(u) != 0:
            u = u / np.std(u)
            v = v / np.std(v)
            w = w / np.std(w)

        # return the time series
        return np.transpose(u), np.transpose(v), np.transpose(w)

    def getForecast(self):  # using the Open-Meteo API
        # get the upper level winds from the Open-Meteo API if a user-defined wind model is not used
        # otherwise, just use the simple user-defined wind model like openRocket does (boring!)
        if self.userWind:
            data = np.array([[0, self.windSpeed, self.windDirection], [22000, self.windSpeed, self.windDirection]])
        else:
            data = []
            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            # Make sure all required weather variables are listed here
            # The order of variables in hourly or daily is important to assign them correctly below
            url = "https://api.open-meteo.com/v1/forecast"
            responses = openmeteo.weather_api(url, params=self.params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]
            print("Forecast fetched successfully")

            # Process hourly data. The order of variables needs to be the same as requested.
            hourly = response.Hourly()
            if hourly is None:
                print("No hourly data available")
            else:
                n = 19  # n is the total number of pressure levels from the API
                data = []
                surfaceValues = [
                    0,
                    hourly.Variables(76).ValuesAsNumpy()[0],  # type: ignore
                    hourly.Variables(77).ValuesAsNumpy()[0],  # type: ignore
                    hourly.Variables(78).ValuesAsNumpy()[0] + 273.15,  # type: ignore
                ]
                data.append(surfaceValues)  # add the surface values to the data array
                for i in range(n):  # Loop through all pressure levels
                    geopotential_height = hourly.Variables(i + 57).ValuesAsNumpy()  # type: ignore
                    windspeed = hourly.Variables(i + 19).ValuesAsNumpy()  # type: ignore
                    winddirection = hourly.Variables(i + 38).ValuesAsNumpy()  # type: ignore
                    temperature = hourly.Variables(i).ValuesAsNumpy()  # type: ignore
                    data.append([geopotential_height[0], windspeed[0], winddirection[0], temperature[0] + 273.15])  # type: ignore

        self.upperLevelWinds = np.array(data[1:])  # TODO: fix this as it shouldn't be like this i dont think

    def getUpperLevelWinds(self, altitude):
        # returns the mean wind speed and direction at a given altitude using the forecast/user-defined wind model
        if altitude > max(self.upperLevelWinds[:, 0]):
            return 0, 0
        else:
            # interpolate speed
            speed = np.interp(altitude, self.upperLevelWinds[:, 0], self.upperLevelWinds[:, 1])
            # interpolate direction
            direction = np.interp(altitude, self.upperLevelWinds[:, 0], self.upperLevelWinds[:, 2])
            U = speed * np.sin(direction * np.pi / 180)
            V = speed * np.cos(direction * np.pi / 180)
            return U, V

    def getForecastProperties(self, altitude):  # using the Open-Meteo API
        # returns the forecast temperature and pressure at a given altitude. Forecast must be fetched first
        if altitude < min(self.upperLevelWinds[:, 0]):
            return self.upperLevelWinds[0, 3], self.pressureList[0] * 100
        elif altitude > max(self.upperLevelWinds[:, 0]):
            return 288.15, 0
        else:
            # interpolate temperature
            temperature = np.interp(altitude, self.upperLevelWinds[:, 0], self.upperLevelWinds[:, 3])
            # interpolate pressure
            pressureList = [
                1000,
                975,
                950,
                925,
                900,
                850,
                800,
                700,
                600,
                500,
                400,
                300,
                250,
                200,
                150,
                100,
                70,
                50,
                30,
            ]  # hPa
            # sometimes the code doesnt work because of the weather
            # as a result, the interpolation fails
            if len(self.upperLevelWinds[:, 0]) != len(pressureList):
                # pad the shorter array with leading zeros
                pressureList = np.pad(
                    pressureList, (len(self.upperLevelWinds[:, 0]) - len(pressureList), 0), "constant"
                )
            pressure = np.interp(altitude, self.upperLevelWinds[:, 0], pressureList) * 100  # Pa
            return temperature, pressure


# Test Code
if __name__ == "__main__":
    env = Environment()
    env.getForecast()  # this must be done before the simulation starts - could do it in __init__
    uList = []
    vList = []
    wList = []
    uM = []
    vM = []
    tList = []
    t = 0
    alt = 2  # start at non-zero value to avoid divide by zero errors. This is fine as we will simulate a launch rail
    steps = 0
    stepsSince = 0
    finalCalc = False  # stops turbulence calc once the length scale is constant (above z1)
    # the following dont really need to be here but python throws an error if they are unbound
    u = []
    v = []
    w = []

    # plot the wind over time, while altitude changes (that makes it really complicated)
    # this mess will most likely go in the simulation class or similar but it's here for testing purposes now
    # TODO: see if theres a nice way to smooth the discontinuities from length scale recalculation which doesnt remove the high-frequency content
    while alt < 600:
        t += env.deltaTime
        # every sampleLength seconds, recalculate the turbulence
        if steps % (env.sampleLength / env.deltaTime) == 0:
            if alt < env.z1:
                u, v, w = env.getTurbulence(alt, env.sampleLength)  # get the new time series of wind
                stepsSince = 0
            elif not finalCalc:  # get the new (long-lasting) time series of wind
                u, v, w = env.getTurbulence(alt, env.totalLength)
                stepsSince = 0
                finalCalc = True

        tList.append(t)
        uMean, vMean = env.getUpperLevelWinds(alt)
        uM.append(uMean)
        vM.append(vMean)
        try:
            totalSpeed = np.sqrt(uMean**2 + vMean**2)
            uTotal = uMean + (u[stepsSince] * env.turbulenceIntensity * totalSpeed)
            vTotal = vMean + (v[stepsSince] * env.turbulenceIntensity * totalSpeed)
            wTotal = w[stepsSince] * totalSpeed * 0.1
        except IndexError:
            print(
                "The time for which the turbulence is generated 'totalLength' is too short, increase it to cover the entire flight."
            )
            break
        alt += env.rocketVelocity * env.deltaTime
        steps += 1
        stepsSince += 1
        env.atmosphere(alt, env.rocketVelocity * env.deltaTime)
        uList.append(uTotal)
        vList.append(vTotal)
        wList.append(wTotal)

    # remove the last items of u,v,w, to make them the same length as t as the last step will be incomplete

    # plot the wind
    try:
        plt.plot(tList, uList, label="u")
        plt.plot(tList, vList, label="v")
        plt.plot(tList, wList, label="w")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Wind Speed (m/s)")
        plt.title("Wind Speed vs Time")
        plt.show()
    except ValueError:
        print("The simulation ran into an error and cannot plot the data.")
