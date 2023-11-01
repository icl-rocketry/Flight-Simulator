import Atmosphere
import numpy as np

if __name__ == "__main__":
    r = Atmosphere.Environment()
    n = 50
    a = np.linspace(1000,11000,n)
    for ii in range(0,n-1):
        r.atmosphere(a[ii])