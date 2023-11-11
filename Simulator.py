import Atmosphere2 as a
import numpy as np

if __name__ == "__main__":
    r = a.Environment()
    n = 50
    a = np.linspace(1000,11000,n)
    for ii in range(0,n-1):
        r.wind(a[ii])