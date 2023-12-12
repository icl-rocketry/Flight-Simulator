import numpy as np
from Component.Rocket import Rocket

# plan for this module is to create a Cd vs Mach and Re curve from scratch

# Total Drag = Nose Cone Drag + Body Tube Drag + Base Drag + Fin Drag
# Other forms of drag include: Interference drag

# If we create CD and CL 2D array for angle of attack for each aero surface


class dragCalculator:
    def __init__(self) -> None:
        Rocket.cd

    def beta(_,mach): # _ parameter means any parameter or "i dont care"
        """
        Correction factor for compressibility
        """
        if mach < 0.8:
            return np.sqrt(1 - mach**2)
        elif mach < 1.1:
            return np.sqrt(1 - 0.8**2)
        else:
            return np.sqrt(mach**2 - 1)
