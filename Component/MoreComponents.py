import numpy as np
import Component.Materials as mat
from utils.function import Function

# LIST OF MORE COMPONENTS
# 1. Mass Component - Adds mass to rocket, duh
# 2. Clamp band - Holds the rocket together before it explodes
# 3. Stringer - Stinky airframe component  


# This is so redundant but whatever
class MassComponent():
    def __init__(self, mass, pos):
        self.mass = mass
        self.pos = pos
        return None

    
class ClampBand():
    def __init__(self) -> None:
        pass


class Stringers():
    def __init__(self) -> None:
        pass