import numpy as np

class bodyTube:
    def __init__(self, length, radius, thickness, material):
        '''
        Parameters:
        length: length of the body tube
        radius: radius of the body tube
        thickness: thickness of the body tube
        material: material of the body tube (CFRP, GFRP, Aluminum)
        '''
        self.length = length
        self.radius = radius
        self.thickness = thickness
        self.material = material.lower() # make sure it's lowercase for comparisons

        # restrict materials to those in the list
        if self.material not in ["CFRP", "GFRP", "Aluminum"]:  # I can't remember what was in here
            raise ValueError("Invalid material")
        

    