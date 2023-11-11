import numpy as np

class boattail:
    def __init__(self, length, upperRadius, lowerRadius, thickness, material):
        '''
        Parameters:
        length: length of the boattail
        upperRadius: radius of the top of the boattail (same as body tube)
        lowerRadius: radius of the bottom of the boattail
        thickness: thickness of the boattail
        material: material of the boattail (CFRP, GFRP, Aluminum)
        '''
        self.length = length
        self.upperRadius = upperRadius
        self.lowerRadius = lowerRadius
        self.thickness = thickness
        self.material = material.lower() # make sure it's lowercase for comparisons

        # restrict materials to those in the list
        if self.material not in ["CFRP", "GFRP", "Aluminum"]:  # I can't remember what was in here
            raise ValueError("Invalid material")
        

    