import numpy as np

class noseCone:
    def __init__(self, length, radius, shape, shapeParameter, thickness, material):
        '''
        Parameters:
        length: length of the nose cone
        radius: radius of the nose cone
        shape: shape of the nose cone (ogive, haack)
        shapeParameter: parameter of the shape (from 0 to 1)
        thickness: thickness of the nose cone
        material: material of the nose cone (CFRP, GFRP, Aluminum)
        '''
        self.length = length
        self.radius = radius
        self.shape = shape.lower() # make sure it's lowercase for comparisons
        self.shapeParameter = shapeParameter
        self.thickness = thickness
        self.material = material.lower() # make sure it's lowercase for comparisons

        # restrict shapes to those in the list
        if self.shape not in ["ogive", "haack"]:  # von Karman is just a special case of Haack
            raise ValueError("Invalid shape")

        # restrict materials to those in the list
        if self.material not in ["CFRP", "GFRP", "Aluminum"]:  # I can't remember what was in here
            raise ValueError("Invalid material")
        

    