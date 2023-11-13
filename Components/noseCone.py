import numpy as np


class noseCone:
    def __init__(self, length, radius, shape, shapeParameter, thickness, material):
        """
        Parameters:
        length: length of the nose cone
        radius: radius of the nose cone
        shape: shape of the nose cone (ogive, haack)
        shapeParameter: parameter of the shape (from 0 to 1)
        thickness: thickness of the nose cone
        material: material of the nose cone (CFRP, GFRP, Aluminum)
        """
        self.length = length
        self.radius = radius
        self.shape = shape.lower()  # make sure it's lowercase for comparisons
        self.shapeParameter = shapeParameter
        self.thickness = thickness
        self.material = material.lower()  # make sure it's lowercase for comparisons

        # restrict shapes to those in the list
        if self.shape not in ["ogive", "haack"]:  # von Karman is just a special case of Haack
            raise ValueError("Invalid shape")

        # restrict materials to those in the list
        if self.material not in ["CFRP", "GFRP", "Aluminum"]:  # I can't remember what was in here
            raise ValueError("Invalid material")

    # Barrowman Calcs
    def getBarrowman(self, alpha):  # extended barrowman comes from combining parts, not each part
        """
        Cn is the 'weighting' of the Cp location
        Xn is the location of the Cp"""
        Cn = 2  # this is a property of all nose cones
        if self.shape == "ogive":
            f = 1 / self.shapeParameter  # to be consistent with literature
            lam = np.sqrt(2 * f - 1)
            realOgiveXn = self.radius * ((f**2 - lam**2 / 3) * lam - f**2 * (f - 1) * np.arcsin(lam / f))
            # the actual ogive might be squashed, so we need to scale it
            realOgiveLength = np.sqrt((2 * lam + 1) * self.radius**2)
            Xn = realOgiveXn * self.length / realOgiveLength

        else:  # the only other option is haack
            Xn = self.length / 2 * (1 + 3 * self.shapeParameter / 8)  # relative to tip of nose cone

        return Cn, Xn
