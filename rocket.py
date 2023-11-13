from Components import bodyTube, boattail, noseCone, finSet

class Rocket:
    def __init__(self, name):
        self.name = name

    def addNoseCone(self, length, radius, shape, shapeParameter, thickness, material):
        """
        Parameters:
        length: length of the nose cone
        radius: radius of the nose cone
        shape: shape of the nose cone (ogive, haack)
        shapeParameter: parameter of the shape (from 0 to 1)
        thickness: thickness of the nose cone
        material: material of the nose cone (CFRP, GFRP, Aluminum)
        """
        self.noseCone = noseCone(length, radius, shape, shapeParameter, thickness, material)

    def addBodyTube(self, length, radius, thickness, material):
        """
        Parameters:
        length: length of the body tube
        radius: radius of the body tube
        thickness: thickness of the body tube
        material: material of the body tube (CFRP, GFRP, Aluminum)
        """
        try:
            self.bodyTube = bodyTube(length, radius, thickness, material, self.noseCone.length)
        except AttributeError:
            print("Body tube position could not be calculated. You need to add a nose cone first.")

    def addBoattail(self, length, upperRadius, lowerRadius, thickness, material):
        """
        Parameters:
        length: length of the boattail
        upperRadius: radius of the top of the boattail (same as body tube)
        lowerRadius: radius of the bottom of the boattail
        thickness: thickness of the boattail
        material: material of the boattail (CFRP, GFRP, Aluminum)
        """
        try:
            newLength = self.noseCone.length + self.bodyTube.length
            self.boattail = boattail(length, upperRadius, lowerRadius, thickness, material, newLength)
        except AttributeError:
            print("Boattail position could not be calculated. You need to add a nose cone and body tube first.")

    def addFinSet(self, span, rootChord, tipChord, sweep, thickness, material, rootLocation, aerofoil):
        """
        Parameters:
        span: span of the fin
        rootChord: root chord of the fin
        tipChord: tip chord of the fin
        sweep: sweep of the fin
        thickness: thickness of the fin
        material: material of the body tube (CFRP, GFRP, Aluminum)
        rootLocation: location of the root chord (from the nose cone)
        aerofoil: aerofoil of the fin - 0 is a flat plate, otherwise it's a list of x,y coordinates
        """
        self.finSet = finSet(span, rootChord, tipChord, sweep, thickness, material, rootLocation, aerofoil)

        # just so I remember, the extended part of Barrowman applies when combining parts, not to each part