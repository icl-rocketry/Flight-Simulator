import numpy as np

class finSet:
    def __init__(self, span, rootChord, tipChord, sweep, thickness, material, rootLocation, aerofoil):
        '''
        Parameters:
        span: span of the fin
        rootChord: root chord of the fin
        tipChord: tip chord of the fin
        sweep: sweep of the fin
        thickness: thickness of the fin
        material: material of the body tube (CFRP, GFRP, Aluminum)
        rootLocation: location of the root chord (from the nose cone)
        aerofoil: aerofoil of the fin - 0 is a flat plate, otherwise it's a list of x,y coordinates
        '''
        self.span = span
        self.rootChord = rootChord
        self.tipChord = tipChord
        self.sweep = sweep
        self.thickness = thickness
        self.material = material.lower() # make sure it's lowercase for comparisons
        self.rootLocation = rootLocation
        self.aerofoil = aerofoil

        # restrict materials to those in the list
        if self.material not in ["CFRP", "GFRP", "Aluminum"]:  # I can't remember what was in here
            raise ValueError("Invalid material")
        

    