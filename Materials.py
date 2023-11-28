import numpy as np

class Materials:
    def __init__(self,type):
        """
        materialType: Material name
        density: Density of material (kg/m^3)
        youngModulus: Young Modulus of material (Pa)
        """
        self.materialType = type

        # Physical properties
        self.density = None
        self.youngsModulus = None

        self.setProperties()

    # Set properties of material dependent on type
    def setProperties(self):
        if self.materialType.lower() == "cfrp":
            self.density = 1.27 # g/cm^3
            self.youngsModulus = 43.9 # GPa
        elif self.materialType.lower() == "gfrp":
            self.density = 0.4/0.3 # kgsm (/thickness = 0.3)
            self.youngsModulus = 72.5 # GPa
        else:
            raise ValueError("Material Not Found")
    

