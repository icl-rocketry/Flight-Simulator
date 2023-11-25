import numpy as np

class Materials:
    def __init__(self,type):
        """
        materialType: Material Name
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
            self.density = 0
            self.youngsModulus = 0
        elif self.materialType.lower() == "gfrp":
            self.density = 0
            self.youngsModulus = 0
        else:
            raise ValueError("Mateiral Not Found")
    

