import Atmosphere2 as a
import Component.Rocket as r
import numpy as np

# FUNCTIONS #
def initialiseRocket():
    rocket = r.Rocket(rocketLength=5,
                      rocketRadius=0.2)
    
    nose = rocket.addNose(type="Haack",
                            length=0.6,
                            noseRadius=rocket.rocketRadius,
                            material="CFRP",
                            thickness=0.002) 
    
    bodyTube = rocket.addBodyTube(length=4,
                                    radius=rocket.rocketRadius,
                                    thickness=0.002,
                                    position=0.7,
                                    material="cfrp")
    
    boatTail = rocket.addBoatTail(upperRadius=rocket.rocketRadius, 
                                    lowerRadius=rocket.rocketRadius*0.8, 
                                    length=0.35, 
                                    thickness=0.002, 
                                    position=3.5, 
                                    material='gfrp')
    
    #fins = nimbussy.addFins()

    return rocket, nose, bodyTube, boatTail


# MAIN SCRIPT #
if __name__ == "__main__":
    nimbussy, nose, bodyTube, boatTail = initialiseRocket() # See function above

    # IDEALLY IN THE FUTURE, [DYANMICS MODEL HERE]
    # while...

    # Test Angles of Attack 
    aoa = np.deg2rad(np.linspace(0,5,20))

    # To calculate CL and CD
    #for surface, cnPos in r.aerosurfacelist:
        #cl = aerosurfacelist(aoa, mach)
    
    # PLOTS HERE
    for surface in nimbussy.aerodynamicSurfacesList:
        print(surface.cgPos)


