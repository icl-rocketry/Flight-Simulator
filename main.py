import Atmosphere2 as a
import Component.Rocket as r
import numpy as np

# FUNCTIONS #
def initialiseRocket():
    rocket = r.Rocket(5,0.2)
    nose = rocket.addNose(type="Haack",
                            length=0.6,
                            noseRadius=rocket.rocketRadius,
                            material="CFRP",
                            thickness=0.002,
                            ) 
    
    bodyTube = rocket.addBodyTube(length=4,
                                    radius=rocket.rocketRadius,
                                    thickness=0.002,
                                    material="cfrp")
    
    boatTail = rocket.addBoatTail(upperRadius=rocket.rocketRadius, 
                                    lowerRadius=rocket.rocketRadius*0.8, 
                                    length=0.35, 
                                    thickness=0.002, 
                                    boatTailPos=3.5, 
                                    material='cfrp')
    return rocket

# MAIN SCRIPT #
if __name__ == "__main__":
    nimbussy = initialiseRocket()


    #fins = nimbussy.addFins()

    # IDEALLY IN THE FUTURE, [DYANMICS MODEL HERE]
    # while...


    # PLOTS HERE

    # Test Angles of Attack 
    aoa = np.deg2rad(np.linspace(0,5,20)) 


