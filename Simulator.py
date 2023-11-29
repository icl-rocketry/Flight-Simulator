import Atmosphere2 as a
import Rocket2 as r
import numpy as np

if __name__ == "__main__":
    # Initialising rocket -> can be done in separate file?
    nimbussy = r.Rocket(5,0.2)
    nose = nimbussy.addNose(type="Haack",
                            length=0.6,
                            noseRadius=nimbussy.rocketRadius,
                            material="CFRP",
                            thickness=0.002,
                            ) 
    
    bodyTube = nimbussy.addBodyTube(length=4,
                                    radius=nimbussy.rocketRadius,
                                    thickness=0.002,
                                    material="cfrp")
    
    boatTail = nimbussy.addBoatTail(upperRadius=nimbussy.rocketRadius, 
                                    lowerRadius=nimbussy.rocketRadius*0.8, 
                                    length=0.35, 
                                    thickness=0.002, 
                                    boatTailPos=3.5, 
                                    material='cfrp')

    #fins = nimbussy.addFins()

    # IDEALLY IN THE FUTURE, [DYANMICS MODEL HERE]
    # while...


    # PLOTS HERE

    # Test Angles of Attack 
    aoa = np.deg2rad(np.linspace(0,5,20)) 
    #print(nimbussy.rocketCPPos(aoa)) # issue here
    #print(boatTail.cpPos)