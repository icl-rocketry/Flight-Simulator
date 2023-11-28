import Atmosphere2 as a
import Rocket2 as r
import numpy as np

if __name__ == "__main__":
    # Initialising rocket
    nimbussy = r.Rocket(5,0.2)
    nose = nimbussy.addNose(type="Haack",
                            length=0.6,
                            noseRadius=nimbussy.rocketRadius,
                            material="CFRP",
                            thickness=0.002
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


    #boatTail = nimbussy.addBoatTail(10,2,10,nimbussy.rocketRadius,200)
    aoa = np.deg2rad(np.linspace(0,5,20)) # Test Angles of Attack 
    #print(nose.cn(aoa))
    #print(nose.cnPos)
    #print(nimbussy.rocketCN(aoa))
    print(nimbussy.rocketCPPos(aoa))
    print(nose.volume)
    print(nose.mass)
    print(bodyTube.mass)
    print(nimbussy.surfaceMass)
    #print(boatTail.cpPos)