# import Atmosphere2 as a
import Rocket as r
import numpy as np
import sixdof as sim

# FUNCTIONS #
def initialiseRocket():
    rocket = r.Rocket(4.645, 0.095, 3, 0.002, 1.2, 1.2, 9, 5*np.pi/180, 
                    np.pi/3, 2, 1, 5, 55.5, 55.3, 
                    np.array([[59.2, 0, 0], [0, 59.2, 0], [0, 0, 0.227]]), 
                    np.array([[6.1, 0, 0], [0, 6.1, 0], [0, 0, 0.027]]),
                    2.58, 200, 1.17, 3.24)
    
    nose = rocket.addNose(type="Haack",
                            length=0.6,
                            noseRadius=rocket.rocketRadius,
                            material="CFRP",
                            thickness=0.002) 
    
    bodyTube = rocket.addBodyTube(length=3.645,
                                    radius=rocket.rocketRadius,
                                    thickness=0.002,
                                    position=0.7,
                                    material="cfrp")
    
    boatTail = rocket.addBoatTail(upperRadius=rocket.rocketRadius, 
                                    lowerRadius=rocket.rocketRadius*0.8, 
                                    length=0.4, 
                                    thickness=0.002, 
                                    position=3.5, 
                                    material='gfrp')
    
    fins = rocket.addTrapezoidalFins(numberOfFins=rocket.finNumber,
                                     finSpan=0.22,
                                     finRootChord=0.322,
                                     finMidChord=0.261,
                                     finTipChord=0.2,
                                     sweepLength=0,
                                     sweepAngle=0,
                                     rocketRadius=rocket.rocketRadius,
                                     pos=3.5,
                                     mass=0.5,
                                     thickness=rocket.finThickness)
    rocket.evaluateFinProperties()
    rocket.evaluateRocketVolume()

    return rocket, nose, bodyTube, boatTail, fins


# MAIN SCRIPT #
if __name__ == "__main__":
    # Initialise rocket
    Nimbus, nose, bodyTube, boatTail, fins = initialiseRocket() # See function above
    # Run simulation
    sim.simulate(Nimbus)
    
