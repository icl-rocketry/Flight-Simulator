import Atmosphere2 as a
import Rocket2 as r
import numpy as np

if __name__ == "__main__":
    nimbussy = r.Rocket(5,20)
    nose = nimbussy.addNose("VonKarman",10, nimbussy.rocketRadius, nimbussy.rocketRadius)
    #boatTail = nimbussy.addBoatTail(10,2,10,nimbussy.rocketRadius,200)
    print(nose.cn)
    print(nose.cnPos)
    print(nimbussy.rocketCPPos)
    #print(boatTail.cpPos)