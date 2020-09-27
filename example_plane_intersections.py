import numpy as np


these were member functions of a streamline class, so the `self` refers to the object that contained
all the info.

The points on the line were given by 'tracers', which I believe was also a list of 3-vectors.





def getPositiveCrossings(self, point, normal):
    """
    returns the indexes in the streamline array where the crossings occur
    The crossing occurs between this index and next index.
    """
    sides = np.dot( (self.tracers-point), normal)<0 #calculate the side of the plane each point falls on by projecting on the normal vector
    return np.flatnonzero((sides[:-1]^sides[1:]) & sides[:-1]) #calculate the xor of the boolean array that is 1 if it is above the plane with itself bitshifted. nonzero elements is where a crossing through the plane has taken place. Last and picks only crossings from positive to negative. flatnonzero returns indices of elements that are nonzero.


def pointOnPlane(p1, p2, point, normal):
    """
    calculate the point on the plane that is between the two points.
    """
    if np.sign(dToPlane(p1, point, normal))== np.sign(dToPlane(p2, point, normal)):
        print ('WARNING: POINTS NOT ON DIFFERENT SIDE OF PLANE')
        return
    linevec = p1-p2 #vector along the line
    distance =(np.dot( (point - p1),normal))/(np.dot(linevec, normal)) #see wikipedia, Line-plane_intersection
    return distance*linevec + p1


#here is an example on how the above functions are used to get the points where a fieldline intersects a plane:
def makePoincare(self, point=np.array([0,0,0]), normal =np.array([0,1,0]), verbose = 0):
    """
    returns the points on the plane (defined by point and normal)
    where the streamline crosses
    """
    #calculate two orthonormal vectors in the plane where we are looking for
    #the intersection
    x2=perpZ(normal) #calculate a vector perpendicular to the normal -> normal from frame
    x1=np.cross(normal, x2) # and another vector perpendicular to both -> binormal from frame
    x1/=np.sqrt(np.sum(x1**2))
    if verbose: print( x1, x2)
    crossingIndices=self.getPositiveCrossings(point, normal)
    print(len(crossingIndices))
    crossingsX1X2 = np.empty((len(crossingIndices),2))
    #print(crossingIndices)
    j=0
    for i in crossingIndices:
        crossCoord= pointOnPlane(self.tracers[i], self.tracers[i+1], point, normal)
        crossingsX1X2[j,:]=([np.dot(x1,crossCoord), np.dot(x2,crossCoord)]) #this goes wrong!
        j+=1
    return np.array(crossingsX1X2)


#^ the above for loop could be much improved by pythonic array casting! Betcha it fits on one line :D.
