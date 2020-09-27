import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import collections


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, inspect  # For importing the submodules in a platform-independend robust way
# Make sure that the path to BlenDaViz and the integration library are in the front of the path.
code_folder = os.path.realpath(
    os.path.abspath(
        os.path.join(
            os.path.split(inspect.getfile(inspect.currentframe()))[0], ".")))
if code_folder not in sys.path:
    sys.path.insert(0, code_folder)
import frenet_serret



def Kedia_32(xx):
    '''
    Vector field whose integral curves lie on knotted trefoil surfaces
    See Kedia et al. 10.1103/PhysRevLett.111.150404
    '''
    vv = np.zeros(3)
    vv[0] = (192 *
             (2 * xx[0]**5 * xx[1] + xx[0]**3 * xx[1] *
              (-1 + xx[2]**2) - 2 * xx[1]**2 * xx[2] *
              (-1 + xx[1]**2 + xx[2]**2) + 2 * xx[0]**2 * xx[2] *
              (-1 + 3 * xx[1]**2 + xx[2]**2) - xx[0] * xx[1] *
              (1 + 2 * xx[1]**4 - 6 * xx[2]**2 + xx[2]**4 + 3 * xx[1]**2 *
               (-1 + xx[2]**2)))) / (1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6

    vv[1] = -((
        96 *
        (xx[0]**6 + 12 * xx[0]**3 * xx[1] * xx[2] - 4 * xx[0] * xx[1] * xx[2] *
         (2 + xx[1]**2 - 2 * xx[2]**2) + xx[0]**4 *
         (-2 - 5 * xx[1]**2 + 2 * xx[2]**2) + xx[1]**2 *
         (-1 + xx[1]**4 + 6 * xx[2]**2 - xx[2]**4) + xx[0]**2 *
         (1 - 5 * xx[1]**4 - 6 * xx[2]**2 + xx[2]**4 - 6 * xx[1]**2 *
          (-1 + xx[2]**2)))) / (1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6)

    vv[2] = -(
        (192 *
         (xx[0]**5 - 3 * xx[0]**4 * xx[1] * xx[2] + xx[0]**2 * xx[1] * xx[2] *
          (9 - 2 * xx[1]**2 - 3 * xx[2]**2) + xx[1]**3 * xx[2] *
          (-3 + xx[1]**2 + xx[2]**2) + xx[0]**3 *
          (-1 - 2 * xx[1]**2 + 3 * xx[2]**2) - 3 * xx[0] * xx[1]**2 *
          (-1 + xx[1]**2 + 3 * xx[2]**2))) /
        (1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6)
    return vv / np.sum(np.sqrt(vv**2))


def fieldline(start=np.array([np.random.random_sample(), 0, 0]),
              field=Kedia_32,
              N=100,
              tf=100):
    numargs = len(inspect.signature(field).parameters)
    if numargs == 1:
        # replace with function with proper call signature
        print('adding time dependence to the function')
        placeholder = field
        field = lambda xx, t: placeholder(xx)
    elif numargs > 2:
        print("Error: function call signature takes too many arguments")
        raise TypeError
    print(start)
    print(field(start, 0))
    t = np.linspace(0, tf, N)
    X = odeint(field, start, t, hmax=0.05, rtol=1e-10)
    return X

def calculateTangents(points, function):
    """
    return the tangents of a field line
    by evaluating the function at points
    arguments:
    *points*: 3-arrays where the field is to be evaluated
    *function*: the function that was integrated to generate the points
    """
    #Evaluate the function at point, and use fancy-schmancy
    #python array-casting to make a list of this for every point.
    tangents = [function(point) for point in points]
    return tangents

def fieldline2ListOfPoints(thisFieldline):
    """
    return the fieldline, which is a nx3 array
    as a list (length n) of points
    """
    return [thisFieldline[n,:] for n in range(thisFieldline.shape[0])]

def dToPlane(dispoint, point, normal):
    return np.dot((dispoint-point),normal)


def getPositiveCrossings(points, point, normal):
    """
    returns the indexes in the streamline array where the crossings occur
    The crossing occurs between this index and next index.
    """
    sides = np.dot( (points-point), normal)<0 #calculate the side of the plane each point falls on by projecting on the normal vector
    print('Sides outputs {}'.format(sides))
    return np.flatnonzero((sides[:-1]^sides[1:])) #calculate the xor of the boolean array that is 1 if it is above the plane with itself bitshifted. nonzero elements is where a crossing through the plane has taken place. Last and picks only crossings from positive to negative. flatnonzero returns indices of elements that are nonzero.


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



MyFieldLine = fieldline()
tangents = calculateTangents(fieldline2ListOfPoints(MyFieldLine), Kedia_32)
frames = frenet_serret.computeFrenetSerretFrames(MyFieldLine[:31], tangents[:31])


def frameAngle(points, frames):
    angle = []
    currentPosition = 1 #first point after 0th frame
    currentFrameNumber = 1
    searchForward = 35
    searchBack = 15

    while currentPosition < len(points) - (searchForward+1):
        print('Current Frame Number is {}'.format(currentFrameNumber))
        segmentStart = max(currentPosition-searchBack, 0)
        segmentEnd = currentPosition + searchForward

        segment = points[segmentStart:segmentEnd] #creates an array of 16 points on the Field Line which will be used to find the crossings of the Field Line and the Frenet-Serret Frames
        #The starting slice is chosen as to not take negative elements for the first frame segment
        #print('segment returns {}'.format(segment))

        currentFrame = frames[currentFrameNumber]
        print('searching for an intersection in the range: [{}:{}]'.format(segmentStart, segmentEnd))

        crossingIndices = getPositiveCrossings(segment, currentFrame.position, currentFrame.tangent) #returns the indexes in the segment array where the crossing occurs
        print('crossingIndices returns {}'.format(crossingIndices))

        if len(crossingIndices)>1:
            print('SearchLength is too long! Finding more than one intersection')

        crossCoord = pointOnPlane(segment[crossingIndices[0]], segment[crossingIndices[0]+1], currentFrame.position, currentFrame.tangent ) #calculates the point on the plane that is between the two points in the segment array
        print('crossCoord returns {}'.format(crossCoord))

        normalProjection = np.dot(crossCoord - currentFrame.position, currentFrame.normal)
        print('Output of normalDirection is {}'.format(normalProjection))

        binormalProjection = np.dot(crossCoord - currentFrame.position, currentFrame.binormal)
        print('Output of binormalDirection is {}'.format(normalProjection))

        angle.append(np.arctan2(normalProjection, binormalProjection)) #angle in radians

        currentFrameNumber = ((currentFrameNumber + 1) % (len(frames)))
        print('Current Frame Number is {}'.format(currentFrameNumber))
        currentPosition = currentPosition + crossingIndices[0] +1
        print('Current Position is {}'.format(currentPosition))

    print('The angle array is {}'.format(angle))
    return angle



def displaySegmenAndFrame(segment, frames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(segment[:,0], segment[:,1], segment[:,2])
    if not isinstance(frames, collections.Iterable):
        frames = [frames,]
    for frame in frames
        for vector in (frame.normal, frame.binormal, frame.tangent):
            ax.quiver(frame.position[0], frame.position[1], frame.position[2],
                    vector[0], vector[1], vector[2])



stepSize = 0.1
points = fieldline(start=np.array(frames[0].position + frames[0].normal * stepSize), field=Kedia_32, N=1000, tf=100)
angles = frameAngle(points, frames)
plt.plot(angles)
plt.show()

stepSize = 0.1
points = fieldline(start=np.array(frames[0].position + frames[0].normal * stepSize), field=Kedia_32, N=1000, tf=100)
angles = frameAngle(points, frames)
plt.plot(angles)
plt.show()
