import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion




def computePathTangents(points):
    tangents = []
    isNotLastPoint = index < length(points) - 1

    if points[0] == points[length(points)]:  # isClosed?
        nextPoint = points[index + 1] if isNotLastPoint else points[1]
        tangent = nextPoint * point
    elif isNotLastPoint:
        nextPoint = points[index + 1]
        tangent = nextPoint * point
    else:
        prevPoint = points[index - 1]
        tangent = point * prevPoint

    tangent = normalize(tangent)

    return tangents


class framfram(object):
    """
    Class that contains frame information
    """
    def __init__(self, position=np.zeros(3), tangent=np.array((1,0,0)), normal=np.array((0,1,0)), binormal=np.array((0,0,1))):
        """
        Initialize a frame by giving it a tangent, normal and binormal.
        """
        self.position = position
        self.tangent = tangent
        self.normal = normal
        self.binormal = binormal



def computeFrenetSerretFrames(points, tangents, closed=True, initialNormal=None):
    """
    Computes a set of frames for a given curve using quaternion algebra.
    Arguments:
    *points*:
        a list of points on the curve
    *tangents*:
        a list of tangent vectors at each point on the curve
    *closed*:
        If true, the curve is assumed closed and each frame is rotated such that the frames 'match up'
    *initialNormal*:
        Normal vector for the initial frame.
    Adapted from Github Damien xxx
    """


    # how to write the kwargs for closed=false and initialNormal = Null??
    # C: just add them to the call signature with their default value! (yay Python)
    # creating empty arrays/lists for the rest of the function to store values in
    X_UP = np.array([1, 0, 0])
    Y_UP = np.array([0, 1, 0])
    Z_UP = np.array([0, 0, 1])
    frames = []  # or maybe [0]*length(points)?

    # Compute inital frame

    tangent = tangents[0]
    tangent = tangent/np.linalg.norm(tangent)


    atx = abs(tangent[0]) #absolute value of the x-component of the tangent vector
    aty = abs(tangent[1]) #absolute value of the y-component of the tangent vector
    atz = abs(tangent[2]) #absolute value of the z-component of the tangent vector

    if initialNormal is None:
        if aty > atx and aty >= atz: #
            v = np.cross(tangent, X_UP)
        elif atz > atx and atz >= aty:
            v = np.cross(tangent, Y_UP)
        else:
            v = np.cross(tangent, Z_UP)
        normal = np.cross(tangent, v)
        normal = normal/np.linalg.norm(normal)  # is this right??
    else:
        normal = initialNormal/np.linalg.norm(initialNormal)



    binormal = np.cross(tangent, normal)
    binormal = binormal/np.linalg.norm(binormal)
    print(tangent, normal, binormal)
    firstframe = framfram(position=points[0], tangent=tangent, normal=normal, binormal=binormal)
    frames.append(firstframe)

    #rotation by quarternion calculations

    previousTangent = tangent
    for point, tangent in zip(points[1:], tangents[1:]): #pythonic way of looping over elements
        tangent = tangent/np.linalg.norm(tangent)


        v = np.cross(previousTangent, tangent)

        if np.linalg.norm(v) > np.finfo(float).eps:  # too small to matter
            v = v/np.linalg.norm(v)

        theta = np.arccos(np.dot(previousTangent, tangent))

        RotationQuaternion = Quaternion(axis=v, radians=theta)  # This goes wrong! Need to figure out how to do this correctly
        normal = RotationQuaternion.rotate(normal)

        binormal = np.cross(tangent, normal)

        frames.append(framfram(position=point, normal=normal, binormal=binormal, tangent=tangent))
        previousTangent = tangent

    if closed:
        firstFrame = frames[0]
        firstNormal = firstFrame.normal
        lastFrame = frames[-1]
        lastNormal = lastFrame.normal

        theta = np.arccos(np.dot(firstNormal, lastNormal))
        theta = theta - 2 * np.pi # This line was added specifically to make the trefoil frame untwisted
        theta = theta/(len(frames)-1)

        if np.dot(tangents[0], np.cross(firstNormal, lastNormal)) > 0:
            theta = -1*theta
        print(theta)
        for num, frame in enumerate(frames[1:]):  #zero -indexed? start num with 1?
            RotationQuaternion = Quaternion(axis=frame.tangent, angle=(num+1)*theta)
            frame.normal = RotationQuaternion.rotate(frame.normal)
            frame.binormal = np.cross(frame.tangent, frame.normal)

    return frames
