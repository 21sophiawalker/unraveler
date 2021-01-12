#!/usr/bin/env python
# coding: utf-8

import os, sys, inspect       # For importing the submodules in a platform-independend robust way
# Make sure that the path to BlenDaViz and the integration library are in the front of the path.
code_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],".")))
if code_folder not in sys.path:
     sys.path.insert(0, code_folder)

import numpy as np
import matplotlib.pyplot as plt
from functools import partial  # create new functions from old functions (and strip the kwargs)
from scipy.integrate import odeint
import cmath
import sympy as sym
import frenet_serret
from mpl_toolkits.mplot3d import Axes3D
import BlenDaViz as blt


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return np.array((x,y,z))

def Kedia_32(xx):
    '''
    Vector field whose integral curves lie on knotted trefoil surfaces
    See Kedia et al. 10.1103/PhysRevLett.111.150404
    '''
    vv = np.zeros(3)
    vv[0] = (192*(2*xx[0]**5*xx[1] + xx[0]**3*xx[1]*(-1 + xx[2]**2) - 2*xx[1]**2*xx[2]*(-1 + xx[1]**2 + xx[2]**2) + 2*xx[0]**2*xx[2]*(-1 + 3*xx[1]**2 + xx[2]**2) - xx[0]*xx[1]*(1 + 2*xx[1]**4 - 6*xx[2]**2 + xx[2]**4 + 3*xx[1]**2*(-1 + xx[2]**2))))/(1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6

    vv[1] = -((96*(xx[0]**6 + 12*xx[0]**3*xx[1]*xx[2] - 4*xx[0]*xx[1]*xx[2]*(2 + xx[1]**2 - 2*xx[2]**2) + xx[0]**4*(-2 - 5*xx[1]**2 + 2*xx[2]**2) + xx[1]**2*(-1 + xx[1]**4 + 6*xx[2]**2 - xx[2]**4) + xx[0]**2*(1 - 5*xx[1]**4 - 6*xx[2]**2 + xx[2]**4 - 6*xx[1]**2*(-1 + xx[2]**2))))/(1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6)

    vv[2] = -((192*(xx[0]**5 - 3*xx[0]**4*xx[1]*xx[2] + xx[0]**2*xx[1]*xx[2]*(9 - 2*xx[1]**2 - 3*xx[2]**2) + xx[1]**3*xx[2]*(-3 + xx[1]**2 + xx[2]**2) + xx[0]**3*(-1 - 2*xx[1]**2 + 3*xx[2]**2) - 3*xx[0]*xx[1]**2*(-1 + xx[1]**2 + 3*xx[2]**2)))/(1 + xx[0]**2 + xx[1]**2 + xx[2]**2)**6)
    return vv/np.sum(np.sqrt(vv**2))

def fieldline(start=np.array([np.random.random_sample(),0,0]), field=Kedia_32, N=100, tf=100):
    """
    Integrate a fieldline using scipy.odeint
    """
    numargs = len(inspect.signature(field).parameters)
    if numargs == 1:
        # replace with function with proper call signature
        #print('adding time dependence to the function')
        placeholder = field
        field = lambda xx, t: placeholder(xx)
    elif numargs > 2:
        print("Error: function call signature takes too many arguments")
        raise TypeError
    #print(start)
    #print(field(start, 0))
    t = np.linspace(0, tf, N)
    X = odeint(field, start, t, hmax=0.05, rtol=1e-10)
    return X

magneticAxis = fieldline(start=np.array([1.07222,0.71903,0.816497]), field=Kedia_32, N=100, tf=100)


def fieldline2ListOfPoints(thisFieldline):
    """
    return the fieldline, which is a nx3 array
    as a list (length n) of points
    """
    return [thisFieldline[n,:] for n in range(thisFieldline.shape[0])]

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


def dToPlane(dispoint, point, normal):
    """
    calculates the distance from a point to a plane
    """
    return np.dot((dispoint-point),normal)

def getAllCrossings(points, point, normal):
    """
    returns the indexes in the points array where the crossings occur
    The crossing occurs between this index and next index.
    """
    sides = np.dot( (points-point), normal)<0 #calculate the side of the plane each point falls on by projecting on the normal vector
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


def getAnglefromSegmentCrossingFrame(segment, frame): #function that takes a line segment and corresponding frame
    """
    takes a line segment from a larger field line array and a corresponding frame and calculates the crossing index and angle within the frame
    """

    crossingIndices = getAllCrossings(segment, frame.position, frame.tangent) #finds the crossing of the frame and the segment

    if len(crossingIndices)==0:
        print('Warning! No crossing between segment and frame found! Try changing segment length.')

    if len(crossingIndices)>1:
        print('Warning! More than one crossing found! Try changing segment length.')

    crossCoord = pointOnPlane(segment[crossingIndices[0]], segment[crossingIndices[0]+1], frame.position, frame.tangent) #computes the crossing coordinate on the segment

    normalProjection = np.dot(crossCoord - frame.position, frame.normal) #finds the projection of the crossCoord onto the normal vector of the frame
    binormalProjection = np.dot(crossCoord - frame.position, frame.binormal) #finds the projection of the crossCoord onto the binormal vector of the frame

    crossingIndex = crossingIndices[0]

    angle = np.arctan2(normalProjection, binormalProjection)

    return angle, crossingIndex


tangents = calculateTangents(fieldline2ListOfPoints(magneticAxis), Kedia_32)
frames = frenet_serret.computeFrenetSerretFrames(magneticAxis[:31], tangents[:31])
stepSize = 0.1
points = fieldline(start=np.array(frames[0].position + frames[0].normal * stepSize), field=Kedia_32, N=1000, tf=100)


def rotationalTransform(points, frames):
    """
    calculates the rotational transform of a field line described by points around the core field line specified by frames
    """
    angles = []
    currentPosition = 1 #first point after 0th frame
    currentFrameNumber = 1
    searchForward = 35
    searchBack = 6
    #searchForward and searchBack define the segment as a crossing index plus searchForward and minus searchBack
    numFrames = len(frames)

    while currentPosition < len(points) - (searchForward+1):

        segment = points[max(currentPosition-searchBack, 0) : currentPosition+searchForward] #creates an array of 16 points on the Field Line which will be used to find the crossings of the Field Line and the Frenet-Serret Frames

        currentFrame = frames[currentFrameNumber]

        angle, crossingIndex = getAnglefromSegmentCrossingFrame(segment, currentFrame) #grabes the crossingIndex of the segment and frame and finds the angle of the crossing in frame

        currentFrameNumber = (currentFrameNumber + 1) % (numFrames) #updates the frame to the next frame alone the magnetic axis
        currentPosition = max(currentPosition-searchBack, 0) + crossingIndex +1 #updates the position to the position after the crossing index to define next search segment (NO NEED FOR SEARCHBACK?)

        angles.append(angle) #adds angle from getAnglefromSegmentCrossingFrame to empty list defined above
        

    angles = np.array(angles) #makes angles into an array rather than a list
    angleDiff = (angles[1:]-angles[:-1])%(2*np.pi) #the difference between an angle and the angle before modulo 2pi as to not overcount

    angleDiff = angleDiff - 2*np.pi #ad hoc?

    numCompleteTurns = int((len(angles)/(numFrames))) #int throws away everything after the decimal

    TotalAngle = sum(angleDiff[:numCompleteTurns*numFrames - 1]) #subtract 1 because angleDiff has one less element than angles and python is 0-indexed. Doublecheck!

    RotationalTransform = (TotalAngle / (numCompleteTurns * (2*np.pi)))
    print("The field line has made {} complete turns".format(numCompleteTurns))

    return RotationalTransform

##Comparing the rotational transform and stepSize:
rotationalTransforms = []
tangents = calculateTangents(fieldline2ListOfPoints(magneticAxis), Kedia_32)
frames = frenet_serret.computeFrenetSerretFrames(magneticAxis[:31], tangents[:31])

"""
for stepSize in np.linspace(0.01, 0.2, 20):
    points = fieldline(start=np.array(frames[0].position + frames[0].normal * stepSize), field=Kedia_32, N=10000, tf=1000)

    rotationalTransforms.append(rotationalTransform(points, frames))

plt.plot(rotationalTransforms)

"""

"""
figure out how to make a figure 
"""



def drawFrame(frame, ax):
    """
    plots a frame object in the 3d axes given by ax
    """

    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.normal[0], frame.normal[1], frame.normal[2], color = "r") 
    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.tangent[0], frame.tangent[1], frame.tangent[2], color = "g")
    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.binormal[0], frame.binormal[1], frame.binormal[2], color = "b")

def drawFrameGrey(frame, ax):
    """
    plots a frame object in the 3d axes given by ax
    """


    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.normal[0], frame.normal[1], frame.normal[2], color = "gray") 
    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.tangent[0], frame.tangent[1], frame.tangent[2], color = "gray")
    ax.quiver(frame.position[0], frame.position[1], frame.position[2], frame.binormal[0], frame.binormal[1], frame.binormal[2], color = "gray")

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

for frame in frames:
    drawFrameGrey(frame, ax)
"""
plt.show()
"""
myFieldLine = fieldline(start=np.array([2.4348,0.71903,0.816497]), field=Kedia_32, N=1000, tf=800)
print(myFieldLine[:,0])
print(myFieldLine.shape)

def drawLine(segment, ax, color = "grey"):
    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color = color)



def rotationalTransformImages(points, frames, ax):
    """
    calculates the rotational transform of a field line described by points around the core field line specified by frames
    """
    angles = []
    currentPosition = 1 #first point after 0th frame
    currentFrameNumber = 1
    searchForward = 35
    searchBack = 6
    #searchForward and searchBack define the segment as a crossing index plus searchForward and minus searchBack
    numFrames = len(frames)
    count = 0
    while currentPosition < len(points) - (searchForward+1):

        segment = points[max(currentPosition-searchBack, 0) : currentPosition+searchForward] #creates an array of 16 points on the Field Line which will be used to find the crossings of the Field Line and the Frenet-Serret Frames 

        currentFrame = frames[currentFrameNumber]

        angle, crossingIndex = getAnglefromSegmentCrossingFrame(segment, currentFrame) #grabes the crossingIndex of the segment and frame and finds the angle of the crossing in frame

        currentFrameNumber = (currentFrameNumber + 1) % (numFrames) #updates the frame to the next frame alone the magnetic axis
        currentPosition = max(currentPosition-searchBack, 0) + crossingIndex +1 #updates the position to the position after the crossing index to define next search segment (NO NEED FOR SEARCHBACK?)

        angles.append(angle) #adds angle from getAnglefromSegmentCrossingFrame to empty list defined above

        drawLine(points, ax, color = "grey")

        drawLine(segment, ax, color = "b")

        for frame in frames:
            if (frame != currentFrame):
                drawFrameGrey(frame, ax)

        drawFrame(currentFrame, ax)

        plt.savefig("/Users/sophiawalker/Documents/unraveler_images/unraveler{}.jpg".format(count)) 

        count += 1


        

    angles = np.array(angles) #makes angles into an array rather than a list
    angleDiff = (angles[1:]-angles[:-1])%(2*np.pi) #the difference between an angle and the angle before modulo 2pi as to not overcount

    angleDiff = angleDiff - 2*np.pi #ad hoc?

    numCompleteTurns = int((len(angles)/(numFrames))) #int throws away everything after the decimal

    TotalAngle = sum(angleDiff[:numCompleteTurns*numFrames - 1]) #subtract 1 because angleDiff has one less element than angles and python is 0-indexed. Doublecheck!

    RotationalTransform = (TotalAngle / (numCompleteTurns * (2*np.pi)))
    print("The field line has made {} complete turns".format(numCompleteTurns))

    return RotationalTransform

rotationalTransformImages(points, frames, ax)




