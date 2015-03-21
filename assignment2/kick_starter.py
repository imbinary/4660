import sys
from PIL import Image
import numpy as np
import cv2
import time
from naoqi import ALProxy
import motion
import almath
import vision_definitions


X = 160/2
Y = 120/2

def shift_weight(motion_proxy):
    effectors = ["Head", "Torso"]
    axis_masks = [almath.AXIS_MASK_ROT, almath.AXIS_MASK_VEL]
    times = [[2.0], [2.0]]  # seconds
    paths = [[[0.0, 0.0, 0.0, 0.0, 0.45553, 0.0]],
             [[0.0, -0.1, -0.02, 0.0, 0.04, 0.0]]]
    motion_proxy.positionInterpolations(effectors, motion.FRAME_ROBOT, paths, axis_masks, times, False)

    path = [0.0, 0.00, 0.04, 0.0, 0.0, 0.0]
    motion_proxy.positionInterpolation("LLeg", motion.FRAME_TORSO, path, almath.AXIS_MASK_ALL, 2.0, False)


def kick(motion_proxy):
    try:
        shift_weight(motion_proxy)

        path = [-0.10, 0.0, 0.00, 0.0, -0.03, 0.0]
        motion_proxy.positionInterpolation("LLeg", motion.FRAME_TORSO, path, almath.AXIS_MASK_ALL, 2.0, False)

        path = [[0.15, 0.0, 0.00, 0.0, -0.03, 0.0],
                [0.24, 0.0, 0.00, 0.0, -0.03, 0.0]]
        tm = [0.2, 0.3]  # seconds
        motion_proxy.positionInterpolation("LLeg", motion.FRAME_TORSO, path, almath.AXIS_MASK_ALL, tm, False)

    except Exception as e:
        sys.exit(e)

def getImage(camProxy, camera):
    # get an image
    resolution = vision_definitions.kQQVGA
    colorSpace = vision_definitions.kRGBColorSpace
    fps = 30
    nameId = camProxy.subscribeCamera("python_GVM", camera, resolution, colorSpace, fps)
    naoImage = camProxy.getImageRemote(nameId)
    camProxy.releaseImage(nameId)
    camProxy.unsubscribe(nameId)
    im = np.array(Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6]))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    return im

def findBall(im):
    ORANGE_MIN = np.array([5, 50, 50], np.uint8)
    ORANGE_MAX = np.array([15, 255, 255], np.uint8)
    img2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    img2 = cv2.inRange(img2, ORANGE_MIN, ORANGE_MAX)
    mom = cv2.moments(img2)
    x = -1
    y = -1
    if mom["m00"] != 0:
        y = int(mom["m01"]/mom["m00"])
        x = int(mom["m10"]/mom["m00"])

    return x, y

def showCam(camProxy):
    im1 = getImage(camProxy, 0)
    im2 = getImage(camProxy, 1)

    loc = findBall(im1)
    if loc[0] == -1:
        print "no ball"
    else:
        cv2.circle(im1, loc, 5, (0, 0, 200))

    im = np.concatenate((im1, im2), axis=0)
    cv2.imshow("bottom", im)
    cv2.waitKey(2)

def centerOnBall(motionProxy, camProxy, camera):
    im1 = getImage(camProxy, camera)
    loc = findBall(im1)
    print loc[0]-X

    if abs(loc[0]-X)>50:
        turn = .3
    elif abs(loc[0]-X)>30:
        turn = .2
    else:
        turn = .1

    if loc[0] == -1:
        print "no ball"
        return -1

    if abs(loc[0]-X) < 3:
        print "Heading is on"
        return 1

    if loc[0]-X < 0:
        # turn left
        print "turning left"
        motionProxy.moveTo(0, 0, turn)
    else:
        # turn right
        print "turning right"
        motionProxy.moveTo(0, 0, -turn)
    time.sleep(2)
    return 0

def main():
    pip = "127.0.0.1"
    pport = 9559

    motionProxy = ALProxy("ALMotion", pip, pport)
    camProxy = ALProxy("ALVideoDevice", pip, pport)
    postureProxy = ALProxy("ALRobotPosture", pip, pport)
    # -------------------------------------------
    # YOUR CODE HERE

    # setup additional proxies

    # initialize motion
    motionProxy.wakeUp()
    postureProxy.goToPosture("StandInit", 0.5)
    #motionProxy.moveInit()
    for y in range(5):
        for x in range(10):
            showCam(camProxy)
            centerOnBall(motionProxy, camProxy, 0)
        motionProxy.moveTo(.3, 0, 0)
    #motionProxy.moveToward(0, 0, -.2, [["Frequency", 0.5]])
    #motionProxy.rest()

    # YOUR CODE END
    print "kicking now"
    kick(motionProxy)

if __name__ == "__main__":
    main()