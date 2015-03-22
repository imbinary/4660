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



def shift_weight(use_sensor_values):
    axis_mask = almath.AXIS_MASK_ALL   # full control

    # Lower the Torso and move to the side
    effector = "Torso"
    frame = motion.FRAME_ROBOT
    times = 2.0     # seconds
    try:
        init_tf = almath.Transform(motionProxy.getTransform(effector, frame, use_sensor_values))
    except Exception as e:
        sys.exit(e)
    delta_tf = almath.Transform(0.0, -0.06, -0.03)  # x, y, z
    target_tf = init_tf * delta_tf
    path = list(target_tf.toVector())
    motionProxy.transformInterpolations(effector, frame, path, axis_mask, times)

    # Lift LLeg
    effector = "LLeg"
    frame = motion.FRAME_TORSO
    times = 2.0  # seconds
    try:
        init_tf = almath.Transform(motionProxy.getTransform(effector, frame, use_sensor_values))
    except Exception as e:
        sys.exit(e)
    delta_tf = almath.Transform(0.0, 0.0, 0.04)
    target_tf = init_tf * delta_tf
    path = list(target_tf.toVector())
    motionProxy.transformInterpolations(effector, frame, path, axis_mask, times)


def kick():
    frame = motion.FRAME_TORSO
    axis_mask = almath.AXIS_MASK_ALL   # full control
    use_sensor_values = False

    shift_weight(use_sensor_values)

    # move LLeg back
    effector = "LLeg"
    times = 4.0     # seconds
    current_pos = motionProxy.getPosition(effector, frame, use_sensor_values)
    target_pos = almath.Position6D(current_pos)
    target_pos.x -= 0.1
    target_pos.wy -= 0.03
    path_list = [list(target_pos.toVector())]
    motionProxy.positionInterpolations(effector, frame, path_list, axis_mask, times)

    # swing LLeg forward
    times = [0.2, 0.3]  # seconds
    current_pos = motionProxy.getPosition(effector, frame, use_sensor_values)
    target_pos = almath.Position6D(current_pos)
    target_pos.x += 0.15
    target_pos.wy -= 0.03
    path_list = [list(target_pos.toVector())]

    target_pos = almath.Position6D(current_pos)
    target_pos.x += 0.24
    target_pos.wy -= 0.03
    path_list.append(list(target_pos.toVector()))
    motionProxy.positionInterpolations(effector, frame, path_list, axis_mask, times)

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

def centerOnBall(loc, camera):
    tol = 10
    if camera == 1:
        tol = 6

    if loc[0] == -1:
        print "no ball"
        return -1

    if abs(loc[0]-X) < tol:
        print "Heading is on"
        return 1

    return 0
def turnrobot(loc, motionProxy):
    turn = .4*(X-loc[0]/float(X))
    print "turning " + str(turn)
    motionProxy.moveTo(0, 0, turn)


def moveforward(loc, camera, motionProxy):
    dist = 0.6
    print loc[1]
    if camera == 1:
        dist = 0.45 * ((120.0-loc[1])/120.0)

    motionProxy.moveTo(dist, 0, 0)

def main():
    pip = "127.0.0.1"
    pport = 9559


    global motionProxy
    motionProxy = ALProxy("ALMotion", pip, pport)
    postureProxy = ALProxy("ALRobotPosture", pip, pport)
    # -------------------------------------------
    # YOUR CODE HERE
    camera = 0
    seeball = 0
    lowball = 0
    # setup additional proxies
    camProxy = ALProxy("ALVideoDevice", pip, pport)

    # initialize motion
    motionProxy.wakeUp()
    postureProxy.goToPosture("StandInit", 0.5)

    # logic
    while True:
        # showCam(camProxy)
        im1 = getImage(camProxy, camera)
        loc = findBall(im1)
        val = centerOnBall(loc, camera)
        if val == -1:
            # no ball
            if camera == 1 and lowball == 1:
                #kick we had ball on lower and now its gone
                break
            if camera == 1 and seeball == 0:
                # wander no ball in lower or upper
                print "wander"
                motionProxy.moveTo(0, 0, .6)
                camera = 0
            else:
                # look in lower
                camera = 1
        elif val == 1:
            # head on move forward
            moveforward(loc, camera, motionProxy)
            if camera == 1:
                lowball = 1
            seeball = 1
        else:
            # turn
            turnrobot(loc, motionProxy)
            seeball = 1

    motionProxy.moveTo(0, -.2, 0)
    motionProxy.moveTo(.14, 0, 0)
    postureProxy.goToPosture("StandInit", 0.5)
    print "kicking now"
    # YOUR CODE END

    kick()
    postureProxy.goToPosture("StandInit", 0.5)
if __name__ == "__main__":
    main()