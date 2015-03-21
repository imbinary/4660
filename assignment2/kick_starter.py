import sys
import Image
import numpy as np

import cv2

from naoqi import ALProxy, motion, almath, vision_definitions


def shift_weight(motion_proxy):
    effectors = ["Head", "Torso"]
    axis_masks = [almath.AXIS_MASK_ROT, almath.AXIS_MASK_VEL]
    times = [[2.0], [2.0]]  # seconds
    paths = [[[0.0, 0.0, 0.0, 0.0, 0.45553, 0.0]],
             [[0.0, -0.1, -0.02, 0.0, 0.04, 0.0]]]
    motion_proxy.positionInterpolations(effectors, motion.FRAME_TORSO, paths, axis_masks, times, False)

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


def main():
    pip = "127.0.0.1"
    pport = 9559

    motionProxy = ALProxy("ALMotion", pip, pport)

    # -------------------------------------------
    # YOUR CODE HERE
    # setup additional proxies
    camProxy = ALProxy("ALVideoDevice", pip, pport)
    postureProxy = ALProxy("ALRobotPosture", pip, pport)

    # get an image
    resolution = vision_definitions.kQQVGA
    colorSpace = vision_definitions.kRGBColorSpace
    fps = 30
    nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
    naoImage = camProxy.getImageRemote(nameId)
    camProxy.releaseImage(nameId)
    # camProxy.unsubscribe()

    im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6])
    cv2.imshow("nb", np.array(im))
    cv2.waitKey(430)

    motionProxy.wakeUp()
    postureProxy.goToPosture("StandInit", 0.5)
    motionProxy.moveInit()
    # motionProxy.setStiffnesses("Body", 1.0)
    motionProxy.moveTo(0.4, 0.332, 0)

    # print motion_proxy.getSummary()
    # YOUR CODE END

    kick(motionProxy)
if __name__ == "__main__":
    main()