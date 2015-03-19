import sys

import motion
import almath
import vision_definitions
import cv2 as cv
from naoqi import ALProxy


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

    motion_proxy = ALProxy("ALMotion", pip, pport)
    tts = ALProxy("ALTextToSpeech", pip, pport)
    camProxy = ALProxy("ALVideoDevice", pip, pport)
    # -------------------------------------------
    # YOUR CODE HERE
    resolution = vision_definitions.kQQVGA
    colorSpace = vision_definitions.kYUVColorSpace
    fps = 30

    nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
    print nameId
    naoImage = camProxy.getImageRemote(nameId)
    camProxy.releaseImage(nameId)
    # cv.imshow("nb", naoImage[6])
    print naoImage[6]
    motion_proxy.wakeUp()
    motion_proxy.moveInit()
    motion_proxy.moveTo(0.4, 0.332, 0)
    tts.say("Hello World!")
    # print motion_proxy.getSummary()
    # YOUR CODE END

    kick(motion_proxy)

if __name__ == "__main__":
    main()