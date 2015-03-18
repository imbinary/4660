import sys

import motion
import almath
from naoqi import ALProxy


def shift_weight(motion_proxy):
    effectors = ["Head", "Torso"]
    axis_masks = [almath.AXIS_MASK_ROT, almath.AXIS_MASK_VEL]
    times = [[2.0], [2.0]]  # seconds
    paths = [[[0.0, 0.0, 0.0, 0.0, 0.45553, 0.0]],
             [[0.0, -0.1, -0.02, 0.0, 0.04, 0.0]]]
    motion_proxy.positionInterpolations(effectors, motion.FRAME_ROBOT, paths, axis_masks, times, False)

    path = [0.0, 0.00, 0.04, 0.0, 0.0, 0.0]
    motion_proxy.positionInterpolation("LLeg", motion.FRAME_ROBOT, path, almath.AXIS_MASK_ALL, 2.0, False)


def kick(motion_proxy):
    try:
        shift_weight(motion_proxy)

        path = [-0.10, 0.0, 0.00, 0.0, -0.03, 0.0]
        motion_proxy.positionInterpolation("LLeg", motion.FRAME_ROBOT, path, almath.AXIS_MASK_ALL, 2.0, False)

        path = [[0.15, 0.0, 0.00, 0.0, -0.03, 0.0],
                [0.24, 0.0, 0.00, 0.0, -0.03, 0.0]]
        tm = [0.2, 0.3]  # seconds
        motion_proxy.positionInterpolation("LLeg", motion.FRAME_ROBOT, path, almath.AXIS_MASK_ALL, tm, False)

    except Exception as e:
        sys.exit(e)


def main():
    pip = "127.0.0.1"
    pport = 9559

    motion_proxy = ALProxy("ALMotion", pip, pport)

    # -------------------------------------------
    # YOUR CODE HERE


    # YOUR CODE END

    kick(motion_proxy)

if __name__ == "__main__":
    main()