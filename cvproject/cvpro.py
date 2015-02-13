__author__ = 'chris'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import collections


# tests each give a score from 0-1
# histogram test
def htest():
    pass


# template matching
def tmatch():
    pass


# sift
def sift():
    pass


# custom test
def cust():
    pass


def main():
    # Open a file
    path = "images"
    dirs = os.listdir(path)
    l = len(dirs)
    flist = {}
    for fn in dirs:
        flist[fn] = 0

    od = collections.OrderedDict(sorted(flist.items(), key=lambda t: t[0]))
    for x in od.items():
        print x


if __name__ == "__main__":
    main()