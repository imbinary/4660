__author__ = 'chris'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import collections
import random


# tests each give a score from 0-1
# histogram test
def hist(iin, iout):
    hist1 = cv2.calcHist([iin], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([iout], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return 1 - cv2.compareHist(hist1, hist2, 3)


# template matching
def tmatch(intemp, infile):

    # 4 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']

    for meth in methods:
        method = eval(meth)

        img = cv2.imread(infiles[x], 1)
        t, wi, hi = img.shape[::-1]
        # make grayscale image for matching
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(intemps[x], 0)
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # find corners of matched image
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # display plot
        plt.plot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)


# sift
def sift(iin, iout):
    gray = cv2.cvtColor(iin, cv2.COLOR_BGR2GRAY)

    sifted = cv2.SIFT()
    kp = sifted.detect(gray,None)

    img = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kp, des = sift.compute(gray, kp)


# custom test
def cust(iin, iout):
    pass


def main():
    # Open a file
    path = "images"
    dirs = os.listdir(path)
    l = len(dirs)
    # select a random image to use for matching
    t = random.randint(0, l-1)
    print str(l) + " images, " + dirs[t] + " was selected"
    img = cv2.imread(path+'/'+dirs[t], 1)
    plt.plot(121), plt.imshow(img)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.show()

    # collection for data
    flist = {}

    # init to 0 then do 4 tests
    for fn in dirs:
        flist[fn] = 0
        img2 = cv2.imread(path+'/'+fn, 1)
        flist[fn] += hist(img, img2)


    # sort for best match
    od = collections.OrderedDict(sorted(flist.items(), key=lambda t: t[1], reverse=True))

    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    plt.axis("off")

    # initialize the results figure
    fig = plt.figure("Results")

    # loop over the results
    for (i, (k, v)) in enumerate(od.items()):
        # show the result
        ax = fig.add_subplot(1, 4, i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        img2 = cv2.imread(path+'/'+k, 1)
        plt.imshow(img2)
        plt.axis("off")
        if i == 3:
            break

    # show the OpenCV methods
    plt.show()


if __name__ == "__main__":
    main()