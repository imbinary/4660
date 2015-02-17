__author__ = 'William Orem'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import math
import argparse


# tests each give a score from 0-1
# histogram test
def hist(iin, iout, name, show=True):
    img1 = cv2.cvtColor(iin, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(iout, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # use average of two methods
    ret = float(cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL) + (1-cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)))/float(2)
    if show:
        print "hist: " + str(ret)

    # print 1 - cv2.compareHist(hist1, hist2, 3)
    return ret


# template matching
def tmatch(intemp, infile, show=True):

    # 4 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF']
    template = cv2.cvtColor(intemp, cv2.COLOR_BGR2GRAY)
    template = template[100:300, 100:300]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    # cv2.imshow("cropped", template)
    # cv2.waitKey(0)
    for meth in methods:
        method = eval(meth)

        t, wi, hi = infile.shape[::-1]
        # make grayscale image for matching
        img2 = cv2.cvtColor(infile, cv2.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # find corners of matched image
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        ret = ((1-math.sqrt((top_left[0]-100)**2 + (top_left[1]-100)**2)/660.0) + (1-math.sqrt((bottom_right[0]-300)**2 + (bottom_right[1]-300)**2)/453.0))/2.0
        if show:
            print "temp: {}".format(ret)
    return ret


# sift
def sift(iin, iout, show=True):
    img1 = cv2.cvtColor(iin, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(iout, cv2.COLOR_BGR2GRAY)

    sifted = cv2.SIFT()
    kp1, des1 = sifted.detectAndCompute(img1, None)
    kp2, des2 = sifted.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    if show:
        print "sift: " + str(float(len(good))/len(kp1))

    return float(len(good))/float(len(kp1))


# custom test
def cust(iin, iout, show=True):
    img1 = cv2.cvtColor(iin, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(iout, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    # img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    # img1 = cv2.equalizeHist(img1)
    # img2 = cv2.equalizeHist(img2)
    cv2.imshow('clahe', img2)
    cv2.waitKey(10)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    ret = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

    if show:
        print "cust: " + str(ret)
    # print 1 - cv2.compareHist(hist1, hist2, 3)
    return ret


def top4(flist, path, tst, orig, show=True):
    # sort for best match
    od = collections.OrderedDict(sorted(flist.items(), key=lambda t: t[1], reverse=True))
    (setn, picn) = divmod(int(str(orig).split('.')[0].split("ukbench")[1]), 4)
    # initialize the results figure
    fig = plt.figure("Results"+tst)

    score = 0
    tscore = 0
    # loop over the results
    for (i, (k, v)) in enumerate(od.items()):
        img2 = cv2.imread(path+'/'+k, 1)
        # show the result
        if show:
            ax = fig.add_subplot(2, 2, i + 1)
            ax.set_title("%s: %.2f" % (k, v))
            # need to convert to RGB for plt
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.axis("off")
        (sett, pict) = divmod(int(str(k).split('.')[0].split("ukbench")[1]), 4)
        if setn == sett and picn == pict:
            score += 1
        elif setn == sett:
            tscore += 1

        if i == 3:
            break
    if tscore > 1:
        score +=tscore
    if show:
        fig.suptitle(tst + " Score(" + str(score) + ")", fontsize = 20)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the query image")
    ap.add_argument("-p", "--path", required=True, help="Path to the directory of images")
    ap.add_argument("-a", "--automate", required=False, help="run tests for all pictures in path")
    args = vars(ap.parse_args())

    image = args["image"]
    path = args["path"]
    show = False
    # Open a file
    # path = "images"
    dirs = os.listdir(path)
    l = len(dirs)

    print str(l) + " images, " + image + " was selected"
    img = cv2.imread(path+'/'+image, 1)

    # collection for data
    flist = {}
    tlist = {}
    hlist = {}
    clist = {}
    slist = {}

    # init to 0 then do 4 tests
    for fn in dirs:
        flist[fn] = 0
        tlist[fn] = 0
        hlist[fn] = 0
        clist[fn] = 0
        slist[fn] = 0

        img2 = cv2.imread(path+'/'+fn, 1)
        hlist[fn] = hist(img, img2, fn, show)
        flist[fn] += hlist[fn]
        tlist[fn] = tmatch(img, img2, show)
        flist[fn] += tlist[fn]
        slist[fn] = sift(img, img2, show)
        flist[fn] += slist[fn]
        clist[fn] = cust(img, img2, show)
        flist[fn] += clist[fn]
        if show:
            print fn + " " + str(flist[fn]) + "\n"

    top4(hlist, path, "histogram", image, show)
    top4(tlist, path, "template matching", image, show)
    top4(slist, path, "SIFT", image, show)
    top4(clist, path, "custom", image, show)
    # top4(flist, path, " total", image)
    # show the query image
    if show:
        fig = plt.figure("Query")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("%s" % image)
        # need to convert to RGB for plt
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        # show them
        plt.show()


if __name__ == "__main__":
    main()