__author__ = 'William Orem'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import math
import argparse

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    from http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# tests each give a score from 0-1
# histogram test
def hist(iin, iout, name):
    img1 = cv2.cvtColor(iin, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(iout, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # use average of two methods
    ret = float(cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL) + (1-cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)))/float(2)
    print "hist: " + str(ret)

    # print 1 - cv2.compareHist(hist1, hist2, 3)
    return ret


# template matching
def tmatch(intemp, infile):

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
        print "temp: {}".format(ret)
    return ret


# sift
def sift(iin, iout):
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
    print "sift: " + str(float(len(good))/len(kp1))

    return float(len(good))/float(len(kp1))


# custom test
def cust(iin, iout):

    # img1 = cv2.GaussianBlur(iin, (5, 5))
    img1 = iin
    img2 = cv2.GaussianBlur(iout, (5, 5), 1)
    img1 = img1[160:480, 120:360]
    img2 = img2[160:480, 120:360]

    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    ret = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

    print "cust: " + str(ret)
    # print 1 - cv2.compareHist(hist1, hist2, 3)
    return ret


def top4(flist, path, tst):
    # sort for best match
    od = collections.OrderedDict(sorted(flist.items(), key=lambda t: t[1], reverse=True))

    # initialize the results figure
    fig = plt.figure("Results"+tst)
    fig.suptitle(tst, fontsize = 20)
    # loop over the results
    for (i, (k, v)) in enumerate(od.items()):
        # show the result
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        img2 = cv2.imread(path+'/'+k, 1)
        # need to convert to RGB for plt
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if i == 3:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the query image")
    ap.add_argument("-p", "--path", required=True, help="Path to the directory of images")
    args = vars(ap.parse_args())

    image = args["image"]
    path = args["path"]
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
        hlist[fn] = hist(img, img2, fn)
        flist[fn] += hlist[fn]
        tlist[fn] = tmatch(img, img2)
        flist[fn] += tlist[fn]
        slist[fn] = sift(img, img2)
        flist[fn] += slist[fn]
        clist[fn] = cust(img, img2)
        flist[fn] += clist[fn]
        print fn + " " + str(flist[fn]) + "\n"

    top4(hlist, path, "histogram")
    top4(tlist, path, "template matching")
    top4(slist, path, "SIFT")
    top4(clist, path, "custom")
    # top4(flist, path, " total")
    # show the query image
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