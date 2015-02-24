__author__ = 'William Orem, Carolus Andrews'
import cv2
import matplotlib.pyplot as plt
import os
import collections
import math
import argparse
import timeit
import fnmatch

g = {}
# tests each give a score from 0-1
# histogram test
def hist(iin, iout, show=True):
    # convert to HSV
    img1 = cv2.cvtColor(iin, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(iout, cv2.COLOR_BGR2HSV)

    # calc histogram on query image and normalise
    hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    # calc histogram on test image and normalise
    hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # use average of two methods
    ret = float(cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL) + (1-cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)))/float(2)
    if show:
        print "hist: " + str(ret)

    return ret


# template matching
def tmatch(intemp, img2, show=True):
    # grab a portion of query image to use as template
    template = intemp[100:300, 100:300]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # find corners of matched image
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    ret = ((1-math.sqrt((top_left[0]-100)**2 + (top_left[1]-100)**2)/660.0) + (1-math.sqrt((bottom_right[0]-300)**2 + (bottom_right[1]-300)**2)/453.0))/2.0
    if show:
        print "temp: {}".format(ret)
    return ret


# sift
def sift(img1, name1, img2, name2, show=True):

    global g
    # store sift results in G to speed up future tests
    sifted = cv2.SIFT()
    if name1 not in g:
        g[name1] = sifted.detectAndCompute(img1, None)
    if name2 not in g:
        g[name2] = sifted.detectAndCompute(img2, None)

    kp1, des1 = g[name1]
    kp2, des2 = g[name2]

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    if show:
        print "sift: " + str(float(len(good))/len(kp1))
    # use a ratio of good matches to total keypairs as a score
    return float(len(good))/float(len(kp1))


# custom test
def cust(img1, img2, show=True):
    # convert to LAB colorspace
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # grab just the L
    l1 = img1[:, :, 0]
    l2 = img2[:, :, 0]

    # do CLAHE on each L
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img1 = clahe.apply(l1)
    img2 = clahe.apply(l2)

    # calc histograms of each then compare
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    ret = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

    if show:
        print "cust: " + str(ret)
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
            score = 1
        elif setn == sett:
            tscore += 1

        if i == 3:
            break
    if tscore > 1 and score == 1:
        score += tscore
    if show:
        fig.suptitle(tst + " Score( " + str(score) + " )", fontsize=20)
    return score


# run each of the 4 tests for image in path
def runtest(image, path, dirs, show):
    # Open a file
    img = cv2.imread(path+'/'+image, 1)
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # path = "images"

    h = s = t = c = 0
    l = len(dirs)
    if show:
        print str(l) + " images, " + image + " was selected"
    # collection for data
    tlist = {}
    hlist = {}
    clist = {}
    slist = {}

    # init to 0 then do 4 tests
    for fn in dirs:
        tlist[fn] = 0
        hlist[fn] = 0
        clist[fn] = 0
        slist[fn] = 0

        img2 = cv2.imread(path+'/'+fn, 1)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        hlist[fn] = hist(img, img2, show)
        tlist[fn] = tmatch(g1, g2, show)
        slist[fn] = sift(g1, image, g2, fn,  show)
        clist[fn] = cust(img, img2, show)
        if show:
            print fn + "\n"

    h = top4(hlist, path, "histogram", image, show)
    t = top4(tlist, path, "template matching", image, show)
    s = top4(slist, path, "SIFT", image, show)
    c = top4(clist, path, "custom", image, show)



    return h, t, s, c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the query image")
    ap.add_argument("-p", "--path", required=True, help="Path to the directory of images")
    ap.add_argument("-a", "--automate", required=False, help="run tests for all pictures in path", action="store_true")
    ap.add_argument("-o", "--outfile", required=False, help="file to write test stats")

    aarg = ap.parse_args()
    args = vars(aarg)
    image = args["image"]
    path = args["path"]
    output = args["outfile"]

    dirs = os.listdir(path)
    dirs = fnmatch.filter(dirs, 'ukbench*.jpg')
    start = timeit.default_timer()
    if aarg.automate:
        ha = sa = ta = ca = 0
        if output:
            ofile = open(output+".csv", 'w')
            if ofile:
                ofile.write("Image,Histogram,Template Matching,SIFT,Custom\n")
        for i, (f) in enumerate(dirs):
            print i+1,
            (h, t, s, c) = runtest(f, path, dirs, False)
            ha += h
            sa += s
            ta += t
            ca += c
            if output and ofile:
                ofile.write("{0},{1},{2},{3},{4}\n".format(f, h, t, s, c))
        print
        print "{0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(ha/float(len(dirs)), ta/float(len(dirs)), sa/float(len(dirs)), ca/float(len(dirs)))
        if ofile:
            ofile.close()


    else:
        runtest(image, path, dirs, True)



    # show the query image
    if not aarg.automate:
        fig = plt.figure("Query")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("%s" % image)
        # need to convert to RGB for plt
        ax.imshow(cv2.cvtColor(cv2.imread(path+'/'+image, 1), cv2.COLOR_BGR2RGB))
        plt.axis("off")
        # show them
        plt.show()
    print "testing took: " + str(timeit.default_timer()-start)


if __name__ == "__main__":
    main()