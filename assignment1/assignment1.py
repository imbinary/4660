__author__ = 'chris'
# import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    infiles = ['puzzle_1.jpg', 'puzzle_2.png']
    intemps = ['query_1.jpg', 'query_2.png']


    for x in range(0, len(infiles)):

        # 4 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']

        img = cv2.imread(infiles[x], 1)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(intemps[x], 0)
        w, h = template.shape[::-1]

        for meth in methods:
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img2, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # find corners of matched image
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # display plot for saving
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            # plt.show()
            plt.imsave(meth+str(x)+infiles[x], res)

            # Draw rectangle on image where the best score is found
            cv2.rectangle(img, top_left, bottom_right, (127, 0, 127), 2)
            # Display image using OpenCV

            # cv2.imshow('here is Waldo - '+meth, img)
            # cv2.waitKey(0)
            # Save in file using OpenCV
            cv2.imwrite(meth+"_final_"+infiles[x], img)
            cv2.destroyAllWindows()




if __name__ == "__main__":
    main()