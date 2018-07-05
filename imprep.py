# must import if working with opencv in python
import numpy as np
import cv2

# removes pixels in image that are between the range of
# [lower_val,upper_val]
def remove_gray(img,lower_val,upper_val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0,0,lower_val])
    upper_bound = np.array([255,255,upper_val])
    mask = cv2.inRange(gray, lower_bound, upper_bound)
    return cv2.bitwise_and(gray, gray, mask = mask)


# erodes image based on given kernel size (erosion = expands black areas)
def erode( img, kern_size = 3 ):
    retval, img = cv2.threshold(img, 254.0, 255.0, cv2.THRESH_BINARY) # threshold to deal with only black and white.
    kern = np.ones((kern_size,kern_size),np.uint8) # make a kernel for erosion based on given kernel size.
    eroded = cv2.erode(img, kern, 1) # erode your image to blobbify black areas
    y,x = eroded.shape # get shape of image to make a white boarder around image of 1px, to avoid problems with find contours.
    return cv2.rectangle(eroded, (0,0), (x,y), (255,255,255), 1)

# finds contours of eroded image
def prep( img, kern_size = 3 ):    
    img = erode( img, kern_size )
    retval, img = cv2.threshold(img, 200.0, 255.0, cv2.THRESH_BINARY_INV) #   invert colors for findContours
    return cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Find Contours of Image

# given img & number of desired blobs, returns contours of blobs.
def blobbify(img, num_of_labels, kern_size = 3, dilation_rate = 10):
    prep_img, contours, hierarchy = prep( img.copy(), kern_size ) # dilate img and check current contour count.
    while len(contours) > num_of_labels:
        kern_size += dilation_rate # add dilation_rate to kern_size to increase the blob. Remember kern_size must always be odd.
        previous = (prep_img, contours, hierarchy)
        processed_img, contours, hierarchy = prep( img.copy(), kern_size ) # dilate img and check current contour count, again.
    if len(contours) < num_of_labels:
        return (processed_img, contours, hierarchy)
    else:
        return previous

# finds bounding boxes of all contours
def bounding_box(contours):
    bBox = []
    for curve in contours:
        box = cv2.boundingRect(curve)
    bBox.append(box)
    return bBox