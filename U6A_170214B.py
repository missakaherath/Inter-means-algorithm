import cv2 as cv
import numpy as np
import sys


img_name = 'portrait.jpg'
# Read the image
img_rgb = cv.imread(img_name)

if img_rgb is None:
    print("Unable to read the image...")
    sys.exit()

# converting RGB img to gray scale
r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]   
img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

#img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)  # Convert it to grayscale

def get_mean(arr): # get mean value of a list
    tot = 0
    for val in arr:
        tot += val
    return (tot/len(arr))

def partition_img(img, T):
    r1 = []
    r2 = []
    
    for row in img:
        for val in row:
            if val <= T:
                r1.append(val)
            else:
                r2.append(val)
    return r1, r2

def inter_means(img):
    flattenedImg = []
    for row in img:
        for val in row:
            flattenedImg.append(val)
            
    T = get_mean(flattenedImg)  #get initial threshold
    
    print("Initial T :", T)
    
    thresholdVals = []
    
    thresholdVals.append(T)
    
    while(True):
        r1, r2 = partition_img(img, thresholdVals[-1])
        
        mean1 = get_mean(r1)
        mean2 = get_mean(r2)
        
        t = (mean1+mean2)/2

        if(t == thresholdVals[-1]): # optimum threshold found
            print("\nOptimum Threshold value found!, T :", t)
            return(t)
        elif(t in thresholdVals):   # bouncing occured
            print("Bouncing occured...")
            sys.exit()
        else:   # not the optimum threshold
            thresholdVals.append(t)
        print("New T :", t)

def segmentation(img, T):
    segmentedImg = np.empty(shape = img.shape)

    num_rows = len(img)
    num_cols = len(img[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if(img[row, col] <= T):
                segmentedImg[row, col] = 0
            else:
                segmentedImg[row, col] = 255
    return(segmentedImg)

T_val = inter_means(img_gray)

segmentedImg = segmentation(img_gray, T_val)

cv.imshow("Final Output", segmentedImg)
  
cv.waitKey(0)  #waits for user to press any key 

cv.destroyAllWindows() 

cv.imwrite("Segmented - " + img_name, segmentedImg)