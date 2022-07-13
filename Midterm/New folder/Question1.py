import cv2 as cv2
import numpy as np

image = cv2.imread(r'C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Midterm\Q1image.png')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(47,47))
erosion = cv2.erode(image,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
image_edge = cv2.Canny(dilation,100,200)

contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #boundaries
print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(dilation, contours, -1, (0, 255, 0), 7)

cv2.imshow('Contour Image',dilation)
cv2.waitKey(0)