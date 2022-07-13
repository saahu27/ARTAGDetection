import cv2 as cv2
import numpy as np


image_A = cv2.imread('Q2imageA.png')
image_B = cv2.imread('Q2imageB.png')

Sift = cv2.SIFT_create()

Keypoints_A, Descriptors_A = Sift.detectAndCompute(image_A, None)
Keypoints_B, Descriptors_B = Sift.detectAndCompute(image_B, None)


bf = cv2.BFMatcher()
matches = bf.knnMatch(Descriptors_A, Descriptors_B, k = 2)


Good_Matches = []
for i,j in matches:
    if i.distance < 0.7*j.distance:
        Good_Matches.append([i])

Feature_image = cv2.drawMatchesKnn(image_A, Keypoints_A, image_B, Keypoints_B, Good_Matches, flags = 2, outImg = None)

MIN_MATCH_COUNT = 5

if len(Good_Matches) > MIN_MATCH_COUNT:
    srcPoints = np.array([Keypoints_A[i[0].queryIdx].pt for i in Good_Matches])
    dstPoints = np.array([Keypoints_B[i[0].trainIdx].pt for i in Good_Matches])

    H = cv2.findHomography( dstPoints, srcPoints, cv2.RANSAC, 4.0)[0]

    w,h,c = image_B.shape

    image_B_corners = np.array([[0, 0, 1], [0, w, 1], [h, w, 1], [h, 0, 1]])
    warped_corners_Homogeneous = np.dot(H,image_B_corners.T)

    warped_corners = np.int0(np.round(warped_corners_Homogeneous/warped_corners_Homogeneous[2]))

    height = warped_corners[0, 3]

    warped_image = cv2.warpPerspective(image_B, H, (height, w))

    Blank_image = np.zeros((w, height, 3), np.uint8)
    Blank_image[:, :h , :] = image_A

    cv2.fillPoly(Blank_image, [warped_corners[:2, :].T], 0)

    Stiched_image = warped_image + Blank_image

    Stiched_image = cv2.medianBlur(Stiched_image,3)

    cv2.imshow('Stiched', Stiched_image)
    cv2.waitKey(0)

else :
    print( "Not enough matches are found - {}/{}".format(len(Good_Matches), MIN_MATCH_COUNT) )
