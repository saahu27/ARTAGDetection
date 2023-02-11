

#Sahruday Patti

import numpy as np
import cv2    
import scipy.fftpack
import matplotlib.pyplot as plt



def GaussianMask(image_gray, std_x, std_y):
    cols,rows = image_gray.shape
    center_x, center_y = rows / 2, cols / 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(x,y)
    mask = np.exp(-(np.square((X - center_x)/std_x) + np.square((Y - center_y)/std_y)))
    return mask



def BlurImage_FFT(image_gray):

    fft_image = scipy.fft.fft2(image_gray, axes = (0,1))
    fft_image_shifted = scipy.fft.fftshift(fft_image)
    # magnitude_spectrum_fft_image_shifted = 20*np.log(np.abs(fft_image_shifted))
    Gmask = GaussianMask(image_gray,40,40)
    fft_image_blur = fft_image_shifted * Gmask
    # try:
    #     magnitude_spectrum_fft_image_blur = 20*np.log(np.abs(fft_image_blur))
    # except:
    #     print("Divide by Zero error")


    img_shifted_back = scipy.fft.ifftshift(fft_image_blur)
    img_back_blur = scipy.fft.ifft2(img_shifted_back)
    img_back_blur = np.abs(img_back_blur)
    img_blur = np.uint8(img_back_blur)

    # fx, plts = plt.subplots(2,2,figsize = (15, 10))
    # plts[0][0].imshow(image_gray, cmap = 'gray')
    # plts[0][0].set_title('Gray Image')
    # plts[0][1].imshow(magnitude_spectrum_fft_image_shifted, cmap = 'gray')
    # plts[0][1].set_title('FFT of Gray Image')
    # try:
    #     plts[1][0].imshow(magnitude_spectrum_fft_image_blur, cmap = 'gray')
    #     plts[1][0].set_title('magnitude spectrum of Blurred Gray Image')
    # except:
    #     pass
    # plts[1][1].imshow(img_back_blur, cmap = 'gray')
    # plts[1][1].set_title('Blurred Gray Image')

    return img_blur

def CircularMask(image_size, radius, high_pass = True):
    rows, cols = image_size
    center_x, center_y = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= np.square(radius)

    if high_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask

def Edges_using_fft(thresh):

    fft_thresh_img = scipy.fft.fft2(thresh, axes = (0,1))
    fft_thresh_img_shifted = scipy.fft.fftshift(fft_thresh_img)
    # magnitude_spectrum_fft_thresh_img_shifted = 20*np.log(np.abs(fft_thresh_img_shifted))

    Cmask = CircularMask(thresh.shape, 125, True)
    fft_edge_img = fft_thresh_img_shifted * Cmask
    # try:
    #     magnitude_spectrum_fft_edge_img = 20*np.log(np.abs(fft_edge_img))
    # except:
    #     print("Divide by Zero error")


    edge_img_back_shifted = scipy.fft.ifftshift(fft_edge_img)
    img_back_edge = scipy.fft.ifft2(edge_img_back_shifted)
    img_back_edge = np.abs(img_back_edge)

    # fx, plts = plt.subplots(2,2,figsize = (15,10))
    # plts[0][0].imshow(thresh, cmap = 'gray')
    # plts[0][0].set_title('Thresholded Image')
    # plts[0][1].imshow(magnitude_spectrum_fft_thresh_img_shifted, cmap = 'gray')
    # plts[0][1].set_title('FFT of Thresholded Image')
    # try:
    #     plts[1][0].imshow(magnitude_spectrum_fft_edge_img, cmap = 'gray')
    #     plts[1][0].set_title('magnitude spectrum of Thresholded edge Image')
    # except:
    #     pass
    # plts[1][1].imshow(img_back_edge, cmap = 'gray')
    # plts[1][1].set_title('Edge image')

    return img_back_edge



def Process_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    image_blur = BlurImage_FFT(image_gray)
    ret,thresh = cv2.threshold(image_blur, 220 ,255,cv2.THRESH_BINARY)
    image_edge = Edges_using_fft(thresh)
    return image_gray,image_blur, thresh,image_edge

def features(image_gray,image):

    # kernel = np.ones((5,5),np.uint8)                          #Used in Good features to Track
    # erosion = cv2.erode(image_gray,kernel,iterations = 1)
    # image_dilated = cv2.dilate(erosion,kernel,iterations = 1)
    # corners = cv2.goodFeaturesToTrack(image_dilated, 10,0.1,100)
    # corners = np.int0(corners)

    kernel = np.ones((11,11),np.uint8)
    erosion = cv2.erode(image_gray,kernel,iterations = 1)

    dst = cv2.cornerHarris(erosion,6,7,0.05)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

# find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image_gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    if len(corners) > 8:

        # for i in corners:   # Used in Good features to Track

        #     x, y = i.ravel()
        #     cv2.circle(image, (x, y), 3, 255, -1)
        #     cv2.putText(image, "({},{})".format(x, y), (int(x - 50), int(y - 10) - 20),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        x = []
        y = []

        # for i in range(0,len(corners)):   #Used in Good features to Track
        #     a = corners[i]
        #     x.append(a[0,0])
        #     y.append(a[0,1])

        for i in range(0,len(corners)):
            a = corners[i]
            x.append(int(a[0]))
            y.append(int(a[1]))

        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)

        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)

        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)

        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)


        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,0,255),2)   #Drawing lines for outer rectangle
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,0,255),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,0,255),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,0,255),2)

        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)

        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)

        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)

        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)


        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,0,255),2)  #drawing lines for inner rectangle
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,0,255),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,0,255),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,0,255),2)

        corner_points = np.array(([Ymin_x,Ymin],[Xmin,Xmin_y],[Ymax_x,Ymax],[Xmax,Xmax_y]))

        desired_tag_corner = np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]])

        return image,corner_points,desired_tag_corner

    return image,None,None


def processreftag(ref_tag_image):
    tag_size = 160
    ref_tag_image_gray = cv2.cvtColor(ref_tag_image, cv2.COLOR_BGR2GRAY)
    ref_tag_image_thresh = cv2.threshold(ref_tag_image_gray, 230 ,255,cv2.THRESH_BINARY)[1]
    ref_image_thresh_resized = cv2.resize(ref_tag_image_thresh, (tag_size, tag_size))
    grid_size = 8
    stride = int(tag_size/grid_size)
    grid = np.zeros((8,8))
    x = 0
    y = 0
    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            cell = ref_image_thresh_resized[y:y+stride, x:x+stride]
            if cell.mean() > 255//2:  #determining mean and setting the value
                grid[i][j] = 255
            x = x + stride
        x = 0
        y = y + stride
    inner_grid = grid[2:6, 2:6]
    return inner_grid

def getinfotag(inner_grid):
    count = 0
    while not inner_grid[3,3] and count<4 : #using the count value to rotate corners
        inner_grid = np.rot90(inner_grid,1)
        count+=1

    
    info_grid = inner_grid[1:3,1:3]
    info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
    tag_id = 0
    tag_id_bin = []
    for i in range(0,4):
        if(info_grid_array[i]) :
            tag_id = tag_id + 2**(i)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)
    tag_id_bin.reverse()
    return tag_id, tag_id_bin,count


def computeHomography(corners1, corners2):

    x = corners1[:, 0]
    y = corners1[:, 1]
    xp = corners2[:, 0]
    yp = corners2[:,1]

    nrows = 8
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    return H

def warpPerspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for i in range(maxHeight):
        for j in range(maxWidth):
            f = [i,j,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            xb = np.clip(x/z,0,1919)  #corners are not detected, im clipping the values to stay inside the image
            yb = np.clip(y/z,0,1079)
            # x, y, z = np.matmul(H,f)
            warped[i][j] = img[int(yb)][int(xb)]
    return(warped)


cap = cv2.VideoCapture(r'C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\Project1\1tagvideo.mp4')

tag_size = 160

while(True):
    ret, frame = cap.read()
    if not ret:
        print("Stream ended..")
        break
    image = frame.copy()

    image_gray,image_blur, thresh,image_edge = Process_image(image)
    image_edge = np.uint8(image_edge)

    frame,corner_points,desired_tag_corner = features(image_gray,image)
 
    
    if corner_points is not None:

        H = computeHomography( np.float32(corner_points),np.float32(desired_tag_corner))

        tag = warpPerspective( H, image,tag_size, tag_size)

        tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
        ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
        tag = cv2.cvtColor(tag,cv2.COLOR_GRAY2RGB)
        inner_grid = processreftag(tag)
        tag_id, tag_id_bin,count = getinfotag(inner_grid)
        print(tag_id, tag_id_bin)

        for i in range(count):
            tag = np.rot90(tag)

        try:
            cv2.imshow('frame',tag)
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

cap.release()
cv2.destroyAllWindows()


#For only to see one frame comment the above code till capture and uncomment the below code

# cap = cv2.VideoCapture(r'C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\Project1\1tagvideo.mp4')
# frame_index = 20
# tag_size = 160
# i = 0
# while(True):
#     ret, frame = cap.read()
#     if not ret:
#         print("Stream ended..")
#         break
#     i = i + 1
#     if i == frame_index:
#         frame_chosen = frame
#         break
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('d'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# i = frame_index
# image = frame_chosen.copy()

# image_gray,image_blur, thresh,image_edge = Process_image(image)
# image,corner_points,desired_tag_corner = features(image_gray,image)

# H = computeHomography( np.float32(corner_points),np.float32(desired_tag_corner))
# tag = warpPerspective( H, image,tag_size, tag_size)
# tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
# ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
# tag = cv2.cvtColor(tag,cv2.COLOR_GRAY2RGB)
# inner_grid = processreftag(tag)
# tag_id, tag_id_bin,count = getinfotag(inner_grid)
# for i in range(count):
#     tag = np.rot90(tag)
# print(tag_id, tag_id_bin)

# cv2.imshow('dst',tag)
# if cv2.waitKey(100000) & 0xff == 'd' :
# 	cv2.destroyAllWindows()






