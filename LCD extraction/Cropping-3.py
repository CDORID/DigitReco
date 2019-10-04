# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:56:07 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:13:06 2019
​
@author: user
"""


import cv2
import numpy as np
import imutils
import math




def our_max(i):   
   if len(i) == 0 :  
       total_max = np.reshape(np.array([1,1,2,20]),(2,2))
       
   if len(i) != 0 :
       col1 = i[:,1]
       col1_max = max(col1)
       total_max = i[np.where(col1 == col1_max)]
       
   if len(total_max) == 1: 
       added = np.reshape(np.array([2,20]),(1,2))
       total_max =np.append(total_max,added,axis = 0)
   return total_max[1]

def our_min(i):
   
   if len(i) == 0 :  
       total_min = np.reshape(np.array([1,1,2,20]),(2,2))
       
   if len(i) != 0 :
       col1 = i[:,1]
       col1_min = min(col1)
       total_min = i[np.where(col1 == col1_min)]
       
   if len(total_min) == 1: 
       added = np.reshape(np.array([1,10]),(1,2))
       total_min =np.append(total_min,added,axis = 0)
   return total_min[1]

def create_point(image,y,x,size_point=2):
   height, width, c = image.shape
   border_rect = (x-int(round(width/400)),y-int(round(width/400)))
   border_rect2 = (x+int(round(width/400)),y+int(round(width/400)))
   img = cv2.rectangle(image,border_rect,border_rect2,(0,250,0),3)
   return img
   
def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def remove_close_points(centroids,max_point,img,removal_parameter = 0.2):
    height, width, c = img.shape
    possible_points = []
    for i in centroids:
      if (np.abs(i[0]-max_point[0]) > removal_parameter*height):
        possible_points.append(i)
        
    possible_points = np.array(possible_points)  
    return possible_points
    


def first_cropping(image, resize_width = 70,crop_crop = 0.05):
    
    first_height, first_width, c = image.shape
    #Adding borders for better angle selection
    
    cropping_width = int(round(crop_crop*first_width))
    cropping_height = int(round(crop_crop*first_height))
    
    image = image[cropping_height:first_height-cropping_height, 
                  cropping_width:first_width-cropping_width]
    img = cv2.copyMakeBorder(np.array(image), 0, 0, 50, 50, cv2.BORDER_CONSTANT, value = [0,0,0] )
    cropped = cv2.copyMakeBorder(np.array(image), 0, 0, 50, 50, cv2.BORDER_CONSTANT, value = [0,0,0] )
    
    
    origin_height, origin_width, c = img.shape
    transformation_ratio = origin_width/resize_width 
    img = imutils.resize(img, width = resize_width)
#     light preprocessing: grey
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ##passing in float32 for function corner harris
    gray = np.float32(gray)
      
    #### OPTI PARA #######################################################################"""
    dst = cv2.cornerHarris(gray,2,3,0.001)    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    # dst = metric of cornerhood
    img[dst>0.02*dst.max()]=[0,0,255]
    
    pointed = img
    dst = np.uint8(dst)

#     identify where the red (0,0,255) points are
    coord = np.where(np.all(img == (0, 0, 255), axis=-1))    
    centroids = np.array(coord).T
    
    max_left = our_min(centroids)
    max_right = our_max(centroids)
    
    possible_points_left =  remove_close_points(centroids,max_left,img)
    possible_points_right = remove_close_points(centroids,max_right,img)
        
    possible_points_right = np.array(possible_points_right)
    
    second_max_left = our_min(possible_points_left)
    second_max_right = our_max(possible_points_right)
#     print('max left:' + str(max_left))


        
    rect2 = np.zeros((4, 2), dtype = "float32")
    rect2[0] = max_left*transformation_ratio
    rect2[1] = second_max_left*transformation_ratio
    rect2[2] = max_right*transformation_ratio
    rect2[3] = second_max_right*transformation_ratio

    
    rect4 = np.zeros((4, 2), dtype = "float32")
    rect4[:,0] = rect2[:,1]
    rect4[:,1] = rect2[:,0]
    
    A,B,C,D = crop_correcter(rect4)
    rect2[0] = A
    rect2[1] = B
    rect2[2] = C
    rect2[3] = D

    
    

    warped = four_point_transform(cropped, rect4)
    
    print("_")   
    print("_")
    return warped, pointed

def distance(pointA,pointB):
    dist = math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)
    return dist

def get_degrees(segment1,segment2,segment3):
    #get angle between segment1 and 2
    radius = (segment1 * segment1 + segment2 * segment2 - segment3 * segment3)/(2.0 * segment1 * segment2)
    if radius > 1 :
        angle = 180 - math.degrees(math.acos(radius - 2 ))
        print('-2')
    elif radius < -1 : 
        angle = 180 - math.degrees(math.acos(radius + 2 ))
        print('+2')
    else :
        angle = math.degrees(math.acos(radius))
    
    return angle
    
def point_orderer(rect4):
    if rect4[0][1] < rect4[1][1] :
        A = rect4[0]
        B = rect4[1]
    else :
        A = rect4[1]
        B = rect4[0]
        
    if rect4[2][1] < rect4[3][1] :
        C = rect4[3]
        D = rect4[2]
    else :
        C = rect4[2]
        D = rect4[3]   
        
    print(A,B,C,D)
    return A,B,C,D
    
def correct_point(opposed,sided1,sided2,correct_point):
    correct_point[0] = sided1[0] + sided2[0] - opposed[0]
    correct_point[1] = sided1[1] + sided2[1] - opposed[1]
    return correct_point

def crop_crop_strategy(A,B,C,D, tolerance, ratio_tolerance):

    #compute sides
    AB = distance(A,B)
    BC = distance(B,C)
    CD = distance(C,D)
    AD = distance(D,A)
    
    #compute diags
    AC = distance(A,C)
    BD = distance(B,D)
    
    print(AB,BC,CD,AD)
    #print(AB,BC,CD,AD)
    
    Angle_A = get_degrees(AB,AD,BD)
    Angle_B = get_degrees(AB,BC,AC)
    Angle_C = get_degrees(BC,CD,BD)
    Angle_D = get_degrees(AD,CD,AC)
    
    print(Angle_A,Angle_B,Angle_C,Angle_D)
    
    ratio_A = AD/AB*abs(Angle_A-90)/100
    ratio_B = BC/AB*abs(Angle_B-90)/100
    ratio_C = BC/CD*abs(Angle_C-90)/100
    ratio_D = AD/CD*abs(Angle_D-90)/100
    
    print(ratio_A,ratio_B,ratio_C,ratio_D)
    
    
    
    if abs(Angle_A - Angle_C) > tolerance:
        print('difference AC')
        if abs(Angle_A -90) > abs(Angle_C - 90):
            
            ## scenario corret A
            if abs(ratio_A-ratio_tolerance) > abs(ratio_B-ratio_tolerance) and abs(ratio_A-ratio_tolerance) > abs(ratio_D-ratio_tolerance): 
                A = correct_point(C,B,D,A)
                print('correct A')
            
            
        else :
            ## scenario correect C
            if abs(ratio_C-ratio_tolerance) > abs(ratio_B-ratio_tolerance) and abs(ratio_C-ratio_tolerance) > abs(ratio_D-ratio_tolerance)  :
                C = correct_point(A,B,D,C)
                print('correct C')
          
            
    if abs(Angle_B - Angle_D) > tolerance:
        print('difference DB')
        if abs(Angle_B -90) > abs(Angle_D - 90):
            
            ## scenario corret B
            if abs(ratio_B-ratio_tolerance) > abs(ratio_C-ratio_tolerance) and abs(ratio_B-ratio_tolerance) > abs(ratio_A-ratio_tolerance):
                B = correct_point(D,A,C,B)
                print('correct B')
            
        else :
            ## scenario correect D
            if abs(ratio_D-ratio_tolerance) > abs(ratio_C-ratio_tolerance) and abs(ratio_D-ratio_tolerance) > abs(ratio_A-ratio_tolerance) :
                D = correct_point(B,A,C,D)
                print('correct D')

    else: 
        print('fine cropping')
        
            

    print(Angle_A,Angle_B,Angle_C,Angle_D)
    
    return A,B,C,D
    
    

def crop_correcter(rect4, tolerance = 0, ratio_tolerance = 0):
    
    A,B,C,D = point_orderer(rect4)
    
    A,B,C,D = crop_crop_strategy(A,B,C,D,tolerance = tolerance, ratio_tolerance = ratio_tolerance)

    return A,B,C,D

