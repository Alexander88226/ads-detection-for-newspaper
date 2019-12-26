# Text segmentation 
import cv2
import sys
import os
import json
import numpy as np
import math
import imutils
import pdf2image
input_dir  = sys.argv[1]

for filename in os.listdir(input_dir):
   filepath = input_dir + "/" + filename
   fileinfos = os.path.splitext(filename)      
   pdf_file = filepath
   directory = fileinfos[0]

   #create directory
   if not os.path.exists(directory):
      os.makedirs(directory)
   all_boxes = []

   pages = pdf2image.convert_from_path(pdf_file, 500)
   for page in pages:
      filepath = directory + '/' + directory + '.jpg'
      page.save(filepath, 'JPEG')


   # read image
   img = cv2.imread(filepath)
   height, width, channels = img.shape


   # convert bgr image to gray image
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # grayscale

   # thresholding
   _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold

   # remove noise of border
   thresh = cv2.rectangle(thresh, (0, 0), (width, height), (0, 0, 0), 100)

   # connect splitted line of articles
   kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,1))
   dilated = cv2.dilate(thresh,kernel,iterations = 1) # dilate

   # find contours in the image and initialize the mask that will be
   cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   mask = np.ones(img.shape[:2], dtype="uint8") * 255

   # find top edget and bottom edge of article
   top_edge = 0
   bottom_edge = height
   for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      if (w > width / 2 and h < 50):
         if y < height / 10 and y > top_edge:
            top_edge = y          
         if y > height * 9 / 10:
            bottom_edge = y

   # loop over the contours
   # remove the meaningless area(Photo area) and top and bottom area
   paragraph_line = []
   for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      density = float(area)/(w*h)

      # if the contour is bad, draw it on the mask
      if (w > width / 2 and h < height / 150):
         cv2.drawContours(mask, [c], -1, 0, -1)
      if w < width / 10 and h > height / 10:
         cv2.drawContours(mask, [c], -1, 0, -1)          
      if (w > width / 2 and h > height / 10):
         if y < bottom_edge and density > 0.5:
            bottom_edge = y          
         cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 0), -1)
      cv2.rectangle(mask, (0, 0), (width, top_edge), (0, 0, 0), -1)
      cv2.rectangle(mask, (0, bottom_edge), (width, height), (0, 0, 0), -1)          

   image = cv2.bitwise_and(dilated, dilated, mask=mask)

   # get text roi as the criteria of splitted line 

   # first, get upper text roi, second get lower text roi
   # 1. upper text roi is from removed the area - top y = split line.y - gap * 2, bottom y = split line.y
   # 2. lower text roi is from removed the area - top y = split line.y, bottom y = split line.y - gap * 2
   gap = int(height / 70)
   cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   mask_upper = np.ones(img.shape[:2], dtype="uint8") * 255
   mask_lower = np.ones(img.shape[:2], dtype="uint8") * 255

   for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      density = float(area)/(w*h)
      if density > 0.9:
         cv2.drawContours(mask_upper, [c], -1, 0, -1)                
         cv2.drawContours(mask_lower, [c], -1, 0, -1)                

      if (w < width / 2 and w > width / 10 and h < 50):
         paragraph_line.append([x, y, w, h])
         cv2.rectangle(mask_upper, (x, y-2 * gap), (x+w, y+h), (0, 0, 0), -1)
         cv2.rectangle(mask_lower, (x, y), (x+w, y+h+2 * gap), (0, 0, 0), -1)
         # cv2.drawContours(mask, [c], -1, 0, -1)

   image_upper = cv2.bitwise_and(image, image, mask=mask_upper)
   image_lower = cv2.bitwise_and(image, image, mask=mask_lower)

   # dilate the above two text roi areas
   # horizontal length is 20
   # vertical length is 80
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 80))
   extend_upper = cv2.morphologyEx(image_upper, cv2.MORPH_CLOSE, kernel)
   extend_lower = cv2.morphologyEx(image_lower, cv2.MORPH_CLOSE, kernel)

   # merge two areas
   extend = cv2.bitwise_or(extend_lower, extend_upper)

   # get valid text rois
   _, cnts, hierarchy,= cv2.findContours(extend.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   for cnt in cnts:
      x, y, w, h = cv2.boundingRect(cnt)
      if w > width / 10:
         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)
         all_boxes.append({"points": [[x, y], [x, y+h], [x+w, y], [x+w, y+h]]})
   #creating the json and output file
   with open(directory+"/data_file.json", "w") as write_file:
      temp = {"boxes":all_boxes}
      json.dump(temp, write_file)
   cv2.imwrite(directory+"/contour.png", img)
   print("processed " + directory)
