# Text segmentation 
import cv2
import sys
import os
import json
import numpy as np
import math
import imutils
import pdf2image
import matplotlib.pyplot as plt

input_dir  = sys.argv[1]
def horizontalProjection(img):
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(h):
        row = img[j:j+1, 0:w] # y1:y2, x1:x2
        sumCols.append(np.sum(row))
    return sumCols

def verticalProjection(img):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols

for filename in os.listdir(input_dir):
   filepath = input_dir + "/" + filename
   fileinfos = os.path.splitext(filename)      
   pdf_file = filepath
   directory = fileinfos[0]

   #create directory
   if not os.path.exists(directory):
      os.makedirs(directory)
   all_boxes = []

   pages = pdf2image.convert_from_path(pdf_file, 400)
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

   thresh = cv2.rectangle(thresh, (0, 0), (width, height), (0, 0, 0), 100)

   hist = verticalProjection(thresh)
   # hist = [0 if x > height * 255 / 5 else x for x in hist]
   median = np.median(hist)
   minimum = np.min(hist)
   th_val = (median - minimum)/3
   plt.plot(hist)

   sx = 0
   ex = 0
   th_lenght  = width / 10

   hlines = []
   flag = False
   while sx < width - 1 and ex < width - 1:
      while hist[sx] < th_val and sx < width - 1:
         sx += 1
      ex = sx    
      while hist[ex] >= th_val and ex < width - 1:
         ex += 1
      if (ex - sx) > th_lenght:
         hlines.append([sx, ex])
      sx = ex
   

   ex = hlines[0][0]
   thresh = cv2.rectangle(thresh, (0, 0), (ex + 10, height), (0, 0, 0), -1)
   for line in hlines:
      plt.axvline(x= line[0], color='r', linestyle='-')
      thresh = cv2.rectangle(thresh, (max(ex - 10, 0), 0), (min(line[0] + 10, width), height), (0, 0, 0), -1)
      ex = line[1]
      plt.axvline(x= line[1], color='g', linestyle='-')
   thresh = cv2.rectangle(thresh, (ex - 10, 0), (width, height), (0, 0, 0), -1)

   # plt.show()

   cv2.imwrite(directory+"/thresh.png", thresh)
   # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
   # erod = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
   # connect splitted line of articles
   kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,1))
   dilated = cv2.dilate(thresh,kernel,iterations = 1) # dilate
#    cv2.imwrite(directory+"/dilated.png", dilated)
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

   # print(top_edge)
   # print(bottom_edge)

   # loop over the contours
   # remove the meaningless area(Photo area) and top and bottom area
   width_list = []
   hegith_list = []
   for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      density = float(area)/(w*h)
      width_list.append(w)     
      hegith_list.append(h)   
      # if the contour is bad, draw it on the mask
      if (w > width / 2 and h < height / 150):
         cv2.drawContours(mask, [c], -1, 0, -1)
      if w < width / 10 and h > height / 10:
         cv2.drawContours(mask, [c], -1, 0, -1)
      if (w > width / 2 and h > 50 and w / h > 5):
         if y < bottom_edge and density > 0.7 and y > height / 2:
            bottom_edge = y
         if y + h > top_edge and density > 0.7 and y < height / 2:
             top_edge = y + h
         cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 0), -1)
      cv2.rectangle(mask, (0, 0), (width, top_edge), (0, 0, 0), -1)
      cv2.rectangle(mask, (0, bottom_edge), (width, height), (0, 0, 0), -1)    

   # print(bottom_edge)
   # print(top_edge)
   width_median = np.median(np.array(width_list))       
   height_median = np.median(np.array(hegith_list))
   # print(width_median)       
   # print(height_median)
   image = cv2.bitwise_and(dilated, dilated, mask=mask)
#    cv2.imwrite(directory+"/mask.png", mask)
#    cv2.imwrite(directory+"/image.png", image)
   # get text roi as the criteria of splitted line 

   # first, get upper text roi, second get lower text roi
   # 1. upper text roi is from removed the area - top y = split line.y - gap * 2, bottom y = split line.y
   # 2. lower text roi is from removed the area - top y = split line.y, bottom y = split line.y - gap * 2
   gap = int(height_median * 3)
   cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   mask_upper = np.ones(img.shape[:2], dtype="uint8") * 255
   mask_lower = np.ones(img.shape[:2], dtype="uint8") * 255


   paragraph_line = []

   for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      density = float(area)/(w*h)
      if density > 0.9 and w > int(width_median * 10) and h > int(height_median * 3):
         cv2.drawContours(mask_upper, [c], -1, 0, -1)                
         cv2.drawContours(mask_lower, [c], -1, 0, -1)                

      if (w < width / 2 and w > width / 10 and h < 50):
         paragraph_line.append([x, y, w, h])
         cv2.rectangle(mask_upper, (x, y-2 * gap), (x+w, y+h), (0, 0, 0), -1)
         cv2.rectangle(mask_lower, (x, y), (x+w, y+h+2 * gap), (0, 0, 0), -1)
         # cv2.drawContours(mask, [c], -1, 0, -1)

   image_upper = cv2.bitwise_and(image, image, mask=mask_upper)
   image_lower = cv2.bitwise_and(image, image, mask=mask_lower)
#    cv2.imwrite(directory+"/mask_upper.png", mask_upper)
#    cv2.imwrite(directory+"/mask_lower.png", mask_lower)

#    cv2.imwrite(directory+"/image_lower.png", image_lower)
#    cv2.imwrite(directory+"/image_upper.png", image_upper)

   # dilate the above two text roi areas
   # horizontal length is 20
   # vertical length is 80
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width_median * 3), int(height_median * 7)))
   extend_upper = cv2.morphologyEx(image_upper, cv2.MORPH_CLOSE, kernel)
   extend_lower = cv2.morphologyEx(image_lower, cv2.MORPH_CLOSE, kernel)
   cv2.imwrite(directory+"/extend_upper.png", extend_upper)
   cv2.imwrite(directory+"/extend_lower.png", extend_lower)

   # merge two areas
   extend = cv2.bitwise_or(extend_lower, extend_upper)
#    cv2.imwrite(directory+"/extend.png", extend)

   # get valid text rois
   _, cnts, hierarchy,= cv2.findContours(extend.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   for cnt in cnts:
      x, y, w, h = cv2.boundingRect(cnt)
      if w > width / 10 and y + h > height / 10:
         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)
         all_boxes.append({"points": [[x, y], [x, y+h], [x+w, y], [x+w, y+h]]})
   #creating the json and output file
   with open(directory+"/data_file.json", "w") as write_file:
      temp = {"boxes":all_boxes}
      json.dump(temp, write_file)
   cv2.imwrite(directory+"/contour.png", img)
   print("processed " + directory)
