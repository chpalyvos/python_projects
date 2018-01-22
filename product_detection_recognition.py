#######
# Into a directory you must have these folders:
# 'InputFolder' :with the input images
# 'Outputs' : in this folder there will be the images with the recognized products, so have it empty



from __future__ import print_function
import keras,os,cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import load_model
import glob
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3
import dlib
if PY3:
	xrange = range

import numpy as np
import cv2
from numpy import array
import json
import itertools

color_dict={
						'altis': 'green',
						'omo' : 'red',
						'skip' : 'blue',
						'unknown': 'black'						
						}

def found_sure(a):
	return max(a)>.51  # threshold parameter to consider a label as the master-label
	
### A function to draw the classified objects on the image & give the Output folder
def second_guess(probs,i,x):
	p1=np.argmax(x)
	probs[p1]=0
	n2=np.argmax(probs)
	c2=maxim(probs)
	return c2+'-->'+str(probs[n2])
def maxim(pred):
	if np.argmax(pred)==3:
		return 'unknown'
	elif np.argmax(pred)==0:
		return 'altis'
	elif np.argmax(pred)==1:
		return 'omo'
	elif np.argmax(pred)==2:
		return 'skip'


def drawrects(name,coords,labels,predictions,classes):
	source_img = Image.open(name).convert("RGBA")
	for i in range(len(coords)):
		draw = ImageDraw.Draw(source_img)
		x=coords[i]
		fnt = ImageFont.truetype('arial.ttf', 25)
		draw.rectangle(((x[0], x[1]), (x[2], x[3])),outline = color_dict[labels[i]])
		draw.rectangle(((x[0], x[1]), (x[0]+200,x[1]+30)),fill=(0,150,0),outline = color_dict[labels[i]])
		y=predictions[i,np.argmax(dictionary[labels[i]])]
		y=float('.'.join([str(y).split('.')[0],str(y).split('.')[-1][:2]]))
		draw.text((x[0], x[1]), labels[i]+'-->'+str(y),font=fnt,fill=(255,255,255,255))
#+'\n'+second_guess(predictions[i],i,classes[i]))
	nm = (name.split('/')[-1]).split('.')[0]
	print (nm)
	source_img.save(x_dir+'Outputs/'+nm+'.jpeg', "JPEG")


def max_distance(a):
	a1,a2,a3=a[0],a[1],a[2]
	return max(abs(a1-a2),abs(a2-a3),abs(a3-a1)) < .35

def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)[::-1]
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))	
	return boxes[pick].astype("int")


def P_MARAG(img):
	kernel=np.array([[0,0,0,0,1,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,1,1,1,1,1,0,0], [0,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,0], [0,0,1,1,1,1,1,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,1,0,0,0,0]], dtype=np.uint8)
	img_d=cv2.dilate(img,kernel)
	img_e=cv2.erode(img,kernel)
	img = img_d-img_e
	return img

def find_squares(img,e,thr_type):
	image = img
	image = cv2.bilateralFilter(image,5,200,200)
	image = cv2.filter2D(image,-1,np.array([ [-1,-1,-1], [-1,9,-1],  [-1,-1,-1] ]))
	for gray in cv2.split(image):
		for thrs in xrange(0, 255, 26):
			if thrs == 0:
				bin = P_MARAG(gray)
				bin = cv2.dilate(bin, None)
			else:
				_retval, bin = cv2.threshold(gray, thrs, 255, thr_type)

				bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				for cnt in contours:
					cnt_len = cv2.arcLength(cnt, True)
					cnt = cv2.approxPolyDP(cnt, e*cnt_len, True)
					if len(cnt) == 4 and (cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 20000) and cv2.isContourConvex(cnt):
						[x,y,w,h] = cv2.boundingRect(cnt)
						objects.append([x,y,x+w,y+h])


def im7070(x):
	return cv2.resize(x,(70,70),cv2.INTER_AREA)
def coords2img(coords,nm):
	img=cv2.imread(nm)
	objects=[]
	for x in coords:
		ob = img[x[1]:x[3],x[0]:x[2],:]
		ob = im7070(ob)
		objects.append(ob)
	return np.asarray(objects)


def draw_templates(i_min,centres):
	grid_binary_table[(-1)*l,0]=True
	grid[y_axis[(-1)*l]+margin:y_axis[(-1)*l]+200-margin,x_axis[0]+margin:x_axis[0]+200-margin]=diction[labels[i_min]]
	indexes=find_close(i_min,centres)		
	centres_close=centres[indexes]
	labels_close = np.asarray(labels)[indexes]
	k=1
	indexes_sort=np.argsort(centres_close[:,0]).tolist()
	centres_close=centres_close[indexes_sort]
	labels_close=labels_close[indexes_sort]
	centres_close = centres_close.tolist()
	while k<=min(9,len(indexes)):
		grid_binary_table[(-1)*l,k]=True
		grid[y_axis[(-1)*l]+margin:y_axis[(-1)*l]+200-margin,x_axis[k]+margin:x_axis[k]+200-margin]=diction[labels_close[k-1]]
		k+=1
	centres=centres.tolist()
	return centres_close
	


def find_close(k,centres):
	idxs=[]
	y=centres[k][1]
	for i,item in zip(range(centres.shape[0]),centres): 	
		#print(abs(item[1]-y))
		if abs(item[1]-y)<150:
			idxs.append(i)
			#print (idxs)
	idxs.remove(k)
	return idxs

if __name__ == '__main__':
	from glob import glob
	idx=0
	ccc=0
	x_dir='/home/christos/Documents/pr_s/train/data/final/'
	binary_model = load_model('/home/christos/Documents/pr_s/train/CroppedFolder/my__binary_model_rgb_new_new.h5')
	model = load_model('/home/christos/Documents/pr_s/train/CroppedFolder/my_model_rgb_with_new_new.h5')	
	images_dir = '/home/christos/Documents/pr_s/train/data/final/Input/'
	
	altis_templ=cv2.imread(x_dir+'Templates/'+'altis.jpeg')
	unknown_templ=cv2.imread(x_dir+'Templates/'+'gray.jpeg')
	skip_templ=cv2.imread(x_dir+'Templates/'+'skip.jpeg')
	omo_templ=cv2.imread(x_dir+'Templates/'+'omo.jpeg')
	
	margin=10	
	diction = {
							'altis'		:		altis_templ[margin:(-1)*margin,margin:(-1)*margin],
							'omo'			:		omo_templ[margin:(-1)*margin,margin:(-1)*margin],
							'skip'		:		skip_templ[margin:(-1)*margin,margin:(-1)*margin],
							'unknown'	:		unknown_templ[margin:(-1)*margin,margin:(-1)*margin]
							}
	for fn in sorted(glob('./Input/Image*.jpeg')):
		print ('I am in file  ----> '+str(fn))
		kl=((fn.split('/')[-1]).split('.')[0]).split('e')[-1]
		ccc+=1
		idx+=1
		image = cv2.imread(fn)
		image1 = cv2.imread(fn)
		image2 = cv2.imread(fn) 
		objects=[]
		squares=[]
		coords=[]
		for e in np.arange(.04,.066,.005): ## run the edge detection for more parameters to detect many ROI's 
			for thr_type in [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TOZERO,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.ADAPTIVE_THRESH_GAUSSIAN_C]:
				find_squares(image,e,thr_type)
		coords=objects
		rects=[]
			#### TO ADD DLIB ROI'S
			#### DLIB CODE  |
#											|
#											|
#											|
#											|
#										 \/
		#dlib.find_candidate_object_locations(image2, rects, min_size=1500)
		#rectList=[[rects[i].left(),rects[i].top(),rects[i].right(),rects[i].bottom()] for i in range(len(rects))]
		#coords=coords+rectList
		print ('found: '+str(len(coords))+' objects')	
		objects = coords2img(coords,fn)
						
#   --------->                    ALL OBJECTS ARE FOUND, OVERLAPPED AND NOT              <---------

		objects=objects.astype('float32')
		objects/=255
		binary_predictions = binary_model.predict(objects)
		indexes=[]
		products=[]
		for i in range(objects.shape[0]):
			if np.argmax(binary_predictions[i])==0:						
				products.append(objects[i])
			else:
				indexes.append(i)
		
		products=np.asarray(products)
		print ('found: '+str(products.shape[0])+' products')		
		print (len(coords))
		map(lambda x: coords.pop(x), sorted(indexes, key = lambda x:-x))

#   --------->                    ALL PRODUCTS ARE FOUND, OVERLAPPED AND NOT              <---------

		print (len(coords))
		products=products.astype('float32')
		products/=255
		probs = model.predict(products)
		probs[:,3]=0
		probs = [max(item) for item in probs]
		objects=non_max_suppression_fast(np.asarray(coords), .3) ## parameter: overlap theshold (as a percentage (<1)) to delete objects  
# with greater overlap percentage
#   ---------->                   FOUND NON_OVERLAPPED COORDINATES OF PRODUCTS             <----------

		coords=objects
		print ('found: '+str(coords.shape[0])+' non-overlapped products')	
		objects = coords2img(coords,fn)
		objects = objects.astype('float32')
		objects /=255
		# run the model
		predictions = model.predict(objects) #predictions
		labels=[]
		classes=[]
		# a dictionary to match classes-CNN output
		dictionary = {
									'altis'  :[1,0,0,0],
									'omo'    :[0,1,0,0],
									'skip'   :[0,0,1,0],
									'unknown':[0,0,0,1]
									}
		#for each prediction
		for pred in predictions:
			if found_sure(pred): # if there is a big enough probability, classify directly it 
				labels.append(maxim(pred))
				classes.append(dictionary[maxim(pred)])
				continue

			if pred[-1]>.1 or max_distance(pred) : # if the unknown class gives a non-small enough probability 
																							 # OR the other given probabilities are close enough to each other
																							 # --> give 'unknown' prediction 
				labels.append('unknown')
				classes.append(dictionary['unknown'])
				continue
			labels.append(maxim(pred))
			classes.append(dictionary[maxim(pred)])
		name = images_dir + fn.split('/')[-1]
		# having predicted all object classes, draw them on the input image
		drawrects(name,coords,labels,predictions,classes)


		#### PRINT TEMPLATES TO GRID	
		### TO BE READY BY MONDAY	

		grid = cv2.imread(x_dir+'grid.jpeg')
		centres = np.asarray([((coords[i][0]+coords[i][2])/2 ,(coords[i][1]+coords[i][3])/2) for i in range(coords.shape[0])])
		grid_binary_table = np.asarray([[False]*10]*6)
		x_axis=[0,211,421,629,839,1047,1257,1465,1677,1885]
		y_axis = [0,209,417,633,847,1051]
		l=1
		while l<=6 and not(centres==[]):
			centres=np.asarray(centres)	
			low_left_corner = [0,image.shape[0]]
			min_dist = 100000000
			for i,item in zip(range(centres.shape[0]),centres):
				dist= abs(item[0]-low_left_corner[0])+abs(item[1]-low_left_corner[1]) # in pixels
				if dist<min_dist:
					min_dist=dist
					i_min=i
			centres_close=draw_templates(i_min,centres)
			centres=centres.tolist()
			centres_close.append(centres[i_min])
			for it in centres_close:
					p=centres.index(it)
					del labels[p]
					centres.remove(it)
			l+=1
		cols_to_delete=0
		ind=-1
		flag_cols=True
		while flag_cols:
			col = grid_binary_table[:,ind]
			for val in col:
				if val:
					flag_cols=False
					break
			if flag_cols:
				cols_to_delete+=1
			ind-=1
		rows_to_delete=0
		ind=0
		flag_rows=True
		while flag_rows:
			row = grid_binary_table[ind,:]
			for val in row:
				if val:
					flag_rows=False
					break
			if flag_rows:
				rows_to_delete+=1
			ind+=1
		grid=np.delete(grid,range(x_axis[(-1)*cols_to_delete],grid.shape[1]),1)
		grid=np.delete(grid,range(0,y_axis[rows_to_delete]),0)
		cv2.imwrite(x_dir+'/final/'+'final'+str(kl)+'.png',grid)		
		

