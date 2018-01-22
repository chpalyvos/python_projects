from __future__ import print_function
import keras,os,cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import load_model


#from sklearn import mixture
###########
def y(y_true,y_pred):
	print  (K.variable(value=model.get_layer(index=2).get_weights()[0])	)
	return keras.losses.categorical_crossentropy


##########
batch_size = 30
num_classes = 2
epochs = 15

# input image dimensions
img_rows, img_cols = 70,70
img_size=img_rows
img_size_flat=img_size*img_size
input_shape = (img_rows, img_cols, 3)
x_train=[]
y_train=[]
data_dir_prod_tr = '/home/christos/Documents/pr_s/train/CroppedFolder/Product/Trainn_new/'
data_dir_non_prod_tr ='/home/christos/Documents/pr_s/train/CroppedFolder/Non_Product/Trainn_new/'
for fn in os.listdir(data_dir_prod_tr):
	y_train.append(np.array([1,0]))
	im=cv2.imread(data_dir_prod_tr+fn)
	im=im.reshape((img_size_flat,3))
	x_train.append(im)
	im=[]
for fn in os.listdir(data_dir_non_prod_tr):
	y_train.append(np.array([0,1]))
	im=cv2.imread(data_dir_non_prod_tr+fn)
	im=im.reshape((img_size_flat,3))
	x_train.append(im)
	im=[]

x_valid=[]
y_valid=[]
data_dir_prod_val = '/home/christos/Documents/pr_s/train/CroppedFolder/Product/Validd_new/'
data_dir_non_prod_val ='/home/christos/Documents/pr_s/train/CroppedFolder/Non_Product/Validd_new/'
for fn in os.listdir(data_dir_prod_val):
	y_valid.append(np.array([1,0]))
	im=cv2.imread(data_dir_prod_val+fn)
	im=im.reshape((img_size_flat,3))
	x_valid.append(im)
	im=[]
for fn in os.listdir(data_dir_non_prod_val):
	y_valid.append(np.array([0,1]))
	im=cv2.imread(data_dir_non_prod_val+fn)
	im=im.reshape((img_size_flat,3))
	x_valid.append(im)
	im=[]

y_train=np.asarray(y_train)
y_valid=np.asarray(y_valid)

x_train=np.asarray(x_train)
print (x_train.shape)
s1=x_train.shape[0]
s2=x_train.shape[1]
x_train=x_train.reshape((s1,int(s2**.5),int(s2**.5),3))
print (x_train.shape)
x_valid=np.asarray(x_valid)
s1=x_valid.shape[0]
s2=x_valid.shape[1]
x_valid=x_valid.reshape((s1,int(s2**.5),int(s2**.5),3))






x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
					shuffle=True,
          verbose=1,
          validation_data=(x_valid, y_valid))

model.save('./my__binary_model_rgb_new_new.h5')

