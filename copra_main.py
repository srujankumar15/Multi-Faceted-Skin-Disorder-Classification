import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras.optimizers import RMSprop   
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import ReduceLROnPlateau
import cv2
import glob as gb

%matplotlib inline

path="C:/Multi Faceted Skin Disorder Classification/train"

code={'Acne and Rosacea Photos':0,'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions':1,'Atopic Dermatitis Photos':2, 'Eczema Photos':3, 'Nail Fungus and other Nail Disease':4, 'Psoriasis pictures Lichen Planus and related diseases':5,'Monkeypox':6}
def getcode(n):
    for x,y in code.items():
        if n==y:
            return x

for folder in os.listdir(path):   
    files=gb.glob(str(path+'/'+folder+'/*.jpg'))
    print(f"for traning data found,{len(files)} in folder {folder}")


x_train=[]
y_train=[]
for folder in os.listdir(path):# path  
    files=gb.glob(str(path+'/'+folder+'/*.jpg'))#
    for file in files:#ر 
        image=cv2.imread(file)# 
        image_array=cv2.resize(image,(224,224))#ة
        x_train.append(image_array)# 
        y_train.append(code[folder])#  

print(len(x_train),len(y_train))#

plt.imshow(x_train[200])#
y_train[200]#

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.33, random_state=44, shuffle=True)#

X_train=np.array(x_train)
Y_train=np.array(y_train)
X_test=np.array(x_test)
Y_test=np.array(y_test)

X_train=X_train/255.0
X_test=X_test/255.0

plt.figure(figsize=(15,10))
for n,i in enumerate(list(np.random.randint(0,len(X_train),12))):
    plt.subplot(3,4,n+1)
    plt.imshow(X_train[i])
    plt.axis("off")
    plt.title(getcode(Y_train[i]))

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
learning_rate_reduction=ReduceLROnPlateau(monitor="val_loss",patience=2
                                          ,verbose=1,factor=0.2
                                          ,min_lr=0.00001)

input_shape = (224,224, 3)
model1 = Sequential()

model1.add(Conv2D(64, (3, 3), input_shape=input_shape))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(16))
model1.add(Dense(7, activation='softmax'))
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()

hist=model1.fit(X_train, Y_train, epochs = 10,validation_data=(X_test, Y_test),callbacks=[learning_rate_reduction])

model1.save('skin_model2.h5')
print('Model Saved!')

plt.plot(hist.history['accuracy'], label='Train')
plt.plot(hist.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(hist.history['loss'], label='Train')
plt.plot(hist.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

import numpy as np
from keras.preprocessing import image
image_pred=image.load_img("C:/Multi Faceted Skin Disorder Classification/test/Monkeypox/M48_01_04.jpg",target_size=(224,224))# Add ur test image path
image_pred=image.img_to_array(image_pred)
image_pred=np.expand_dims(image_pred,axis=0)
rslt=model1.predict(image_pred)
if rslt[0][0]==1.0:
  print("Acne and Rosacea Photos")
if rslt[0][1]==1.0:
  print("Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions")
if rslt[0][2]==1.0:
  print("Atopic Dermatitis")
if rslt[0][3]==1.0:
  print("Eczema")
if rslt[0][4]==1.0:
  print("Nail Fungus and other Nail Disease")
if rslt[0][5]==1.0:
  print("Psoriasis pictures Lichen Planus and related diseases")
if rslt[0][6]==1.0:
  print("Monkeypox")
