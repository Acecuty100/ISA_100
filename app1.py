
#importing required libraries

import pandas as pd
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import pickle
import bz2
import _pickle as cPickle
import cv2
#import matplotlib.pyplot as plt
import math
import cv2
import hashlib
import glob
import os
from scipy import fft
from mtcnn.mtcnn import MTCNN
#import matplotlib.image as img

#from keras.models import load_model

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import hashlib
import glob
import os
from scipy import fft
from mtcnn.mtcnn import MTCNN
#import matplotlib.pyplot as plt
#import matplotlib.image as img

idname = ''
masked = True
cclv = ''
ccohort = ''
offer = ''
gender = ''

IPath = 'test/SoniaLaskar_Mask_test.jpg'

#st.header('')
st.subheader('Intelligent Sales Assisstant (Conceptualized & executed by Sonia Laskar)')
from PIL import Image
image = Image.open(IPath)

#st.image(image, caption='Current Customer Image',use_column_width=True)


def detect_mask(IPath):
    #model=load_model("my_model.h5")
    model=load_model("Mask_Detector.h5")
    results={0:'without mask',1:'mask'}
    GR_dict={0:(255,0,0),1:(0,0,255)}
    rect_size = 4
   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    im = cv2.imread(IPath)
    
    global imgg1
    imgg1 = img.imread(IPath)
    #plt.imshow(im)
    #plt.show()

    im=cv2.flip(im,1,1)

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = face_cascade.detectMultiScale(rerect_size)

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        print(x, y, w, h)

        face_img = im[y:y+h, x:x+w]


        #plt.imshow(face_img)
        #plt.show()

        rerect_sized=cv2.resize(face_img,(150,150))

        #plt.imshow(rerect_sized)
        #plt.show()

        normalized=rerect_sized/255.0

        #plt.imshow(normalized)
        #plt.show()

        reshaped=np.reshape(normalized,(1,150,150,3))

        reshaped = np.vstack([reshaped])

        result=model.predict(reshaped)

        print('Result Probabilities: ',result)

        label=np.argmax(result,axis=1)[0]

        print('Result label: ',label)
    

        print('Result:',results[label])
        
        if label == 1:
            global masked
            masked = True
        
        #st.write('Masked: ', masked)

        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        cv2.rectangle(imgg1,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(imgg1,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(imgg1, results[label], (x, y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        #plt.imshow(im)
        #plt.show()
        
        #st.image(imgg1,caption='Current Customer Image',use_column_width=True)
        
        #-------------------------------------------------------
        
#i = IPath
#detect_mask(i)
#print('---------------------------------------------\n')

#----------------------------------------------------------------

# reading the image
#IPath = 'test/SoniaLaskar_Mask_test.jpg'
testImage = cv2.imread(IPath) #img.imread(IPath)
  
# displaying the image

#plt.imshow(testImage)



MEAN_IMG = []
FSHIFT_IMG = []
HIST_IMG = []
DCT_IMG = []

x2,y2,w2,h2 = 0,0,0,0

GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

originalImage = cv2.imread(IPath)#(fls)
#plt.imshow(originalImage, cmap = 'gray')

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


H2 = grayImage.copy()
grayImage.resize(180,180)
X = grayImage.flatten()

img = grayImage.copy()

im=cv2.flip(H2,1,1)



rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
faces = face_cascade.detectMultiScale(rerect_size)



#print(rerect_size)

#print(faces)

for f in faces:
    (x, y, w, h) = [v * rect_size for v in f]

    #print(x,y,w,h)

    face_img = im[y:y+h, x:x+w]
    rerect_sized=cv2.resize(face_img,(150,150))
    normalized=rerect_sized/255.0

    label=0

    cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
    cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
    #cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    roi_face=face_img.copy()

    roi_face_half=roi_face[0:int(face_img.shape[0]/2),:]


    #folder_path = os.path.join(folder_path,file_name)

    image_content = roi_face_half.copy()


    

f = np.fft.fft2(roi_face_half)
fshift = np.fft.fftshift(f)

MEAN_IMG.append(roi_face_half.mean())

FSHIFT_IMG.append(fshift.mean())

HIST_IMG.append(np.histogram(roi_face_half)[0])

HI = pd.DataFrame(data = HIST_IMG)

gfg = fft.dct(roi_face_half)

DCT_IMG.append(gfg)

#16 Feature Vectors
FV_LIST = [np.mean(FSHIFT_IMG).real,
   np.mean(FSHIFT_IMG).imag,
   np.mean(MEAN_IMG),
   np.mean(HI).mean(),
   HI.iloc[:,0].mean(),
   HI.iloc[:,1].mean(),
   HI.iloc[:,2].mean(),
   HI.iloc[:,3].mean(),
   HI.iloc[:,4].mean(),
   HI.iloc[:,5].mean(),
   HI.iloc[:,6].mean(),
   HI.iloc[:,7].mean(),
   HI.iloc[:,8].mean(),
   HI.iloc[:,9].mean(),
   np.mean(DCT_IMG).real,
   np.mean(DCT_IMG).imag]

face_half_area = roi_face_half.shape[0]*roi_face_half.shape[1]
        
def shortest_distance(x1, y1, a, b, c):
    return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
            
detector = MTCNN()

OI = originalImage.copy()
#cv2.resize(OI,(180,180)).shape

# detect faces in the image
faces = detector.detect_faces(OI)#grayImage)
        
try:
    
    left_eye_distance = shortest_distance(faces[0]['keypoints']['left_eye'][0],
                                                 faces[0]['keypoints']['left_eye'][1],
                                                 1,0,-1*(faces[0]['box'][0]))
            
    right_eye_distance = shortest_distance(faces[0]['keypoints']['right_eye'][0],
                                                 faces[0]['keypoints']['right_eye'][1],
                                                 1,0,-1*(faces[0]['box'][0]+faces[0]['box'][2]))
        
    
    p1_x = faces[0]['box'][0]
    p1_y = faces[0]['box'][1]

    p2_x = faces[0]['box'][0]+faces[0]['box'][3]
    p2_y = faces[0]['box'][1]

    #print(p1_x,p1_y,p2_x,p2_y)

    #forehead centre

    fhead_x = 0.5*(p1_x+p2_x)
    fhead_y = 0.5*(p1_y+p2_y)
    #print(fhead_x,fhead_y)

    #print(head_x,head_y)
    head_x = 0.5*(faces[0]['keypoints']['left_eye'][0]+faces[0]['keypoints']['right_eye'][0])
    head_y = 0.5*(faces[0]['keypoints']['left_eye'][1]+faces[0]['keypoints']['right_eye'][1])
    #print(head_x,head_y)

    fhead_measure = np.linalg.norm(np.array([fhead_x,fhead_y])-np.array([head_x,head_y]))
    #print(fhead_measure)



    FV_LIST[15] = np.linalg.norm(np.array(faces[0]['keypoints']['left_eye'])-np.array(faces[0]['keypoints']['right_eye']))
    FV_LIST.append(fhead_measure)
    FV_LIST.append(left_eye_distance)
    FV_LIST.append(right_eye_distance)
    FV_LIST.append(np.std(OI[:,:,0]))
    FV_LIST.append(np.std(OI[:,:,1]))
    FV_LIST.append(np.std(OI[:,:,2]))
    
except:
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    pass

FV_LIST.append(face_half_area)

#resizing image - HOG
#print(len(FV_LIST))
#print(roi_face_half.shape)
resized_img = resize(roi_face_half, (128,64))
#print(resized_img.shape)
#plt.imshow(resized_img)
#plt.show()
#print(resized_img.shape)

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True)
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
print(hog_image_rescaled.shape)

hog_image_rescaled_flat = hog_image_rescaled.flatten()

FV_LIST.extend(hog_image_rescaled_flat)

#print(len(FV_LIST))
#print(FV_LIST)

#plt.imshow(hog_image_rescaled)
#plt.show()

df = pd.DataFrame(data = [FV_LIST])

#_______________________________________________________________________________


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

data = decompress_pickle('model_MF.pbz2')

load_clf = data


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

citi = ['Male','Female']

#print('Identified:',citi[int(prediction)])

gender = citi[int(prediction)]

gender_code = int(prediction)

 
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)

  
# displaying the image
#plt.imshow(testImage)

#plt.show()

#st.write('Identified:',citi[int(prediction)])
gender = citi[int(prediction)]
#---------------------------------------------------------------------------


#____________________________________________________________________________

#predict person/ customer


  
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)
  
# displaying the image
#plt.imshow(testImage)



MEAN_IMG = []
FSHIFT_IMG = []
HIST_IMG = []
DCT_IMG = []

x2,y2,w2,h2 = 0,0,0,0

GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

originalImage = cv2.imread(IPath)#(fls)
#plt.imshow(originalImage, cmap = 'gray')




#from skimage import io

#img = io.imread(file_path)



grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


H2 = grayImage.copy()
grayImage.resize(180,180)
X = grayImage.flatten()

img = grayImage.copy()

im=cv2.flip(H2,1,1)



rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
faces = face_cascade.detectMultiScale(rerect_size)



#print(rerect_size)

#print(faces)

for f in faces:
    (x, y, w, h) = [v * rect_size for v in f]

    #print(x,y,w,h)

    face_img = im[y:y+h, x:x+w]
    rerect_sized=cv2.resize(face_img,(150,150))
    normalized=rerect_sized/255.0

    label=0

    cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
    cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
    #cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    roi_face=face_img.copy()

    roi_face_half=roi_face[0:int(face_img.shape[0]/2),:]


    #folder_path = os.path.join(folder_path,file_name)

    image_content = roi_face_half.copy()


    

f = np.fft.fft2(roi_face_half)
fshift = np.fft.fftshift(f)

MEAN_IMG.append(roi_face_half.mean())

FSHIFT_IMG.append(fshift.mean())

HIST_IMG.append(np.histogram(roi_face_half)[0])

HI = pd.DataFrame(data = HIST_IMG)

gfg = fft.dct(roi_face_half)

DCT_IMG.append(gfg)

#16 Feature Vectors
FV_LIST = [np.mean(FSHIFT_IMG).real,
   np.mean(FSHIFT_IMG).imag,
   np.mean(MEAN_IMG),
   np.mean(HI).mean(),
   HI.iloc[:,0].mean(),
   HI.iloc[:,1].mean(),
   HI.iloc[:,2].mean(),
   HI.iloc[:,3].mean(),
   HI.iloc[:,4].mean(),
   HI.iloc[:,5].mean(),
   HI.iloc[:,6].mean(),
   HI.iloc[:,7].mean(),
   HI.iloc[:,8].mean(),
   HI.iloc[:,9].mean(),
   np.mean(DCT_IMG).real,
   np.mean(DCT_IMG).imag]

face_half_area = roi_face_half.shape[0]*roi_face_half.shape[1]
        
def shortest_distance(x1, y1, a, b, c):
    return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
            
detector = MTCNN()

OI = originalImage.copy()
#cv2.resize(OI,(180,180)).shape
# detect faces in the image
faces = detector.detect_faces(OI)#grayImage)
        
try:
    
    left_eye_distance = shortest_distance(faces[0]['keypoints']['left_eye'][0],
                                                 faces[0]['keypoints']['left_eye'][1],
                                                 1,0,-1*(faces[0]['box'][0]))
            
    right_eye_distance = shortest_distance(faces[0]['keypoints']['right_eye'][0],
                                                 faces[0]['keypoints']['right_eye'][1],
                                                 1,0,-1*(faces[0]['box'][0]+faces[0]['box'][2]))
        
    
    p1_x = faces[0]['box'][0]
    p1_y = faces[0]['box'][1]

    p2_x = faces[0]['box'][0]+faces[0]['box'][3]
    p2_y = faces[0]['box'][1]

    #print(p1_x,p1_y,p2_x,p2_y)

    #forehead centre

    fhead_x = 0.5*(p1_x+p2_x)
    fhead_y = 0.5*(p1_y+p2_y)
    #print(fhead_x,fhead_y)

    #print(head_x,head_y)
    head_x = 0.5*(faces[0]['keypoints']['left_eye'][0]+faces[0]['keypoints']['right_eye'][0])
    head_y = 0.5*(faces[0]['keypoints']['left_eye'][1]+faces[0]['keypoints']['right_eye'][1])
    #print(head_x,head_y)

    fhead_measure = np.linalg.norm(np.array([fhead_x,fhead_y])-np.array([head_x,head_y]))
    #print(fhead_measure)



    FV_LIST[15] = np.linalg.norm(np.array(faces[0]['keypoints']['left_eye'])-np.array(faces[0]['keypoints']['right_eye']))
    FV_LIST.append(fhead_measure)
    FV_LIST.append(left_eye_distance)
    FV_LIST.append(right_eye_distance)
    FV_LIST.append(np.std(OI[:,:,0]))
    FV_LIST.append(np.std(OI[:,:,1]))
    FV_LIST.append(np.std(OI[:,:,2]))
    
except:
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    FV_LIST.append(0)
    pass

FV_LIST.append(face_half_area)

#resizing image - HOG
#print(len(FV_LIST))
#print(roi_face_half.shape)
resized_img = resize(roi_face_half, (128,64))
#print(resized_img.shape)
#plt.imshow(resized_img)
#plt.show()
#print(resized_img.shape)

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True)
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#print(hog_image_rescaled.shape)

hog_image_rescaled_flat = hog_image_rescaled.flatten()

FV_LIST.extend(hog_image_rescaled_flat)

FV_LIST.append(gender_code)

#print(len(FV_LIST))
#print(FV_LIST)

#plt.imshow(hog_image_rescaled)
#plt.show()

df = pd.DataFrame(data = [FV_LIST])
#-----------------------------------------------------------------

mods_faces = []
mods_faces_pct =[]

#_________________________________________________________________


#RFC

cities = ['Shah Rukh Khan','Sonia Laskar','Preity Zinta','Bobby Deol','Priyanka Chopra','Angelina Jolie','Anne Hathaway','Nicolas Cage','Amitabh Bacchan','Brad Pitt','Abhishek Bacchan','Ananya Pandey','Saif Ali Khan','Kareena Kapoor','Malaika Arora']



def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

#choose a different model here
#data = decompress_pickle('rfc_c_16_stacked_5_xg_stacked_FINAL.pbz2')
data = decompress_pickle('rfc_c_16_2.pbz2')

load_clf = data

prediction = load_clf.predict(df)
prediction_proba_3 = load_clf.predict_proba(df)

#print('Identified:',cities[int(prediction)])

#import matplotlib.pyplot as plt
#import matplotlib.image as img
  
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)
  
# displaying the image
#plt.imshow(testImage)

#plt.show()


r_prob_2 = prediction_proba_3[0][prediction[0]]
#print('Surety Percentage:',round(r_prob_2*100,2),'%')
#print(prediction[0])
#print(prediction_proba_3)

list_indx_3 = []
for i in range(len(prediction_proba_3[0])):
    if prediction_proba_3[0][i]>0:
        list_indx_3.append(i)
        
#print('\nProbabilties: ')
#for i in list_indx_3:
#    print(cities[i],prediction_proba_3[0][i])

mods_faces.append(cities[int(prediction)])
mods_faces_pct.append(round(r_prob_2*100,2))

#st.write('RFC')
#st.write('Identified:',cities[int(prediction)])
#st.write('Surety Percentage:',round(r_prob_2*100,2),'%')


#----------------------------------------------------------------

#XGBOOST

cities = ['Shah Rukh Khan','Sonia Laskar','Preity Zinta','Bobby Deol','Priyanka Chopra','Angelina Jolie','Anne Hathaway','Nicolas Cage','Amitabh Bacchan','Brad Pitt','Abhishek Bacchan','Ananya Pandey','Saif Ali Khan','Kareena Kapoor','Malaika Arora']


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

#choose a different model here
#data = decompress_pickle('rfc_c_16_stacked_5_xg_stacked_FINAL.pbz2')
data = decompress_pickle('rfc_c_16_3.pbz2')

load_clf = data

prediction = load_clf.predict(df)
prediction_proba_3 = load_clf.predict_proba(df)

#print('Identified:',cities[int(prediction)])

#import matplotlib.pyplot as plt
#import matplotlib.image as img
  
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)
  
# displaying the image
#plt.imshow(testImage)

#plt.show()


r_prob_2 = prediction_proba_3[0][prediction[0]]
#print('Surety Percentage:',round(r_prob_2*100,2),'%')
#print(prediction[0])
#print(prediction_proba_3)

list_indx_3 = []
for i in range(len(prediction_proba_3[0])):
    if prediction_proba_3[0][i]>0:
        list_indx_3.append(i)
        
#print('\nProbabilties: ')
#for i in list_indx_3:
#    print(cities[i],prediction_proba_3[0][i])

mods_faces.append(cities[int(prediction)])
mods_faces_pct.append(round(r_prob_2*100,2))

#st.write('XGB')
#st.write('Identified:',cities[int(prediction)])
#st.write('Surety Percentage:',round(r_prob_2*100,2),'%')

#-----------------------------------------------------------------


#rfc_c_16_stacked_5_xg_stacked_FINAL

cities = ['Shah Rukh Khan','Sonia Laskar','Preity Zinta','Bobby Deol','Priyanka Chopra','Angelina Jolie','Anne Hathaway','Nicolas Cage','Amitabh Bacchan','Brad Pitt','Abhishek Bacchan','Ananya Pandey','Saif Ali Khan','Kareena Kapoor','Malaika Arora']

#Stacking Run

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data
    
#choose a different model here
data = decompress_pickle('rfc_c_16_stacked_5_xg_stacked_FINAL.pbz2')
#data = decompress_pickle('rfc_c_16_3.pbz2')




load_clf = data

prediction = load_clf.predict(df)
prediction_proba_3 = load_clf.predict_proba(df)

#print('Identified:',cities[int(prediction)])

#import matplotlib.pyplot as plt
#import matplotlib.image as img
  
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)
  
# displaying the image
#plt.imshow(testImage)

#plt.show()


r_prob_2 = prediction_proba_3[0][prediction[0]]
#print('Surety Percentage:',round(r_prob_2*100,2),'%')
#print(prediction[0])
#print(prediction_proba_3)

list_indx_3 = []
for i in range(len(prediction_proba_3[0])):
    if prediction_proba_3[0][i]>0:
        list_indx_3.append(i)
        
#print('\nProbabilties: ')
#for i in list_indx_3:
#    print(cities[i],prediction_proba_3[0][i])

mods_faces.append(cities[int(prediction)])
mods_faces_pct.append(round(r_prob_2*100,2))

#st.write('Stacking')
#st.write('Identified:',cities[int(prediction)])
#st.write('Surety Percentage:',round(r_prob_2*100,2),'%')

#---------------------------------------------------------------------

tot_f = 0

if ( (mods_faces_pct[0]) <= 15 and (mods_faces_pct[0]) >= 10) and (mods_faces[0]!=mods_faces[1] ) and (mods_faces[0]!=mods_faces[2] ):
    tot_f = (5)*mods_faces_pct[0]

if (mods_faces[0] == mods_faces[1]) and (mods_faces[0]!=mods_faces[2] ):
    tot_f = (1.5)*mods_faces_pct[0] + mods_faces_pct[1]
    print(tot_f,mods_faces_pct[2])

if tot_f > mods_faces_pct[2]:
    #idname = cities[int(prediction)]
    idname = mods_faces[0]
else:
    idname = mods_faces[2]
    
#_______________________________________________________________________

#print('Name:' , idname)#cities[int(prediction)])
#print('Surety Percentage:',round(r_prob_2*100,2),'%')
#print('Mask Detected:',masked)

#st.write('Name:' )#cities[int(prediction)])
#st.success(idname)
#st.write('Surety Percentage:',round(r_prob_2*100,2),'%')
#st.write('Mask Detected:',masked)


#plt.imshow(imgg1)
#plt.show()

#---------------------------------------------------------------------

names = ['Sonia Laskar','Shah Rukh Khan','Preity Zinta','Bobby Deol','Priyanka Chopra','Angelina Jolie','Anne Hathaway','Nicolas Cage','Amitabh Bacchan','Brad Pitt','Abhishek Bacchan','Ananya Pandey','Saif Ali Khan','Kareena Kapoor','Malaika Arora']


#---------------------------------------------------------------------


data = pd.DataFrame(data = {'Name':names})
X = pd.read_csv('Transactions',index_col = 'cust')
#Dropping above .11# rows with 0 sales value
X = X[X['sales']>0]
counts = X.reset_index().groupby('cust').count()
X['counts'] = counts['sales']
#only one time transactions
X_1 = X[X['counts'] == 1]
X_1_cust_id = np.unique(X_1.index)

#only two time transactions
X_2 = X[X['counts'] == 2]

p2 = round(X_2.shape[0]*100/X.shape[0],2)

X_2_cust_id = np.unique(X_2.index)

#only three time transactions
X_3 = X[X['counts'] == 3]

p3 = round(X_3.shape[0]*100/X.shape[0],2)

X_3_cust_id = np.unique(X_3.index)

#more than four times transactions
X_4 = X[(X['counts'] >= 4)]

p4 = round(X_4.shape[0]*100/X.shape[0],2)

X_4_cust_id = np.unique(X_4.index.values)

clv_all = pd.read_csv('clv_all.csv')

clv_all.set_index('cust',inplace = True)


import random

#test case
#select randomly from transaction with Ti, with i<=n, where n = 1,2,3
X_1_2_3_cust_id = []
for i in [X_1_cust_id,X_2_cust_id,X_3_cust_id]:
    X_1_2_3_cust_id.extend(i)
chft = random.choice(X_1_2_3_cust_id)
#print(chft,'\n')

#print(clv_all.loc[chft])


ind = data[data['Name']=='Sonia Laskar'].index.values[0]

data.loc[ind,'cust'] = chft
data.loc[ind,'date'] = clv_all.loc[chft]['date']
data.loc[ind,'sales'] = clv_all.loc[chft]['sales']
data.loc[ind,'counts'] = clv_all.loc[chft]['counts']
data.loc[ind,'clv'] = clv_all.loc[chft]['clv']


#test case
#select randomly from transaction with Ti, with i>4


XXX = clv_all[clv_all.counts.gt(3)]

#XXX.set_index('cust',inplace = True)

chft_rest = random.choices(XXX.index.values, k = 14)
#print(chft_rest,'\n')

for i in chft_rest:
    print(clv_all.loc[i])

j = 0
for i in names[1:]:
    ct = chft_rest[j]
    ind = data[data['Name']==i].index.values[0]

    data.loc[ind,'cust'] = ct
    data.loc[ind,'date'] = clv_all.loc[ct]['date']
    data.loc[ind,'sales'] = clv_all.loc[ct]['sales']
    data.loc[ind,'counts'] = clv_all.loc[ct]['counts']
    data.loc[ind,'clv'] = clv_all.loc[ct]['clv']
    j = j + 1
    
data.rename(columns = {'sales':'last_sales'},inplace = True)

def func_email(s):
    s = s.split()
    s1 = []
    for i in s:
        s1.append(i.lower())
    
    #print("".join(s1)+'@'+'hotmail.com')
    return "".join(s1)+'@'+'hotmail.com'
      

data['email'] = data['Name'].apply(func_email)

data['phone'] = 8355937492

#display previous transactions of any customer

clv_all_RFMT =  pd.read_csv('clv_all_RFMT.csv')
clv_all_RFMT.drop(columns = ['Unnamed: 0'],inplace = True)

cts = X['date'].groupby('cust').count()

#transaction count trs_cnt
X = X.reset_index()

#st.write(data)

#print(idname)

trs_cnt = cts[int(data[data['Name'] == idname]['cust'])]

#print('Number of previous transactions: ',trs_cnt)

if trs_cnt==1 :
    f1 = X.groupby('cust').agg({'sales':'first'})

    f = f1.loc[int(data[data['Name'] == idname]['cust'])]
    print(f[0])

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    #print(clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']])

    #print(Xy)

    Xy['First_TValue'] = f[0]

    #md = pickle.load(open('hclv_1.pkl','rb'))
    
    md = decompress_pickle('hclv_1.pbz2')

    cclv = md.predict(Xy)[0]

    #print(md.predict(Xy)[0])
    
elif trs_cnt==2 :

    f1 = X.groupby('cust').agg({'sales':'first'})

    f2 = X.groupby('cust').nth(1)

    f = f1.loc[int(data[data['Name'] == idname]['cust'])]
    #print(f[0])

    ff = f2.loc[int(data[data['Name'] == idname]['cust'][0])]
    print(f[0],ff['sales'])

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    #print(clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']])

    #print(Xy)

    Xy['First_TValue'] = f[0]
    Xy['Second_TValue'] = ff['sales']

    #md = pickle.load(open('hclv_2.pkl','rb'))
    
    md = decompress_pickle('hclv_2.pbz2')

    cclv = md.predict(Xy)[0]
    
    #print(md.predict(Xy)[0])

elif trs_cnt==3 :

    f1 = X.groupby('cust').agg({'sales':'first'})

    f2 = X.groupby('cust').nth(1)

    f3 = X.groupby('cust').nth(2)



    f = f1.loc[int(data[data['Name'] == idname]['cust'])]
    #print(f[0])

    ff = f2.loc[int(data[data['Name'] == idname]['cust'])]

    fff = f3.loc[int(data[data['Name'] == idname]['cust'])]

    #print(f[0],ff['sales'],fff['sales'])

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    Xy['First_TValue'] = f[0]
    Xy['Second_TValue'] = ff['sales']
    Xy['Third_TValue'] = fff['sales']
    
    #print(clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']])

    #print(Xy)
    
    #md = pickle.load(open('hclv_3.pkl','rb'))
    
    md = decompress_pickle('hclv_3.pbz2')
    
    cclv = md.predict(Xy)[0]
    
elif trs_cnt>=4 :
    #print(trs_cnt)
    cclv = round(data[data['Name']==idname]['clv'].values[0],2)
    #print('CLV: ',round(data[data['Name']==idname]['clv'].values[0],2))
    
    
#___________________________________________________________________________

#predict cohort
    
mod_db_cl = pickle.load(open('mod_db_cl.pkl','rb'))

model = mod_db_cl
#mod_db_cl.predict() - no such inbuilt function - so created a user defined function here

X_hclv = cclv #sl_hclv #hclv used to predict cluster

diff = model.components_ - X_hclv  # NumPy broadcasting

dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

shortest_dist_idx = np.argmin(dist)

y_new = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

#print(y_new)

ccohort = y_new

#___________________________________________________________________________

#So decide recommended offer
offer = 'C'+str(y_new)

#print(offer)

#___________________________________________________________________________

#Send through email/msg

em = data[data['Name']==idname]['email']
ph = data[data['Name']==idname]['phone']
#print('Email:',em.values[0])
#print('Phone:',ph.values[0])

#___________________________________________________________________________


import smtplib
from email.mime.text import MIMEText

gmailaddress = 'sonnialaskar@gmail.com'#input("what is your gmail address? \n ")
gmailpassword = 'Sonia1Laskar'#input("what is the password for that email address? \n  ")
mailto = str(em.values[0])#'sonialaskar@hotmail.com'#input("what email address do you want to send your message to? \n ")

str_msg_str = 'Hi '+ idname + '. Welcome to our store. Avail the special offers ' + offer+' for you at our store.'

msg = MIMEText(str_msg_str)

msg['Subject'] = 'CD Store Promotional Offer mail'
msg['From'] = 'Sonia Laskars CD Store'#'sonnialaskar@gmail.com'
msg['To'] = str(em.values[0])#'sonialaskar@hotmail.com'

mailServer = smtplib.SMTP('smtp.gmail.com' , 587)
mailServer.starttls()
mailServer.login(gmailaddress , gmailpassword)
mailServer.sendmail(gmailaddress, mailto , msg.as_string())
print(" \n Sent!")
mailServer.quit()

#___________________________________________________________________________


#import requests
#url = "https://www.fast2sms.com/dev/bulk"
#payload = "sender_id=FSTSMS&message=test&language=english&route=p&numbers="+ph headers = { 'authorization': "xuYRwBcjEH5yfKGor2UNFPCJTQL39s0Sp1IMOz6nkblmdZt4DvIOERw5ie168GUPsbgQJ4KX0T2FDZz9", 'Content-Type': "application/x-www-form-urlencoded", 'Cache-Control': "no-cache", }
#response = requests.request("POST", url, data=payload, headers=headers) print(response.text)'''



#___________________________________________________________________________


#display previous transactions of any customer

def find_cust_prev_transactions_last_clv(idname):

        #st.write('--------------------'*4)
        #st.write('CUSTOMER DETAILS: ')
        #st.write('--------------------'*4)

        st.write("CUSTOMER SUCCESSFULLY MATCHED !")
        w = 180
        col1, mid, col2 = st.beta_columns([1+w,1,20+w])
        with col1:
            st.image(imgg1, width=w, caption='Customer Entering Store')
            st.success(idname)
                    
            #print('Name:' , cities[int(prediction)])
            st.write('Surety Percentage:',round(r_prob_2*100,2),'%')
            st.write('Gender Predicted from Image:',gender)
            st.write('Mask Detected:',masked)
        with col2:
            #st.write(idname)
            
                    #st.write('Customer Name: ')
        
                    

                    #st.image(imgg1)
                    #plt.axis('off')
                    #plt.show()
                    
                    
                    
            st.write('Customer ID: ',int(data[data['Name']==idname]['cust']))
            st.write('Phone :', data[data['Name']==idname]['phone'].values[0] )
            st.write('Email :', data[data['Name']==idname]['email'].values[0] )
            
            trn_counts = np.unique(X[X.cust ==  int(data[data['Name']==idname]['cust']) ]['counts'].values)[0]
            st.write('Number of non zero transactions: ',trn_counts )
            
            #st.write('--------------------'*4)
            #if st.checkbox('Check to see Previous Transaction Details'):
            st.text('Transaction Details:')
            st.dataframe( X[X.cust ==  int(data[data['Name']==idname]['cust']) ][['date','sales']] )
            #st.write('--------------------'*4)
            
            if trn_counts<4 :
                st.write('\nPredicted Potential Future CLV:' , round(cclv,2))#round(clv_all.loc[int(data[data['Name']==idname]['cust']) ]['clv'] , 2))
                st.write('Predicted Potential Cohort: ',ccohort)
            else:
                st.write('\nCLV:' , round(cclv,2))#round(clv_all.loc[int(data[data['Name']==idname]['cust']) ]['clv'] , 2))
            
                st.write('Cohort: ',ccohort)
                
            st.write('Offer given: ',offer)
            st.info('PL. Note: Offer sent through email/ phone')
            st.write('--------------------'*4)
            

    
        #with st.beta_container():
        #    st.write("Customer Matched")
        #    st.image(imgg1, caption='Current Customer Image',use_column_width=True)
        
        #st.dataframe(data)
        
       
#--------------------------------------------------------------------------------

find_cust_prev_transactions_last_clv(idname)
st.balloons()
