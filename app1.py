#coder: Sonia Laskar

#Capstone Project: Intelligent Sales Assistant in Non Contractual Sales Setting

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
import math
import hashlib
import glob
import os
from scipy import fft
from mtcnn.mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as img

idname = ''
masked = True
cclv = ''
ccohort = ''
offer = ''
gender = ''

IPath = 'test/SoniaLaskar_Mask_test.jpg'

st.header('Intelligent Sales Assisstant (Conceptualized & executed by Sonia Laskar)')

st.text('Description: This product called ISA, during Covid 19, enables to detect mask of customers (using CNN),detect customer gender using masked customer image, recognises Masked Customers (Regression Task), predict CLV of relatively new customers with less than three transactions old customers using existing customers (Classification Task), predict their cohorts (Clustering Task) and recommends sales promotional offers to encourage sales and maintain customer satisfaction while maintaining minimum interaction between customer and on ground sales agent on retail shop floor for safety during covid 19. This also is directed to increase footfalls during covid that providing offers through email/sms instantly only when customer visits retail shop.')

from PIL import Image
image = Image.open(IPath)
#st.image(image, caption='Current Customer Image as Input captured at sales outlet',use_column_width=True)

imgg1 = img.imread(IPath)

#----------------------------------------------------------------

# reading the image

testImage = cv2.imread(IPath)
  
# displaying the image

MEAN_IMG = []
FSHIFT_IMG = []
HIST_IMG = []
DCT_IMG = []

x2,y2,w2,h2 = 0,0,0,0

GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

originalImage = cv2.imread(IPath)#(fls)

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

H2 = grayImage.copy()
grayImage.resize(180,180)
X = grayImage.flatten()

img = grayImage.copy()

im=cv2.flip(H2,1,1)

rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
faces = face_cascade.detectMultiScale(rerect_size)

for f in faces:
    (x, y, w, h) = [v * rect_size for v in f]

    face_img = im[y:y+h, x:x+w]
    rerect_sized=cv2.resize(face_img,(150,150))
    normalized=rerect_sized/255.0

    label=0

    cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
    cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
    
    roi_face=face_img.copy()

    roi_face_half=roi_face[0:int(face_img.shape[0]/2),:]

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

resized_img = resize(roi_face_half, (128,64))

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True)
            
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
print(hog_image_rescaled.shape)

hog_image_rescaled_flat = hog_image_rescaled.flatten()

FV_LIST.extend(hog_image_rescaled_flat)

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

gender = citi[int(prediction)]

gender_code = int(prediction)

 
# reading the image
testImage = cv2.imread(IPath)
  
gender = citi[int(prediction)]
#---------------------------------------------------------------------------


#____________________________________________________________________________

#predict person/ customer

# reading the image
testImage = cv2.imread(IPath)

MEAN_IMG = []
FSHIFT_IMG = []
HIST_IMG = []
DCT_IMG = []

x2,y2,w2,h2 = 0,0,0,0

GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

originalImage = cv2.imread(IPath)#(fls)

grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

H2 = grayImage.copy()
grayImage.resize(180,180)
X = grayImage.flatten()

img = grayImage.copy()

im=cv2.flip(H2,1,1)

rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
faces = face_cascade.detectMultiScale(rerect_size)

for f in faces:
    (x, y, w, h) = [v * rect_size for v in f]

    face_img = im[y:y+h, x:x+w]
    rerect_sized=cv2.resize(face_img,(150,150))
    normalized=rerect_sized/255.0

    label=0

    cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
    cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
    
    roi_face=face_img.copy()

    roi_face_half=roi_face[0:int(face_img.shape[0]/2),:]

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

resized_img = resize(roi_face_half, (128,64))

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True)
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#print(hog_image_rescaled.shape)

hog_image_rescaled_flat = hog_image_rescaled.flatten()

FV_LIST.extend(hog_image_rescaled_flat)

FV_LIST.append(gender_code)

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
data = decompress_pickle('rfc_c_16_2.pbz2')

load_clf = data

prediction = load_clf.predict(df)
prediction_proba_3 = load_clf.predict_proba(df)
  
# reading the image
testImage = cv2.imread(IPath)#img.imread(IPath)
  
r_prob_2 = prediction_proba_3[0][prediction[0]]

list_indx_3 = []
for i in range(len(prediction_proba_3[0])):
    if prediction_proba_3[0][i]>0:
        list_indx_3.append(i)
        
mods_faces.append(cities[int(prediction)])
mods_faces_pct.append(round(r_prob_2*100,2))

idname = mods_faces[0]


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

ind = data[data['Name']=='Sonia Laskar'].index.values[0]

data.loc[ind,'cust'] = chft
data.loc[ind,'date'] = clv_all.loc[chft]['date']
data.loc[ind,'sales'] = clv_all.loc[chft]['sales']
data.loc[ind,'counts'] = clv_all.loc[chft]['counts']
data.loc[ind,'clv'] = clv_all.loc[chft]['clv']


#test case
#select randomly from transaction with Ti, with i>4

XXX = clv_all[clv_all.counts.gt(3)]

chft_rest = random.choices(XXX.index.values, k = 14)

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
    
    return "".join(s1)+'@'+'hotmail.com'
      

data['email'] = data['Name'].apply(func_email)

data['phone'] = 8378937538 # default phone number for all

#display previous transactions of any customer

clv_all_RFMT =  pd.read_csv('clv_all_RFMT.csv')
clv_all_RFMT.drop(columns = ['Unnamed: 0'],inplace = True)

cts = X['date'].groupby('cust').count()

#transaction count trs_cnt
X = X.reset_index()

trs_cnt = cts[int(data[data['Name'] == idname]['cust'])]

if trs_cnt==1 :
    f1 = X.groupby('cust').agg({'sales':'first'})

    f = f1.loc[int(data[data['Name'] == idname]['cust'])]
    print(f[0])

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    Xy['First_TValue'] = f[0]
    Xy['Second_TValue'] = 0
    Xy['Third_TValue'] = 0
    
    md = decompress_pickle('hclv_3.pbz2')

    cclv = md.predict(Xy)[0]
    
elif trs_cnt==2 :

    f1 = X.groupby('cust').agg({'sales':'first'})

    f2 = X.groupby('cust').nth(1)

    f = f1.loc[int(data[data['Name'] == idname]['cust'])]
    
    ff = f2.loc[int(data[data['Name'] == idname]['cust'][0])]
    print(f[0],ff['sales'])

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    Xy['First_TValue'] = f[0]
    Xy['Second_TValue'] = ff['sales']
    Xy['Third_TValue'] = 0

    md = decompress_pickle('hclv_3.pbz2')

    cclv = md.predict(Xy)[0]
    
elif trs_cnt==3 :

    f1 = X.groupby('cust').agg({'sales':'first'})

    f2 = X.groupby('cust').nth(1)

    f3 = X.groupby('cust').nth(2)

    f = f1.loc[int(data[data['Name'] == idname]['cust'])]

    ff = f2.loc[int(data[data['Name'] == idname]['cust'])]

    fff = f3.loc[int(data[data['Name'] == idname]['cust'])]

    Xy = clv_all_RFMT[clv_all_RFMT['cust'] == int(data[data['Name'] == idname]['cust'])][['recency','frequency','T','monetary_value']]

    Xy['First_TValue'] = f[0]
    Xy['Second_TValue'] = ff['sales']
    Xy['Third_TValue'] = fff['sales']
    
    md = decompress_pickle('hclv_3.pbz2')
    
    cclv = md.predict(Xy)[0]
    
elif trs_cnt>=4 :
   
    cclv = round(data[data['Name']==idname]['clv'].values[0],2)
    
#___________________________________________________________________________

#predict cohort
    
mod_db_cl = pickle.load(open('mod_db_cl.pkl','rb'))

model = mod_db_cl
#mod_db_cl.predict() - no such inbuilt function in sklearn - so created a user defined function here

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
#___________________________________________________________________________

#Send through email/msg

em = data[data['Name']==idname]['email']
ph = data[data['Name']==idname]['phone']
#___________________________________________________________________________




#___________________________________________________________________________

#display previous transactions of any customer

def find_cust_prev_transactions_last_clv(idname):

        st.write("CUSTOMER SUCCESSFULLY MATCHED !")
        w = 180
        col1, mid, col2 = st.beta_columns([1+w,1,20+w])
        with col1:
            #st.image(imgg1, width=w, caption='Customer Entering Store')
            st.image(imgg1, width=w, caption='Customer Entering Store')
            
            st.success(idname)
            
            st.write('Surety Percentage:',round(r_prob_2*100,2),'%')
            st.write('Gender Predicted from Image:',gender)
            st.write('Mask Detected:',masked)
        with col2:
        
            st.write('Customer ID: ',int(data[data['Name']==idname]['cust']))
            st.write('Phone :', data[data['Name']==idname]['phone'].values[0] )
            st.write('Email :', data[data['Name']==idname]['email'].values[0] )
            
            trn_counts = np.unique(X[X.cust ==  int(data[data['Name']==idname]['cust']) ]['counts'].values)[0]
            st.write('Number of non zero transactions: ',trn_counts )
          
            st.text('Transaction Details:')
            st.dataframe( X[X.cust ==  int(data[data['Name']==idname]['cust']) ][['date','sales']] )
           
            if trn_counts<4 :
                st.write('\nPredicted Potential Future CLV:' , round(cclv,2))
                st.write('Predicted Potential Cohort: ',ccohort)
            else:
                st.write('\nCLV:' , round(cclv,2))
                st.write('Cohort: ',ccohort)
                
            st.write('Offer given: ',offer)
            st.info('PL. Note: Offer sent through email/ phone')
            st.write('--------------------'*4)
            
#--------------------------------------------------------------------------------

find_cust_prev_transactions_last_clv(idname)
st.balloons()
