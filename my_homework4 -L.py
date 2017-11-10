#This is the sample code of discrere hopfield network

import numpy as np
import random
from PIL import Image

#convert matrix to a vector
def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)
    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1

#Create Weight matrix for a single image
def create_W(x):
    if len(x.shape) != 1:
        print ("The input is not vector")
        return
    else:
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w

#Read Image file and convert it to Numpy array
def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")  #图片读入并灰度化
    pilIN= pilIN.resize(size)#
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    #x = np.zeros(imgArray.shape,dtype=np.float64)
    #x[imgArray > threshold] = 1
    #x[x==0] = -1
    return imgArray

#Convert Numpy array to Image file like Jpeg
def array2img(data):
    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    return img

#Update
def update(w,y_vec,theta=0.5,time=100):
    for s in range(time):
        m = len(y_vec)
        i = random.randint(0,m-1)
        u = np.dot(w[i][:],y_vec) - theta
        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1
    return y_vec

#The following is training pipeline
#Initial setting
def hopfield(train_files, test_files,theta=0.5, time=1000, size=(100,100),threshold=60):

    #read image and convert it to Numpy array
    print ("Importing images and creating weight matrix....")
    x = readImg2array(file=train_files,size=size,threshold=threshold)
    x_vec = mat2vec(x)
    print (len(x_vec))
    w = create_W(x_vec)  #run time is too long
    print ("Weight matrix is done!!")

    #Import test data and update
    y = readImg2array(file=test_files,size=size,threshold=threshold)
    print ("Imported test data")
    y_vec = mat2vec(y)
    print ("Updating...")
    y_vec_after = update(w=w,y_vec=y_vec,theta=theta,time=time)
    y_vec_after = y_vec_after.reshape(y.shape)
    after_img = array2img(y_vec_after)
    after_img.save("lufei_after_test.jpg")

#Main
#First, you can create a list of input file path
train_pic='lufei.jpg'
test_pic='lufei_test.jpg'
#Hopfield network starts!
hopfield(train_files=train_pic, test_files=test_pic, theta=0.5,time=100000,size=(100,100),threshold=120)