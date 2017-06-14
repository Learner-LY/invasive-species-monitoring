import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform
import matplotlib.image as mping
train_labels=pd.read_csv('/home/liyi/kag/train_labels.csv')

test_path='/home/liyi/kag/test/'
train_path='/home/liyi/kag/train/'
batch=40
high=577
width=433
train_image = np.empty(shape=(batch, width, high, 3))
#val_image = np.empty(shape=(295, width, high, 3))

def read_train_picture(num):
    train_label=np.array(train_labels.invasive.values[(0+batch*num):(batch+batch*num)])
    for i in range(batch):
        train_image[i]=transform.resize(mping.imread(train_path+str(1+i+batch*num)+'.jpg'),output_shape=(width,high,3))
        #if mping.imread(train_path+str(1+i+batch*num)+'.jpg').shape==(866,1154,3):
            #train_image[i]=mping.imread(train_path+str(1+i+batch*num)+'.jpg')
        #else:
            #train_image[i]=transform.resize(mping.imread(train_path+str(1+i+batch*num)+'.jpg'),output_shape=(866,1154,3))
    return train_image,train_label

#def read_val_picture():
#    val_label=np.array(train_labels.invasive.values[(2000):(2000+295)])
#    for i in range(295):
#        val_image[i]=transform.resize(mping.imread(train_path+str(2001+i)+'.jpg'),output_shape=(866,1154,3))
#    return val_image,val_label

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
model1 = Sequential()
model1.add(Convolution2D(64, 9,9,activation='relu', input_shape=(width, high,3)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Convolution2D(64, 9,9, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Convolution2D(32, 7,7, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Convolution2D(32, 7,7, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Convolution2D(16, 3,3,activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Convolution2D(16, 3,3,activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))


model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy', metrics=['accuracy'])
print(model1.summary())
#model1.load_weights('/home/liyi/kag/invasive_weight_6_14_217.h5')
for j in range(5):
    print 'epoch  '+str(j+1)+'  begin'+'-------------'
    for i in range(2000/batch):
        train_image,train_label=read_train_picture(i)
        a=model1.train_on_batch(train_image,train_label)
        print 'batch  '+str(i+1)+'  in  '+str(2000/batch+1)+':'+'loss:'+str(a[0])+';'+'acc:'+str(a[1])+';'
    print 'epoch  '+str(j+1)+'  end'+'-------------'
          
#model1.save_weights('/home/liyi/kag/invasive_weight_6_14_217.h5')

#model1.fit(train_image, train_label, nb_epoch=1, batch_size=10)
#val_image,val_label=read_val_picture()
#acc = model1.evaluate(val_image,val_label)
#print acc
