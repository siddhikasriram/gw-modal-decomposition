
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from tensorflow import keras
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

model = keras.models.load_model('/home/siddhika/gw-modal-decomposition/new_model.h5')
print(model.summary())

img_size = 128
def getdata(data_path):
  #print(len(os.listdir(data_path)))
  data=[]
  for img in os.listdir(data_path):
    img_arr = cv2.imread(os.path.join(data_path, img), 0)#[..., ::-1]
    img_arr = img_arr[..., np.newaxis]

    #resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    data.append(img_arr)
  
  return data

pathName = '/home/siddhika/gw-modal-decomposition/Output/test'
testing = getdata(pathName)
x_test = np.array(testing) / 255
x_test.reshape(-1, img_size, img_size, 1)

def labelling (fileName):
  yParam = []
  for image in fileName:
    labels = []
    labels.append(int(image[9]))
    labels.append(int(image[11]))
    if image[18] == 'p':
      labels.append(int("+"+image[19:21]))
    
    else:
      labels.append(int("-"+image[19:21]))
      
    if image[22] == 'p':
      labels.append(int("+"+image[23:25]))
      
    else:
      labels.append(int("-"+image[23:25]))
    yParam.append(labels)
  yParam = np.array([np.array(x) for x in yParam])
  return yParam

fileName = os.listdir(pathName)
y_test = labelling(fileName)

yTest = model.predict(x_test)
evaluate = model.evaluate(x_test, y_test)
print(evaluate)
print("Actual values\n", y_test)
print("Predicted values\n", yTest)

yTest[:,0] = [round(x) for x in yTest[:,0]]
yTest[:,1] = [round(x) for x in yTest[:,1]]
yTest[:,2] = [round(x) for x in yTest[:,2]]
yTest[:,3] = [round(x) for x in yTest[:,3]]
print(yTest.shape)
print(yTest[:,0])

fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_title('m Modes')
ax.set_xlabel('TEM m actual')
ax.set_ylabel('TEM n actual')
ax.set_zlabel('m actual - m predicted')
ax.scatter(y_test[0], y_test[1], y_test[0]-yTest[0],  color='red', marker='*')

ax1 = fig.add_subplot(2, 2, 2, projection='3d')
ax1.set_title('n Modes')
ax1.set_xlabel('TEM m actual')
ax1.set_ylabel('TEM n actual')
ax1.set_zlabel('n actual - n predicted')
ax1.scatter(y_test[0], y_test[1], y_test[1]-yTest[1],  color='red', marker='*')

plt.subplot(2,2,3)
plt.title('X Offset')
plt.xlabel('x off actual')
plt.ylabel('x off predicted')
plt.axis([0, 128, 0, 128])
plt.scatter(y_test[2], yTest[2], color='green', marker='*')

plt.subplot(2,2,4)
plt.title('Y Offset')
plt.xlabel('y off actual')
plt.ylabel('y off predicted')
plt.axis([0, 128, 0, 128])
plt.scatter(y_test[3], yTest[3], color='green', marker='*')

plt.show()   
plt.savefig("modeloutput_new.png")
