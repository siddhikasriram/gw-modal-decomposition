
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from tensorflow import keras
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
      labels.append(64 + int(image[19:21]))

    else:
      labels.append(64 + int("-"+image[19:21]))

    if image[22] == 'p':
      labels.append(64 + int("+"+image[23:25]))

    else:
      labels.append(64 + int("-"+image[23:25]))
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

plt.subplot(3,3,1)
plt.title('m modes')
plt.xlabel('TEM m actual')
plt.ylabel('TEM n actual')
cm = plt.cm.get_cmap('Set1')
sc = plt.scatter(y_test[0:100, 0], y_test[0:100, 1], c=(y_test[0:100, 0]-yTest[0:100, 0]), cmap=cm)
plt.colorbar(sc)

plt.subplot2grid(shape=(3, 3), loc=(0,1), colspan=2)
plt.title('m actual - m predicted graph')
#plt.xlabel('TEM m actual')
plt.ylabel('m Deviation')
m_deviation = y_test[0:100, 0]-yTest[0:100, 0]
#plt.legend(m_deviation, loc = 'best')
plt.plot(m_deviation, 'o' )
for i, j in zip(np.arange(len(m_deviation)), m_deviation):
   plt.text(i, j+0.02, '({}, {})'.format(i, j))

plt.subplot(3,3,4)
plt.title('n modes')
plt.xlabel('TEM m actual')
plt.ylabel('TEM n actual')
cm = plt.cm.get_cmap('Set1')
sc = plt.scatter(y_test[0:100, 0], y_test[0:100, 1], c=(y_test[0:100, 1]-yTest[0:100, 1]), cmap=cm)
plt.colorbar(sc)

plt.subplot2grid(shape=(3, 3), loc=(1,1), colspan=2)
plt.title('n actual - n predicted graph')
#plt.xlabel('TEM n actual')
plt.ylabel('n Deviation')
n_deviation = y_test[0:100, 1]-yTest[0:100, 1]
plt.plot(n_deviation, 'o')
for i, j in zip(np.arange(len(n_deviation)), n_deviation):
   plt.text(i, j+0.03, '({}, {})'.format(i, j))

plt.subplot(3,3,7)
plt.title('X Offset')
plt.xlabel('x off actual')
plt.ylabel('x off predicted')
plt.axis([30, 100, 30, 100])
plt.scatter(y_test[0:100, 2], yTest[0:100, 2], color='green', marker='*')
for i, j in zip(y_test[0:100, 2], yTest[0:100, 2]):
   plt.text(i, j-0.03, '({}, {})'.format(i, j))

plt.subplot(3,3,8)
plt.title('Y Offset')
plt.xlabel('y off actual')
plt.ylabel('y off predicted')
plt.axis([30, 100, 30, 100])
plt.scatter(y_test[0:100, 3], yTest[0:100, 3], color='green', marker='*')
for i, j in zip(y_test[0:100, 3], yTest[0:100, 3]):
   plt.text(i, j-0.03, '({}, {})'.format(i, j))

plt.show()   
plt.savefig("output8.png")

