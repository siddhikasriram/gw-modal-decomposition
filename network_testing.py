from statistics import mean
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

actual = y_test.copy()
predicted = yTest.copy()
actual = [tuple(x[:2]) for x in actual]
predicted = [tuple(x[:2]) for x in predicted]

diff=[]
for ind, act in enumerate(actual):
  pred = predicted[ind]
  z = [x - pred[i] for i, x in enumerate(act)]
  diff.append(z)

actual_keys = [tuple(i) for i in actual]

mdiff = {}
ndiff = {}
for x in actual_keys:
  mdiff[x]=[]
  ndiff[x]=[]
    
for i, x in enumerate(actual):
  if x in mdiff.keys():
    mdiff[x].append(diff[i][0])
    ndiff[x].append(diff[i][1])

for k, v in mdiff.items():
  mdiff[k]=mean(v)
for k, v in ndiff.items():
  ndiff[k]=mean(v)

fig = plt.figure(figsize=(13,20))

plt.subplot(3,2,1)
plt.title('TEM m - Mean deviation for all mode combinations')
plt.xlabel('TEM m actual')
plt.ylabel('TEM n actual')
cm = plt.cm.get_cmap('autumn')
actual = [tuple(x[:2]) for x in actual]
c_list_m = []

for i, a in enumerate(y_test[:,0]):
  print(i)
  p = yTest[i:, 0]
  z = actual[i]
  if z in mdiff.keys():
    c_list_m.append(mdiff[z])
sc = plt.scatter(y_test[:, 0], y_test[:, 1], c=c_list_m, cmap = cm)
cbar1 = plt.colorbar(sc)
cbar1.mappable.set_clim(vmin=-1,vmax=1)

plt.subplot(3,2,2)
plt.title('TEM m - Individual deviation from original value')
plt.xlabel('m deviation')
plt.ylabel('No. of Samples')
plt.yscale('log')
m_deviation = y_test[:, 0]-yTest[:, 0]
plt.xlim(xmin=-5, xmax = 5)
plt.ylim([1,10**5])
#plt.hist(m_deviation)
counts, bins, _ = plt.hist(m_deviation)

for n, b in zip(counts, bins):
  plt.gca().text(b + 0.05, n, int(n), rotation = 45)  # +0.1 to center text

plt.subplot(3,2,3)
plt.title('TEM n - Mean deviation for all mode combinations')
plt.xlabel('TEM m actual')
plt.ylabel('TEM n actual')
cm = plt.cm.get_cmap('autumn')
c_list_n = []
for i, a in enumerate(y_test[:,0]):
  p = yTest[i:, 0]
  z = actual[i]
  if z in ndiff.keys():
    c_list_n.append(ndiff[z])
sc = plt.scatter(y_test[:, 0], y_test[:, 1], c=c_list_n, cmap=cm)
cbar2 = plt.colorbar(sc)
cbar2.mappable.set_clim(vmin=-1,vmax=1)

plt.subplot(3,2,4)
plt.title('TEM n - Individual deviation from original value')
plt.xlabel('n deviation')
plt.ylabel('No. of Samples')
plt.yscale('log')
n_deviation = y_test[:, 1]-yTest[:, 1]
plt.xlim(xmin=-5, xmax = 5)
plt.ylim([1,10**5])
#plt.hist(n_deviation)
counts, bins, _ = plt.hist(n_deviation)

for n, b in zip(counts, bins):
  plt.gca().text(b+0.05, n, int(n), rotation = 45)  # +0.1 to center text

plt.subplot(3,2,5)
plt.title('X Offset')
plt.xlabel('x off actual')
plt.ylabel('x off predicted')
plt.axis([30, 100, 30, 100])
plt.scatter(y_test[:, 2], yTest[:, 2], color='green', marker='*')

plt.subplot(3,2,6)
plt.title('Y Offset')
plt.xlabel('y off actual')
plt.ylabel('y off predicted')
plt.axis([30, 100, 30, 100])
plt.scatter(y_test[:, 3], yTest[:, 3], color='green', marker='*')

plt.show()   
plt.savefig("output13f.png")


