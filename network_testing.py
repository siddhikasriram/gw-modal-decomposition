from statistics import mean
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from tensorflow import keras
import numpy as np
from matplotlib.pyplot import figure, title
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Loading the saved model
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

#Fetching the test data
pathName = '/home/siddhika/gw-modal-decomposition/Output/test'
testing = getdata(pathName)
x_test = np.array(testing) / 255
x_test.reshape(-1, img_size, img_size, 1)

#Labelling the test data for plots
def labelling (fileName):
  yParam = []
  for image in fileName:
    labels = []
    labels.append(int(image[9])) # TEM m value
    labels.append(int(image[11])) # TEM n value

    if image[18] == 'p': # x-offset value
      labels.append(64 + int(image[19:21]))
    else:
      labels.append(64 + int("-"+image[19:21]))

    if image[22] == 'p': # y-offset value
      labels.append(64 + int("+"+image[23:25]))
    else:
      labels.append(64 + int("-"+image[23:25]))
    
    labels.append(float(image[13:17])) # noise level

    yParam.append(labels)
  yParam = np.array([np.array(x) for x in yParam])
  return yParam

fileName = os.listdir(pathName)
y_test = labelling(fileName)

#Testing the model with the test data
y_test_pred = model.predict(x_test)
#evaluate = model.evaluate(x_test, y_test)

#Plot the performance of the model
def show_output(y_test, y_test_pred, fname, heading):
  y_test_pred[:,0] = [round(x) for x in y_test_pred[:,0]]
  y_test_pred[:,1] = [round(x) for x in y_test_pred[:,1]]
  y_test_pred[:,2] = [round(x) for x in y_test_pred[:,2]]
  y_test_pred[:,3] = [round(x) for x in y_test_pred[:,3]]

  actual = y_test.copy()
  predicted = y_test_pred.copy()
  actual = [tuple(x[:2]) for x in actual] 
  predicted = [tuple(x[:2]) for x in predicted]

  diff=[]
  for ind, act in enumerate(actual):
    pred = predicted[ind]
    z = [x - pred[i] for i, x in enumerate(act)] # z axis i.e, the colorbar
    diff.append(z)

  actual_keys = [tuple (i) for i in actual] # (m, n) keys for dictionary

  mdiff = {}
  ndiff = {}
  for x in actual_keys: # to create list and append to it later
    mdiff[x]=[]
    ndiff[x]=[]
      
  for i, x in enumerate(actual):
    if x in mdiff.keys():
      mdiff[x].append(diff[i][0])
      ndiff[x].append(diff[i][1])

  # finding mean for each (m, n) combo for plotting colectively
  for k, v in mdiff.items():
    mdiff[k]=mean(v) 
  for k, v in ndiff.items():
    ndiff[k]=mean(v)

  plt.figure(figsize=(13,20)) # verticle rectangle sheet for 3 x 2 subplots
  plt.suptitle(heading)

  plt.subplot(3,2,1)
  plt.title('TEM m - Mean deviation for all mode combinations')
  plt.xlabel('TEM m actual')
  plt.ylabel('TEM n actual')
  cm = plt.cm.get_cmap('autumn')
  actual = [tuple(x[:2]) for x in actual]
  c_list_m = [] # list has to be passed for python version 3.7

  for i, a in enumerate(y_test[:,0]):
    print(i)
    p = y_test_pred[i:, 0]
    z = actual[i]
    if z in mdiff.keys():
      c_list_m.append(mdiff[z])
  sc = plt.scatter(y_test[:, 0], y_test[:, 1], c=c_list_m, cmap = cm)
  #sc = plt.scatter(y_test[:, 0][i], y_test[:, 1][i], mdiff[z], cmap = cm) - indexing not allowed in v-3.7
  cbar1 = plt.colorbar(sc)
  cbar1.mappable.set_clim(vmin=-1,vmax=1)

  plt.subplot(3,2,2)
  plt.title('TEM m - Individual deviation from original value')
  plt.xlabel('m deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  m_deviation = y_test[:, 0]-y_test_pred[:, 0]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(m_deviation, bins = 2)
  for n, b in zip(counts, bins):
    plt.gca().text(b + 0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(3,2,3)
  plt.title('TEM n - Mean deviation for all mode combinations')
  plt.xlabel('TEM m actual')
  plt.ylabel('TEM n actual')
  cm = plt.cm.get_cmap('autumn')
  c_list_n = []
  for i, a in enumerate(y_test[:,0]):
    p = y_test_pred[i:, 0]
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
  n_deviation = y_test[:, 1]-y_test_pred[:, 1]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(n_deviation, bins=len(set(n_deviation)), edgecolor="white")
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(3,2,5)
  plt.title('X Offset')
  plt.xlabel('x off actual')
  plt.ylabel('x off predicted')
  plt.axis([30, 100, 30, 100])
  plt.scatter(y_test[:, 2], y_test_pred[:, 2], color='green', marker='*')

  plt.subplot(3,2,6)
  plt.title('Y Offset')
  plt.xlabel('y off actual')
  plt.ylabel('y off predicted')
  plt.axis([30, 100, 30, 100])
  plt.scatter(y_test[:, 3], y_test_pred[:, 3], color='green', marker='*')

  plt.show()   
  plt.savefig(fname)

main_op = '3out.png'
main_heading = 'Performance of the model for the entire test dataset'
show_output(y_test, y_test_pred, main_op, main_heading)

#Access perfromace based on noise levels - split to three ranges
low_noise_act = []
low_noise_pred =[]
med_noise_act =[]
med_noise_pred =[]
high_noise_act =[]
high_noise_pred =[]

for ind, act in enumerate(y_test):
  pred = y_test_pred[ind]
  noise = y_test[ind,4]
  if noise >= 0.05 and noise <= 0.3:
    low_noise_act.append(y_test[ind,0:4])
    low_noise_pred.append(y_test_pred[ind, 0:4])
  elif noise > 0.3 and noise <=0.6:
    med_noise_act.append(y_test[ind,0:4])
    med_noise_pred.append(y_test_pred[ind, 0:4])
  else:
    high_noise_act.append(y_test[ind,0:4])
    high_noise_pred.append(y_test_pred[ind, 0:4])
      
low_noise_act = np.array([np.array(x) for x in low_noise_act])
low_noise_pred = np.array([np.array(x) for x in low_noise_pred])
low_noise_heading = 'Performance of the model when the noise is between 0.05 and 0.3'
low_op = '3lowout.png'
show_output(low_noise_act, low_noise_pred, low_op, low_noise_heading)

med_noise_act = np.array([np.array(x) for x in med_noise_act])
med_noise_pred = np.array([np.array(x) for x in med_noise_pred])
med_noise_heading = 'Performance of the model when the noise is between 0.3 and 0.6'
med_op = '3medout.png'
show_output(med_noise_act, med_noise_pred, med_op, med_noise_heading)

high_noise_act = np.array([np.array(x) for x in high_noise_act])
high_noise_pred = np.array([np.array(x) for x in high_noise_pred])
high_noise_heading = 'Performance of the model when the noise is between 0.6 and 0.9'
high_op = '3highout.png'
show_output(high_noise_act, high_noise_pred, high_op, high_noise_heading)

