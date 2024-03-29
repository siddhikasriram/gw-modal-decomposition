from statistics import mean
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from tensorflow import keras
import numpy as np
from matplotlib.pyplot import figure, title
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from PIL import Image
import pickle

random.seed(7561)
#plt.rcParams.update({'font.size': 20})
  
def getdata(data_path):
  #print(len(os.listdir(data_path)))
  data=[]
  for img in os.listdir(data_path):
    img_arr = cv2.imread(os.path.join(data_path, img), 0)#[..., ::-1]
    img_arr = img_arr[..., np.newaxis]
    #resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    data.append(img_arr)
  
  return data

def labelling (fileName):
  
  yParam = []
  for image in fileName:
    #print(image)
    labels = []
    #print(int(image[9]))
    labels.append(int(image[9]))
    labels.append(int(image[11]))
    labels.append(int(image[13]))
    labels.append(int(image[15]))

    if image[22] == 'p':
      labels.append(64 + int(image[23:25]))
    else:
      labels.append(64 + int("-"+image[23:25]))

    if image[26] == 'p':
      labels.append(64 + int("+"+image[27:29]))
    else:
      labels.append(64 + int("-"+image[27:29]))
      
    labels.append(float(image[17:21])) # noise level
    yParam.append(labels)
  yParam = np.array([np.array(x) for x in yParam])
  return yParam

def show_mode_output(y_test, y_test_pred, fname, heading):

  #default size = 10
  params = {'axes.labelsize': 14, #Fontsize of the x and y labels
        'axes.titlesize': 15, #Fontsize of the axes title
        'figure.titlesize': 18, #Size of the figure title (.suptitle())
        'xtick.labelsize': 12, #Fontsize of the tick labels
        'ytick.labelsize': 12,
        'legend.fontsize': 12} #Fontsize for legends (plt.legend()

  plt.rcParams.update(params) 

  #print("n actual", y_test[:,1])
  #print("n pred", y_test_pred[:,1])

  #catalog = open(f'results_{fname}.txt', 'w')
  sample_size = len(y_test)
  actual = y_test.copy()
  predicted = y_test_pred.copy()
  actual = [tuple(x[:4]) for x in actual] 
  predicted = [tuple(x[:4]) for x in predicted]

  #print("actual combos of m and n ", actual)

  diff=[]
  for ind, act in enumerate(actual):
    pred = predicted[ind]
    z = [x - pred[i] for i, x in enumerate(act)] # z axis i.e, the colorbar
    diff.append(z)
  
  l = [diff[i][1] for i in range(0,len(diff))]
  #print("n difference:", l)
  actual_keys = [tuple (i) for i in actual] # (m, n) keys for dictionary
  #print("actual_keys - same as actual", actual_keys)
  mdiff1 = {}
  ndiff1 = {}
  mdiff2 = {}
  ndiff2 = {}

  for x in actual_keys: # to create list and append to it later
    mdiff1[x]=[]
    ndiff1[x]=[]
    mdiff2[x]=[]
    ndiff2[x]=[]
      
  for i, x in enumerate(actual):
    if x in mdiff1.keys():
      mdiff1[x].append(diff[i][0])
  
  for i, x in enumerate(actual):
    if x in ndiff1.keys():
      ndiff1[x].append(diff[i][1])

  for i, x in enumerate(actual):
    if x in mdiff2.keys():
      mdiff2[x].append(diff[i][2])

  for i, x in enumerate(actual):
    if x in ndiff2.keys():
      ndiff2[x].append(diff[i][3])

  #print("before mean m", mdiff)
  #print("before mean n", ndiff)
  # finding mean for each (m, n) combo for plotting colectively
  for k, v in mdiff1.items():
    mdiff1[k]=mean(v) 
  for k, v in ndiff1.items():
    ndiff1[k]=mean(v)
  for k, v in mdiff2.items():
    mdiff2[k]=mean(v) 
  for k, v in ndiff2.items():
    ndiff2[k]=mean(v)

  #print("mdiff after mean ", mdiff)
  #print(f"{heading} ndiff after mean", ndiff1)
  plt.figure(figsize=(26,40)) # verticle rectangle sheet for 3 x 2 subplots
  plt.suptitle(f'{heading} with {sample_size} samples')

  plt.subplot(4,3,1)
  plt.title('Set 1 TEM m - Mean deviation for all mode combinations')
  plt.xlabel('TEM m actual')
  plt.ylabel('TEM n actual')
  cm = plt.cm.get_cmap('brg')
  actual = [tuple(x[:4]) for x in actual]
  c_list_m = [] # list has to be passed for python version 3.7

  for i in range(0, len(actual)):
    z = actual[i]
    if z in mdiff1.keys():
      c_list_m.append(mdiff1[z])
  sc = plt.scatter(y_test[:, 0], y_test[:, 1], c=c_list_m, cmap = cm)
  #sc = plt.scatter(y_test[:, 0][i], y_test[:, 1][i], mdiff[z], cmap = cm) - indexing not allowed in v-3.7
  cbar1 = plt.colorbar(sc)
  cbar1.mappable.set_clim(vmin=-1,vmax=1)

  plt.subplot(4,3,2)
  plt.title('TEM m - Individual deviation from original value')
  plt.xlabel('m deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  m_deviation = y_test[:, 0]-y_test_pred[:, 0]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(m_deviation, edgecolor="white")
  # bins=len(set(m_deviation)), 
  for n, b in zip(counts, bins):
    plt.gca().text(b + 0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(4,3,4)
  plt.title('TEM n - Mean deviation for all mode combinations')
  plt.ylabel('TEM m actual')
  plt.xlabel('TEM n actual')
  cm = plt.cm.get_cmap('brg')
  actual = [tuple(x[:4]) for x in actual]
  c_list_n = []

  for i in range(0, len(actual)):
    z = actual[i]
    #print(z)
    if z in ndiff1.keys():
      c_list_n.append(ndiff1[z])
      
      print(y_test[:, 0][i], y_test[:, 1][i], ndiff1[z])
  sc = plt.scatter(y_test[:, 1], y_test[:, 0], c=c_list_n, cmap=cm)
  print("c_list_n", c_list_n)
  cbar2 = plt.colorbar(sc)
  cbar2.mappable.set_clim(vmin=-1,vmax=1)

  plt.subplot(4,3,5)
  plt.title('TEM n - Individual deviation from original value')
  plt.xlabel('n deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  n_deviation = y_test[:, 2]-y_test_pred[:, 2]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  #counts, bins, _ = plt.hist(n_deviation, bins=len(set(n_deviation)), edgecolor="white")
  counts, bins, _ = plt.hist(n_deviation, edgecolor = 'white')
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(4,3,7)
  plt.title('Set 2 TEM m - Mean deviation for all mode combinations')
  plt.xlabel('TEM m actual')
  plt.ylabel('TEM n actual')
  cm = plt.cm.get_cmap('brg')
  actual = [tuple(x[:4]) for x in actual]
  c_list_m = [] # list has to be passed for python version 3.7

  for i in range(0, len(actual)):
    z = actual[i]
    if z in mdiff2.keys():
      c_list_m.append(mdiff2[z])
  sc = plt.scatter(y_test[:, 2], y_test[:, 3], c=c_list_m, cmap = cm)
  #sc = plt.scatter(y_test[:, 0][i], y_test[:, 1][i], mdiff[z], cmap = cm) - indexing not allowed in v-3.7
  cbar1 = plt.colorbar(sc)
  cbar1.mappable.set_clim(vmin=-1,vmax=1)

  

  plt.subplot(4,3,8)
  plt.title('TEM m - Individual deviation from original value')
  plt.xlabel('m deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  m_deviation = y_test[:, 2]-y_test_pred[:, 2]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(m_deviation, edgecolor="white")
  # bins=len(set(m_deviation)), 
  for n, b in zip(counts, bins):
    plt.gca().text(b + 0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(4,3,10)
  plt.title('TEM n - Mean deviation for all mode combinations')
  plt.ylabel('TEM m actual')
  plt.xlabel('TEM n actual')
  cm = plt.cm.get_cmap('brg')
  actual = [tuple(x[:4]) for x in actual]
  c_list_n = []

  for i in range(0, len(actual)):
    z = actual[i]
    #print(z)
    if z in ndiff2.keys():
      c_list_n.append(ndiff2[z])
    
  sc = plt.scatter(y_test[:, 3], y_test[:, 2], c=c_list_n, cmap=cm)
  print("c_list_n", c_list_n)
  cbar2 = plt.colorbar(sc)
  cbar2.mappable.set_clim(vmin=-1,vmax=1)

  plt.subplot(4,3,11)
  plt.title('TEM n - Individual deviation from original value')
  plt.xlabel('n deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  n_deviation = y_test[:, 3]-y_test_pred[:, 3]
  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  #counts, bins, _ = plt.hist(n_deviation, bins=len(set(n_deviation)), edgecolor="white")
  counts, bins, _ = plt.hist(n_deviation, edgecolor = 'white')
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.05, n, int(n), rotation = 45)  # +0.1 to center text
  
  plt.subplot(4,3,3)
  plt.title('TEM m')
  plt.xlabel('m actual')
  plt.ylabel('m predicted')
  plt.axis([-1, 7, -1, 7])
  plt.scatter(y_test[:, 0], y_test_pred[:, 0], color = 'green', marker='*')

  plt.subplot(4,3,6)
  plt.title('TEM m')
  plt.xlabel('n actual')
  plt.ylabel('n predicted')
  plt.axis([-1, 7, -1, 7])
  plt.scatter(y_test[:, 1], y_test_pred[:, 1], color = 'green', marker='*')

  plt.subplot(4,3,9)
  plt.title('TEM m')
  plt.xlabel('m actual')
  plt.ylabel('m predicted')
  plt.axis([-1, 7, -1, 7])
  plt.scatter(y_test[:, 2], y_test_pred[:, 2], color = 'green', marker='*')

  plt.subplot(4,3,12)
  plt.title('TEM m')
  plt.xlabel('n actual')
  plt.ylabel('n predicted')
  plt.axis([-1, 7, -1, 7])
  plt.scatter(y_test[:, 3], y_test_pred[:, 3], color = 'green', marker='*')

  #for i, (a,p,n) in enumerate(zip(y_test, y_test_pred, noise)):
    #print(int(y_test[i, 0]), int(y_test_pred[i,0]), int(y_test[i, 1]), int(y_test_pred[i, 1]), int(y_test[i, 2]), int(y_test_pred[i, 2]), int(y_test[i, 3]), int(y_test_pred[i, 3]), noise[i])
    #catalog.write(" %d %f %d %f %d %f %d %f %.2f \n" %(y_test[i, 0], y_test_pred[i,0], y_test[i, 1], y_test_pred[i, 1], y_test[i, 2], y_test_pred[i, 2],y_test[i, 3], y_test_pred[i, 3], noise[i]))
    
  #catalog.close()
  plt.show()   
  plt.savefig(fname)


def show_two_modes_output(y_test, y_test_pred, fname, heading):

    #default size = 10
  params = {'axes.labelsize': 17, #Fontsize of the x and y labels
        'axes.titlesize': 18, #Fontsize of the axes title
        'figure.titlesize': 22, #Size of the figure title (.suptitle())
        'xtick.labelsize': 16, #Fontsize of the tick labels
        'ytick.labelsize': 16,
        'legend.fontsize': 15} #Fontsize for legends (plt.legend()

  plt.rcParams.update(params) 

  #print("n actual", y_test[:,1])
  #print("n pred", y_test_pred[:,1])

  #catalog = open(f'results_{fname}.txt', 'w')
  sample_size = len(y_test)
  actual = y_test.copy()
  predicted = y_test_pred.copy()
  actual = [tuple(x[:4]) for x in actual] 
  predicted = [tuple(x[:4]) for x in predicted]

  #print("actual combos of m and n ", actual)
  
  
  mdiff=[]
  ndiff=[]
  print("actual:", actual)
  print("predicted:", predicted)
  for ind, act in enumerate(actual):
    pred = predicted[ind]
    # print("act:", act)
    # print("pred:", pred)
    z = pred[0] - pred[2] # z axis i.e, the colorbar
    q = pred[1] - pred[3] 
    mdiff.append(z)
    ndiff.append(q)
  print(mdiff)
  print(ndiff)
  
  #l = [diff[i][1] for i in range(0,len(diff))]
  #print("n difference:", l)
  actual_m_keys = [tuple (i[0::2]) for i in actual] # (m, n) keys for dictionary
  actual_n_keys = [tuple (i[1::2]) for i in actual]
  #print("actual_keys - same as actual", actual_keys)
  dmdiff = {}
  dndiff = {}
  
  for x in actual_m_keys: # to create list and append to it later
    dmdiff[x]=[]
  for x in actual_n_keys:
    dndiff[x]=[]
  
      
  for i, x in enumerate(actual_m_keys):
    if x in dmdiff.keys():
      dmdiff[x].append(mdiff[i])
  
  for i, x in enumerate(actual_n_keys):
    if x in dndiff.keys():
      dndiff[x].append(ndiff[i])
 

  #print("before mean m", mdiff)
  #print("before mean n", ndiff)
  # finding mean for each (m, n) combo for plotting colectively
  for k, v in dmdiff.items():
    dmdiff[k]=mean(v) 
  for k, v in dndiff.items():
    dndiff[k]=mean(v)


  #print("mdiff after mean ", mdiff)
  #print(f"{heading} ndiff after mean", ndiff1)
  plt.figure(figsize=(15,15)) # verticle rectangle sheet for 3 x 2 subplots
  plt.suptitle(f'{heading} with {sample_size} samples')

  plt.subplot(2,2,1)
  plt.title('TEM m1 & m2 - Mean difference for m combinations')
  plt.xlabel('TEM m1 actual')
  plt.ylabel('TEM m2 actual') 
  cm = plt.cm.get_cmap('brg')
  c_list_m = [] # list has to be passed for python version 3.7

  for i in range(0, len(actual)):
    z = actual_m_keys[i]
    if z in dmdiff.keys():
      c_list_m.append(dmdiff[z])
  sc = plt.scatter(y_test[:, 0], y_test[:, 2], c=c_list_m, cmap = cm)
  #sc = plt.scatter(y_test[:, 0][i], y_test[:, 1][i], mdiff[z], cmap = cm) - indexing not allowed in v-3.7
  cbar1 = plt.colorbar(sc)
  cbar1.mappable.set_clim(vmin=-0.5,vmax=0.5)

  

  plt.subplot(2,2,2)
  plt.title('TEM m1 - m2 - Individual deviation from original value')
  plt.xlabel('m1 - m2 deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  n_deviation = y_test_pred[:, 0]-y_test_pred[:, 2]
  plt.xlim(xmin=-0.5, xmax = 0.5)
  plt.ylim([1,10**5])
  #counts, bins, _ = plt.hist(n_deviation, bins=len(set(n_deviation)), edgecolor="white")
  counts, bins, _ = plt.hist(n_deviation, edgecolor = 'white')
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.01, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(2,2,3)
  plt.title('TEM n1 & n2 - Mean difference for n combinations')
  plt.xlabel('TEM n1 actual')
  plt.ylabel('TEM n2 actual')
  cm = plt.cm.get_cmap('brg')
  actual = [tuple(x[:4]) for x in actual]
  c_list_n = []

  for i in range(0, len(actual)):
    z = actual_n_keys[i]
    #print(z)
    if z in dndiff.keys():
      c_list_n.append(dndiff[z])
      
      #print(y_test[:, 0][i], y_test[:, 1][i], ndiff1[z])
  sc = plt.scatter(y_test[:, 1], y_test[:, 3], c=c_list_n, cmap=cm)
  print("c_list_n", c_list_n)
  cbar2 = plt.colorbar(sc)
  cbar2.mappable.set_clim(vmin=-0.5,vmax=0.5)
  
  
  plt.subplot(2,2,4)
  plt.title('TEM n1 - n2 - Individual deviation from original value')
  plt.xlabel('n1 - n2 deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  n_deviation = y_test_pred[:, 1]-y_test_pred[:, 3]
  plt.xlim(xmin=-0.5, xmax = 0.5)
  plt.ylim([1,10**5])
  #counts, bins, _ = plt.hist(n_deviation, bins=len(set(n_deviation)), edgecolor="white")
  counts, bins, _ = plt.hist(n_deviation, edgecolor = 'white')
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.01, n, int(n), rotation = 45)  # +0.1 to center text
    
  plt.show()
  plt.savefig(fname)
  
def show_offset_output(y_test, y_test_pred, fname, heading):
  #default size = 10
  params = {'axes.labelsize': 14, #Fontsize of the x and y labels
        'axes.titlesize': 15, #Fontsize of the axes title
        'figure.titlesize': 18, #Size of the figure title (.suptitle())
        'xtick.labelsize': 12, #Fontsize of the tick labels
        'ytick.labelsize': 12,
        'legend.fontsize': 12} #Fontsize for legends (plt.legend())

  plt.rcParams.update(params) 
  
  #catalog = open(f'results_{fname}.txt', 'w')
  sample_size = len(y_test)
  actual = y_test.copy()
  predicted = y_test_pred.copy()
  actual = [tuple(x[4:6]) for x in actual] 
  #print("actual", actual)
  predicted = [tuple(x[4:]) for x in predicted]
  #print("pridicted", predicted)
  diff=[]
  for ind, act in enumerate(actual):
    pred = predicted[ind]
    z = [x - pred[i] for i, x in enumerate(act)] # z axis i.e, the colorbar
    diff.append(z)
  #print("diff", diff)
  #actual_keys = [tuple(x[]) for x in actual]  # (x,y) keys for dictionary
  #print("actual keys", actual_keys)
  xdiff = {}
  ydiff = {}
  for x in actual: # to create list and append to it later
    xdiff[x]=[]
    ydiff[x]=[]
      
  for i, x in enumerate(actual):
    if x in xdiff.keys():
      xdiff[x].append(diff[i][0])
      ydiff[x].append(diff[i][1])
  #print("xdiff", xdiff)   
  # finding mean for each (x,y) combo for plotting colectively
  for k, v in xdiff.items():
    xdiff[k]=mean(v) 
  for k, v in ydiff.items():
    ydiff[k]=mean(v)

  #print("ydiff mean", ydiff)
  plt.figure(figsize=(13,20)) # verticle rectangle sheet for 3 x 2 subplots
  plt.suptitle(f'{heading} with {sample_size} samples')

  plt.subplot(3,2,1)
  plt.title('Offset x - Mean deviation for all offset combinations')
  plt.xlabel('X actual')
  plt.ylabel('Y actual')
  cm = plt.cm.get_cmap('brg')
  c_list_x = [] # list has to be passed for python version 3.7

  for i, a in enumerate(y_test[:,4]):
    
    p = y_test_pred[i:, 4]
    z = actual[i]
    if z in xdiff.keys():
      c_list_x.append(xdiff[z])
  sc = plt.scatter(y_test[:, 4], y_test[:, 5], s=5, c=c_list_x, cmap = cm)
  #sc = plt.scatter(y_test[:, 0][i], y_test[:, 1][i], mdiff[z], cmap = cm) - indexing not allowed in v-3.7
  cbar1 = plt.colorbar(sc)
  cbar1.mappable.set_clim(vmin=-5,vmax=5)

  plt.subplot(3,2,2)
  plt.title('Offset x - Individual deviation from original value')
  plt.xlabel('X deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  x_deviation = y_test[:, 4]-y_test_pred[:, 4]
  plt.xlim(xmin=-15, xmax = 15)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(x_deviation, edgecolor="white")
  # bins=len(set(x_deviation)), 
  for n, b in zip(counts, bins):
    plt.gca().text(b + 0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(3,2,3)
  plt.title('Offset y - Mean deviation for all offset combinations')
  plt.xlabel('Y actual')
  plt.ylabel('X actual')
  cm = plt.cm.get_cmap('brg')
  c_list_y = []
  for i, a in enumerate(y_test[:,3]):
    p = y_test_pred[i:, 5]
    z = actual[i]
    if z in ydiff.keys():
      c_list_y.append(ydiff[z])
  sc = plt.scatter(y_test[:, 5], y_test[:, 4], s=5, c=c_list_y, cmap=cm)
  cbar2 = plt.colorbar(sc)
  cbar2.mappable.set_clim(vmin=-5,vmax=5)

  plt.subplot(3,2,4)
  plt.title('Offset y - Individual deviation from original value')
  plt.xlabel('Y deviation')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  y_deviation = y_test[:, 5]-y_test_pred[:, 5]
  plt.xlim(xmin=-15, xmax = 15)
  plt.ylim([1,10**5])
  counts, bins, _ = plt.hist(y_deviation, edgecolor = 'white')
  # bins=len(set(y_deviation)), 
  for n, b in zip(counts, bins):
    plt.gca().text(b+0.05, n, int(n), rotation = 45)  # +0.1 to center text

  plt.subplot(3,2,5)
  plt.title('X Offset')
  plt.xlabel('x off actual')
  plt.ylabel('x off predicted')
  plt.axis([30, 100, 30, 100])
  plt.scatter(y_test[:, 4], y_test_pred[:, 4], color='green', marker='*')

  plt.subplot(3,2,6)
  plt.title('Y Offset')
  plt.xlabel('y off actual')
  plt.ylabel('y off predicted')
  plt.axis([30, 100, 30, 100])
  plt.scatter(y_test[:, 5], y_test_pred[:, 5], color='green', marker='*')

  #for i, (a,p,n) in enumerate(zip(y_test, y_test_pred, noise)):
    #print(int(y_test[i, 0]), int(y_test_pred[i,0]), int(y_test[i, 1]), int(y_test_pred[i, 1]), int(y_test[i, 2]), int(y_test_pred[i, 2]), int(y_test[i, 3]), int(y_test_pred[i, 3]), noise[i])
    #catalog.write(" %d %f %d %f %d %f %d %f %.2f \n" %(y_test[i, 0], y_test_pred[i,0], y_test[i, 1], y_test_pred[i, 1], y_test[i, 2], y_test_pred[i, 2],y_test[i, 3], y_test_pred[i, 3], noise[i]))
    
  #catalog.close()
  plt.show()   
  plt.savefig(fname)




def show_noise_plot(low_noise_act, low_noise_pred, med_noise_act, med_noise_pred, high_noise_act, high_noise_pred, fname):

  plt.figure(figsize=(10,20)) 

  plt.suptitle('Output according to noise ranges')
  plt.subplot(2,1,1)
  plt.title('TEM m - Individual deviation from original value')
  plt.xlabel('m deviation for different noises')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  low_m_deviation = low_noise_act[:, 0]-low_noise_pred[:, 0]
  med_m_deviation = med_noise_act[:, 0]-med_noise_pred[:,0]
  high_m_deviation = high_noise_act[:, 0]-high_noise_pred[:, 0]

  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])
  
  plt.hist([low_m_deviation,med_m_deviation, high_m_deviation],bins = len(set(high_m_deviation)), label=['0.05 - 0.3', '0.3 - 0.6', '0.6 - 0.9'])
  plt.legend(loc='upper right')
  
  plt.subplot(2,1,2)
  plt.title('TEM n - Individual deviation from original value')
  plt.xlabel('n deviation for different noises')
  plt.ylabel('No. of Samples')
  plt.yscale('log')
  low_n_deviation = low_noise_act[:, 1]-low_noise_pred[:, 1]
  med_n_deviation = med_noise_act[:, 1]-med_noise_pred[:,1]
  high_n_deviation = high_noise_act[:, 1]-high_noise_pred[:, 1]

  plt.xlim(xmin=-5, xmax = 5)
  plt.ylim([1,10**5])

  plt.hist([low_n_deviation,med_n_deviation, high_n_deviation], bins = len(set(high_n_deviation)), label=['0.05 - 0.3', '0.3 - 0.6', '0.6 - 0.9'])
  plt.legend(loc='upper right')

  plt.show()
  #plt.savefig(fname)

def show_noisy_imgs(data_path, fname):

  #default size = 10
  params = {'axes.labelsize': 14, #Fontsize of the x and y labels
        'axes.titlesize': 16, #Fontsize of the axes title
        'figure.titlesize': 20, #Size of the figure title (.suptitle())
        'xtick.labelsize': 12, #Fontsize of the tick labels
        'ytick.labelsize': 12,
        'legend.fontsize': 12} #Fontsize for legends (plt.legend()

  plt.rcParams.update(params) 

  low_noise_imgs =[]
  med_noise_imgs =[]
  high_noise_imgs=[]
  img_list=[]
  low_noise_titles =[]
  med_noise_titles=[]
  high_noise_titles=[]
  name_list=[]


  for im in os.listdir(data_path):
    #img_path = os.path.join(data_path, im)
    img_arr = Image.open(os.path.join(data_path, im))
    img_arr = np.asarray(img_arr)
    
    noise = float(im[13:17])

    if noise >= 0.05 and noise <= 0.3:
      
      low_noise_imgs.append(img_arr)
      low_noise_titles.append(im)
    elif noise > 0.3 and noise <=0.6:
      med_noise_imgs.append(img_arr)
      med_noise_titles.append(im)
    else:
      high_noise_imgs.append(img_arr)
      high_noise_titles.append(im)
  #print(low_noise_imgs)  
  img_list.append(low_noise_imgs)
  img_list.append(med_noise_imgs)
  img_list.append(high_noise_imgs)

  name_list.append(low_noise_titles)
  name_list.append(med_noise_titles)
  name_list.append(high_noise_titles)

  for _, (img, name) in enumerate(zip(img_list, name_list)):
    combined = list(zip(img, name))
    l=random.sample(combined,9)
    data=[]
    for d, n in l:
      img_arr = cv2.imread(os.path.join(data_path, n), 0)#[..., ::-1]
      img_arr = img_arr[..., np.newaxis]
      data.append(img_arr)
    
      img_size = 128
      x_test = np.array(data) / 255
      x_test.reshape(-1, img_size, img_size, 1)
  
    #Testing the model with the test data
    y_test_pred = model.predict(x_test)
    # Rounding off as modes are integers
    y_test_pred[:,0] = [round(x) for x in y_test_pred[:,0]]
    y_test_pred[:,1] = [round(x) for x in y_test_pred[:,1]]
    y_test_pred[:,2] = [round(x) for x in y_test_pred[:,2]]
    y_test_pred[:,3] = [round(x) for x in y_test_pred[:,3]]
    y_test_pred[:,4] = [round(x) for x in y_test_pred[:,4]]
    y_test_pred[:,5] = [round(x) for x in y_test_pred[:,5]]

    rows=3
    cols =3 
    img_count = 0
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))
    fig.suptitle(f'{fname}{_}') 

    for i in range(rows):
      for j in range(cols):        
        if img_count < len(l):
          axes[i, j].imshow(l[img_count][0])
          #print(l[img_count][1])
          axes[i,j].set_title(f'{l[img_count][1]}\n Predicted: {y_test_pred[img_count,0]}, {y_test_pred[img_count, 1]}, {y_test_pred[img_count, 2]}, {y_test_pred[img_count, 3]}')
          img_count+=1
    #plt.show()  
    plt.savefig(fname)

if __name__ == '__main__':

  
  #Loading the saved model
  #model = keras.models.load_model('C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/comb_new_model.h5')
  model = keras.models.load_model('/home/siddhika/gw-modal-decomposition/new_model_comb.h5')
  #print(model.summary())
  
  
  #Fetching the test data
  img_size = 128
  #pathName = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/dataset/normal'
  pathName = '/home/siddhika/gw-modal-decomposition/Output/c_test'
  testing = getdata(pathName)
  x_test = np.array(testing) / 255
  x_test.reshape(-1, img_size, img_size, 1)
  
  #Labelling the test data for plots
  fileName = os.listdir(pathName)
  y_test = labelling(fileName)

  #Testing the model with the test data
  y_test_pred = model.predict(x_test)
  
  '''
  y_test_pkl = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/y_test.pkl'
  open_file1 = open(y_test_pkl, "rb")
  y_test = pickle.load(open_file1)
  open_file1.close()
  #print(y_test)
 
  with open('comb_y_test.pkl', "wb") as f:
    pickle.dump(y_test, f)
    #print(y_test)
  
  #y_test_pred is rounded | y_test_pred1 is not rounded
  y_test_pkl = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/y_test_pred.pkl'
  open_file2 = open(y_test_pkl, "rb")
  y_test_pred = pickle.load(open_file2)
  open_file2.close()
  
  
  with open('comb_y_test_pred.pkl', "wb") as q:
    pickle.dump(y_test_pred, q)
  #print(y_test)

  #print("ytest", y_test)
  #print("ypred", y_test_pred)
  ''' 

  # Rounding off as modes are integers
  # y_test_pred[:,0] = [round(x, 2) for x in y_test_pred[:,0]]
  # y_test_pred[:,1] = [round(x, 2) for x in y_test_pred[:,1]]
  # y_test_pred[:,2] = [round(x, 2) for x in y_test_pred[:,2]]
  # y_test_pred[:,3] = [round(x, 2) for x in y_test_pred[:,3]]

  #evaluate = model.evaluate(x_test, y_test)

  #Plot the performance of the model
  noiselist=[]
  #print(y_test)
  for i, noise in enumerate(y_test[:, 4]):
    noiselist.append(noise)
  
  main_mode_op = '1.png'
  main_off_op = '2.png'
  main_two_mode_op = '025.png'
  main_heading = 'Performance of the model for the entire test dataset'
  #show_mode_output(y_test, y_test_pred, main_mode_op, main_heading)
  #show_offset_output(y_test, y_test_pred, main_off_op, main_heading)
  show_two_modes_output(y_test, y_test_pred, main_two_mode_op, main_heading)

  #Access perfromace based on noise levels - split to three ranges
  low_noise_act = []
  low_noise_pred =[]
  med_noise_act =[]
  med_noise_pred =[]
  high_noise_act =[]
  high_noise_pred =[]
  #for catalog
  lownoiselist=[]
  mednoiselist=[]
  highnoiselist=[]

  for ind, act in enumerate(y_test):
    pred = y_test_pred[ind]
    noise = y_test[ind,6]
    if noise >= 0.05 and noise <= 0.3:
      low_noise_act.append(y_test[ind,0:6])
      low_noise_pred.append(y_test_pred[ind, 0:6])
      lownoiselist.append(noise)
    elif noise > 0.3 and noise <=0.6:
      med_noise_act.append(y_test[ind,0:6])
      med_noise_pred.append(y_test_pred[ind, 0:6])
      mednoiselist.append(noise)
    else:
      high_noise_act.append(y_test[ind,0:6])
      high_noise_pred.append(y_test_pred[ind, 0:6])
      highnoiselist.append(noise)  

  low_noise_act = np.array([np.array(x) for x in low_noise_act])
  low_noise_pred = np.array([np.array(x) for x in low_noise_pred])
  low_noise_heading = 'Performance of the model when the noise is between 0.05 and 0.3'
  low_op_mode = '3.png'
  low_op_off = '12.png'
  low_two_mode_op = '024.png'
  #show_mode_output(low_noise_act, low_noise_pred, low_op_mode, low_noise_heading)
  #show_offset_output(low_noise_act, low_noise_pred, low_op_off, low_noise_heading)
  show_two_modes_output(y_test, y_test_pred, low_two_mode_op, low_noise_heading)
  

  med_noise_act = np.array([np.array(x) for x in med_noise_act])
  med_noise_pred = np.array([np.array(x) for x in med_noise_pred])
  med_noise_heading = 'Performance of the model when the noise is between 0.3 and 0.6'
  med_op_mode = '5.png'
  med_op_off = '13.png'
  med_two_mode_op = '023.png'
  #show_mode_output(med_noise_act, med_noise_pred, med_op_mode, med_noise_heading)
  #show_offset_output(med_noise_act, med_noise_pred, med_op_off, med_noise_heading)
  show_two_modes_output(y_test, y_test_pred, med_two_mode_op, med_noise_heading)
  

  high_noise_act = np.array([np.array(x) for x in high_noise_act])
  high_noise_pred = np.array([np.array(x) for x in high_noise_pred])
  high_noise_heading = 'Performance of the model when the noise is between 0.6 and 0.9'
  high_op_mode = '7.png'
  high_op_off = '14.png'
  high_two_mode_op = '022.png'
  #show_mode_output(high_noise_act, high_noise_pred, high_op_mode, high_noise_heading)
  #show_offset_output(high_noise_act, high_noise_pred, high_op_off, high_noise_heading)
  show_two_modes_output(y_test, y_test_pred, high_two_mode_op, high_noise_heading)
  
  #combined plot
  noise_op = '3noisyplot.png'
  #show_noise_plot(low_noise_act, low_noise_pred, med_noise_act, med_noise_pred, high_noise_act, high_noise_pred, noise_op)
  
  #what are those noisy imgs?
  savename = 'noise_'
  #show_noisy_imgs(pathName, savename)
  
  

