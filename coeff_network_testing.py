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
from sklearn.preprocessing import OneHotEncoder
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

    '''

    if image[22] == 'p':
      labels.append(64 + int(image[23:25]))
    else:
      labels.append(64 + int("-"+image[23:25]))

    if image[26] == 'p':
      labels.append(64 + int("+"+image[27:29]))
    else:
      labels.append(64 + int("-"+image[27:29]))
    '''

    labels.append(float(image[17:21])) # noise level

    yParam.append(labels)
  yParam = np.array([np.array(x) for x in yParam])
  return yParam

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
  for i, a in enumerate(y_test[:,5]):
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

def show_coeff_output(y_test, y_test_pred, fname, heading):

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
  actual = [tuple(x[0:4]) for x in actual] 
  predicted = [tuple(x[0:4]) for x in actual] 

  #print("actual combos of m and n ", actual)

  diff=[]
  # a_dev = []
  # b_dev = []
  # c_dev=[] 
  # d_dev=[]
  for ind, act in enumerate(actual):
    pred = predicted[ind]
    z = [x - pred[i] for i, x in enumerate(act)]
    # a = [x - pred[i] for i, x in enumerate(act)]
    # b = [x - pred[i] for i, x in enumerate(act)]
    # c = [x - pred[i] for i, x in enumerate(act)]
    # d = act[ind, 3]-pred[ind, 3] # z axis i.e, the colorbar
    diff.append(z)
    # a_dev.append(a)
    # b_dev.append(b)
    # c_dev.append(c)
    # d_dev.append(d)  

  plt.figure(figsize=(20,20)) # verticle rectangle sheet for 3 x 2 subplots
  plt.suptitle(f'{heading} with {sample_size} samples')

  # labels = ['a (0,0)', 'b(1,0)', 'c(0,1)', 'd(1,1)']
  # for i in actual:
  #   plt.legend( i, labels[i] )
  
  plt.xlabel('Deviation of co-efficients')
  plt.ylabel('Number of samples')
  plt.yscale('log')

  plt.hist([diff[0],diff[1], diff[2], diff[3]], label = ['a(0,0)', 'b(1,0)', 'c(0,1)', 'd(1,1)'])
  plt.legend(loc='upper right')
  
  plt.show()
  plt.savefig(fname)


if __name__ == '__main__':

  
  #Loading the saved model
  #model = keras.models.load_model('/home/sid/GWCNN/GW_input_data/gw-modal-decomposition/model_coeff.h5')
  model = keras.models.load_model('/home/siddhika/gw-modal-decomposition/model_coeff.h5')
  #print(model.summary())
  
  
  #Fetching the test data
  img_size = 128
  #pathName = '/home/sid/GWCNN/GW_input_data/gw-modal-decomposition/dataset/coeff'
  pathName = '/home/siddhika/gw-modal-decomposition/Output/coeff'
  testing = getdata(pathName)
  x_test = np.array(testing) / 255
  x_test.reshape(-1, img_size, img_size, 1)
  
  #Labelling the test data for plots
  fileName = os.listdir(pathName)
  y_test = labelling(fileName)

  #Testing the model with the test data
  y_test_pred = model.predict(x_test)

  y_test_pred[:,0] = [round(x, 2) for x in y_test_pred[:,0]]
  y_test_pred[:,1] = [round(x, 2) for x in y_test_pred[:,1]]
  y_test_pred[:,2] = [round(x, 2) for x in y_test_pred[:,2]]
  y_test_pred[:,3] = [round(x, 2) for x in y_test_pred[:,3]]
  
  '''
  y_test_pkl = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/comb_y_test.pkl'
  open_file1 = open(y_test_pkl, "rb")
  y_test = pickle.load(open_file1)
  open_file1.close()
  #print(y_test)
 
  with open('comb_y_test.pkl', "wb") as f:
    pickle.dump(y_test, f)
    #print(y_test)
  
  #y_test_pred is rounded | y_test_pred1 is not rounded
  y_test_pkl = 'C:/Users/siddh/Desktop/GW CNN/GW_input_data/1064 data/test/gw-modal-decomposition/comb_y_test_pred.pkl'
  open_file2 = open(y_test_pkl, "rb")
  y_test_pred = pickle.load(open_file2)
  open_file2.close()
  
  
  with open('comb_y_test_pred.pkl', "wb") as q:
    pickle.dump(y_test_pred, q)
  #print(y_test)

  
  y_test = y_test[0:20]
  y_test_pred = y_test_pred[0:20]

  '''

 
  print("ytest", y_test)
  print("ypred", y_test_pred)

  #evaluate = model.evaluate(x_test, y_test)
  
  #Plot the performance of the model
  noiselist=[]
  #print(y_test)
  for i, noise in enumerate(y_test[:, 4]):
    noiselist.append(noise)
  
  main_mode_op = 'c1.png'
  main_off_op = '2.png'
  main_heading = 'Co-efficient of the modes [ a(0,0) + b(1,0) + c(0,1) + d(1,1) ] for the entire test dataset'
  show_coeff_output(y_test, y_test_pred, main_mode_op, main_heading)
  #show_offset_output(y_test, y_test_pred, main_off_op, main_heading)

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
    noise = y_test[ind,4]
    if noise >= 0.05 and noise <= 0.3:
      low_noise_act.append(y_test[ind,0:4])
      low_noise_pred.append(y_test_pred[ind, 0:4])
      lownoiselist.append(noise)
    elif noise > 0.3 and noise <=0.6:
      med_noise_act.append(y_test[ind,0:4])
      med_noise_pred.append(y_test_pred[ind, 0:4])
      mednoiselist.append(noise)
    else:
      high_noise_act.append(y_test[ind,0:4])
      high_noise_pred.append(y_test_pred[ind, 0:4])
      highnoiselist.append(noise)  

  low_noise_act = np.array([np.array(x) for x in low_noise_act])
  low_noise_pred = np.array([np.array(x) for x in low_noise_pred])
  low_noise_heading = 'Performance of the model when the noise is between 0.05 and 0.3'
  low_op_mode = 'c3.png'
  low_op_off = '4.png'
  show_coeff_output(low_noise_act, low_noise_pred, low_op_mode, low_noise_heading)
  #show_offset_output(low_noise_act, low_noise_pred, low_op_off, low_noise_heading)

  med_noise_act = np.array([np.array(x) for x in med_noise_act])
  med_noise_pred = np.array([np.array(x) for x in med_noise_pred])
  med_noise_heading = 'Performance of the model when the noise is between 0.3 and 0.6'
  med_op_mode = 'c5.png'
  med_op_off = '6.png'
  show_coeff_output(med_noise_act, med_noise_pred, med_op_mode, med_noise_heading)
  #show_offset_output(med_noise_act, med_noise_pred, med_op_off, med_noise_heading)

  high_noise_act = np.array([np.array(x) for x in high_noise_act])
  high_noise_pred = np.array([np.array(x) for x in high_noise_pred])
  high_noise_heading = 'Performance of the model when the noise is between 0.6 and 0.9'
  high_op_mode = 'c7.png'
  high_op_off = '8.png'
  show_coeff_output(high_noise_act, high_noise_pred, high_op_mode, high_noise_heading)
  #show_offset_output(high_noise_act, high_noise_pred, high_op_off, high_noise_heading)

  #combined plot
  noise_op = '3noisyplot.png'
  #show_noise_plot(low_noise_act, low_noise_pred, med_noise_act, med_noise_pred, high_noise_act, high_noise_pred, noise_op)
  
  #what are those noisy imgs?
  savename = 'noise_'
  #show_noisy_imgs(pathName, savename)
  
  

