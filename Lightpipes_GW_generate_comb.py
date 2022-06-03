import numpy as np
import os
from LightPipes import *
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import random
from numpy.lib.index_tricks import index_exp
np.random.seed(7561)

# Number of images to be generated
Ntot=10

# Defining variables: Wavelength of 1064 Nanometer, 40*40 mm2 sqaure grid with 128x128 pixels
wavelength = 1064*nm
size = 40*mm
Npix = 128
w0=3*mm
LG=False

# Maximum number of modes to be generated 
mode_max=6
mode_m1=np.random.randint(0,mode_max,Ntot)
mode_n1=np.random.randint(0,mode_max,Ntot)
mode_m2=np.random.randint(0,mode_max,Ntot)
mode_n2=np.random.randint(0,mode_max,Ntot)

# Noise distribution
mean = 0
sigma = np.random.uniform(0.05,0.9, Ntot)

# Offset 
x_offset = np.random.randint(-30,30, Ntot)
y_offset = np.random.randint(-30,30, Ntot)

#The Begin command generates a field with amplitude 1.0 and phase zero, a plane wave. 
#So, all the 128x128 elements of array contain the complex number: 1.0 + j0.0
F0=Begin(size,wavelength,Npix)

# Catalog is the record of all the images and it's parameters generated
catalog = open("sample_catalog.txt","w")
m_list = []
n_list = []

for num in range(Ntot):
   nn1=mode_n1[num]
   mm1=mode_m1[num]
   nn2=mode_n2[num]
   mm2=mode_m2[num]
   x_off=x_offset[num] 
   y_off=y_offset[num] 
   F1=GaussBeam(F0, w0, LG=LG, n=nn1, m=mm1)
   F2=GaussBeam(F0, w0, LG=LG, n=nn2, m=mm2)
   Iimg1=Intensity(F1,1) #Intensity is calculated and normalized to 255 (2 -> 255, 1 -> 1.0, 0 -> not normalized)
   Iimg2=Intensity(F2,1)
   Iimg3 = Iimg1+Iimg2

   # Adding Noise
   gauss = np.random.normal(mean,sigma[num],Iimg3.shape)
   gauss_img = gauss.reshape(Iimg3.shape)
   noisyIimg = Iimg3 + gauss_img
   
   # Index creates unique IDs for each image 
   index = str(int(num +1)).zfill(5)
   noiseFile = f'{index}_HG_{nn1}_{mm1}_{nn2}_{mm2}'
   fname = f'{noiseFile}.png'
   plt.imsave(fname, noisyIimg, cmap='gray')

   # Adding offset 
   im = Image.open(fname)
   offset_im = ImageChops.offset(im, x_off, y_off)

   # Naming the images before saving
   sigma_value = format(sigma[num], '.2f')
   catalog.write("%s %d %d %d %d %s %d %d \n"%(index,nn1,mm1,nn2,mm2,sigma_value,x_off,y_off))

   # n_list.append(nn)
   # m_list.append(mm)
   
   if x_off < 0 and y_off < 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{x_off}_n{y_off}.png'
   elif x_off >= 0 and y_off >= 0:
      x_off = str(x_off).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{x_off}_p{y_off}.png'
   elif x_off >= 0 and y_off < 0:
      x_off = str(x_off).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{x_off}_n{y_off}.png'
   elif x_off < 0 and y_off >= 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{x_off}_p{y_off}.png'

      
   os.rename(fname, filename)
   offset_im.save(filename)
'''
plt.figure(figsize=(10,20)) 
plt.suptitle('Distribution of dataset')
plt.subplot(2,1,1)

plt.xlabel('TEM m')
plt.ylabel('TEM n')
sc = plt.scatter(m_list, n_list, c=sigma, cmap=cm)
cbar = plt.colorbar(sc)
cbar.mappable.set_clim(vmin=0.05,vmax=0.9)
'''
catalog.close()
            
