from asyncio import AbstractChildWatcher
import numpy as np
import os
from LightPipes import *
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import random
from numpy.lib.index_tricks import index_exp
np.random.seed(7561)

# Number of images to be generated
Ntot=20

# Defining variables: Wavelength of 1064 Nanometer, 40*40 mm2 sqaure grid with 128x128 pixels
wavelength = 1064*nm
size = 40*mm
Npix = 128
w0=3*mm
LG=False

# Maximum number of modes to be generated 

max_coeff = 9

mode_m1=0
mode_n1=0
mode_m2=1
mode_n2=0
mode_m3=0
mode_n3=1
mode_m4=1
mode_n4=1

a = np.random.randint(1,max_coeff,Ntot) 
b = np.random.randint(1,max_coeff,Ntot) #a(0,0) + b (1, 0) + c(0,1) 
c = np.random.randint(1,max_coeff,Ntot)
d = np.random.randint(1,max_coeff,Ntot)



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
catalog.write("Index | a | b | c | d |sigma value | x offset | y offset ")
m_list = []
n_list = []

for num in range(Ntot):
   
   aa = a[num]
   bb = b[num]
   cc = c[num]
   dd = d[num]
   
   x_off=x_offset[num] 
   y_off=y_offset[num] 

   F1=GaussBeam(F0, w0, LG=LG, n=mode_n1, m=mode_m1)
   F2=GaussBeam(F0, w0, LG=LG, n=mode_n2, m=mode_m2)
   F3=GaussBeam(F0, w0, LG=LG, n=mode_n3, m=mode_m3)
   F4=GaussBeam(F0, w0, LG=LG, n=mode_n4, m=mode_m4)
   Iimg1=Intensity(F1,1) #Intensity is calculated and normalized to 255 (2 -> 255, 1 -> 1.0, 0 -> not normalized)
   Iimg1=Iimg1+aa
   Iimg2=Intensity(F2,1)
   Iimg2=Iimg2+bb
   Iimg3=Intensity(F3,1)
   Iimg3=Iimg3+cc
   Iimg4=Intensity(F4,1)
   Iimg4=Iimg4+dd
   Iimg5=Iimg1+Iimg2+Iimg3+Iimg4
  

   # Adding Noise
   gauss = np.random.normal(mean,sigma[num],Iimg5.shape)
   gauss_img = gauss.reshape(Iimg5.shape)
   noisyIimg = Iimg5 + gauss_img
   
   # Index creates unique IDs for each image 
   index = str(int(num +1)).zfill(5)
   noiseFile = f'{index}_HG_{aa}_{bb}_{cc}_{dd}'
   fname = f'{noiseFile}.png'
   plt.imsave(fname, noisyIimg, cmap='gray')

   # Adding offset 
   im = Image.open(fname)
   offset_im = ImageChops.offset(im, x_off, y_off)

   # Naming the images before saving
   sigma_value = format(sigma[num], '.2f')
   
   catalog.write("%s %d %d %d %d %s %d %d \n"%(index,aa , bb, cc, dd, sigma_value, x_off,y_off))

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
            
