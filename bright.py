
#%%
import cv2
import numpy as np
import skimage
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage import data, io
from matplotlib import pyplot as plt  # doctest: +SKIP

#%%
image_sk = io.imread('data/test1.jpg')
image = rgb2gray(image)
print(image.shape)


plt.figure()            # doctest: +SKIP
io.imshow(image)    # doctest: +SKIP
io.show()               # doctest: +SKIP

#%%
# detecting edges in a coin image
filt_real, filt_imag = gabor(image, frequency=0.6)
plt.figure()            # doctest: +SKIP
io.imshow(filt_real)    # doctest: +SKIP
io.show()               # doctest: +SKIP
# less sensitivity to finer details with the lower frequency kernel
filt_real, filt_imag = gabor(image, frequency=0.1)
plt.figure()            # doctest: +SKIP
io.imshow(filt_real)    # doctest: +SKIP
io.show()               # doctest: +SKIP

# %%
image_cv = cv2.imread('data/test1.jpg')
gray_img=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)

plt.figure()            # doctest: +SKIP
io.imshow(gray_img)    # doctest: +SKIP
io.show()               # doctest: +SKIP

#%%
hist=cv2.calcHist(gray_img,[0],None,[256],(0,256), accumulate=False)

plt.title("Image")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist)            # doctest: +SKIP
# %%
gray_img_eqhist=cv2.equalizeHist(gray_img)
hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])


plt.title("Image gray eqhist")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(gray_img_eqhist) 

plt.figure()            # doctest: +SKIP
io.imshow(gray_img)    # doctest: +SKIP
io.show()               # doctest: +SKIP

# %%
clahe=cv2.createCLAHE(clipLimit=40)
gray_img_clahe=clahe.apply(gray_img_eqhist)

plt.figure()            # doctest: +SKIP
io.imshow(gray_img)    # doctest: +SKIP
io.show()               # doctest: +SKIP

# %%
th=30
max_val=255
ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
cv2.putText(o1,"Thresh_Binary",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY_INV)
cv2.putText(o2,"Thresh_Binary_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
cv2.putText(o3,"Thresh_Tozero",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO_INV)
cv2.putText(o4,"Thresh_Tozero_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
cv2.putText(o5,"Thresh_trunc",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
ret ,o6=  cv2.threshold(gray_img_clahe, th, max_val,  cv2.THRESH_OTSU)
cv2.putText(o6,"Thresh_OSTU",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)

final=np.concatenate((o1,o2,o3),axis=1)
final1=np.concatenate((o4,o5,o6),axis=1)

plt.figure()            # doctest: +SKIP
io.imshow(final)    # doctest: +SKIP
io.show()               # doctest: +SKIP

plt.figure()            # doctest: +SKIP
io.imshow(final1)    # doctest: +SKIP
io.show()     
# %%
