import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

BLUR_WIDTH = 10
FILTER_SIZE = 9 #must be odd integer

s = input('Desired string: ')

img = mpimg.imread(input('Path to style image: '))

imgs = list()
for c in s:
	#TODO: append image produced by generator(c, img)
	imgs.append(img)

h, w, _ = imgs[0].shape
new_img = np.zeros((h, w*len(imgs), 3))

for i in range(len(imgs)):
	new_img[0:h, i*w:(i+1)*w] = imgs[i]

for i in range(1, len(imgs)):
	new_img[0:h, i*w-BLUR_WIDTH:i*w+BLUR_WIDTH] = cv2.GaussianBlur(
		new_img[0:h, i*w-BLUR_WIDTH:i*w+BLUR_WIDTH], (FILTER_SIZE, FILTER_SIZE), cv2.BORDER_DEFAULT)

plt.imshow(new_img)
plt.show()