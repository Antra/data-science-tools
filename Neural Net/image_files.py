import numpy as np
from PIL import Image
from pathlib import Path


# load the image and convert into numpy array
img = Image.open('images/random_sample1.jpeg')
numpy_data = np.asarray(img)

print(numpy_data.shape)  # (256,256,3) - height, width, channel
print(numpy_data)

# convert numpy array back to an image
img = Image.fromarray(numpy_data)
img.save('images/temp.jpeg')


# load 1000 image data from npy files and save them individually for inspection
numpy_data = np.load('images/images.npy', allow_pickle=True)
imgs = [arr for arr in numpy_data]
for i, image in enumerate(imgs):
    img = Image.fromarray(image)
    img.save(f'images/temp{i}.jpeg')

# or load 1000 images from jpeg files and save as npy file
imgs = Path('images').glob('temp*.jpeg')
numpy_data = np.array([np.array(Image.open(file)) for file in imgs])
np.save('images/images.npy', numpy_data)

# turn the images into a 1d array (vector format)
X = numpy_data.reshape(numpy_data.shape[0], -1).T
