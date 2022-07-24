# Notes
Notes from playing around with a mini neural net.
In a production setting, one would use a deep learning framework like [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), or [PyTorch](https://pytorch.org/) instead of building ones own neural network.

Off-hand, I'd use:
- `Keras` for rapid prototyping and problems with small datasets
- `TensorFlow` for large datasets, high performance and object detection
- `PyTorch` for flexibility, faster training and good debugging capabilities

## Images
Random images generated with [DALL-e](https://huggingface.co/spaces/dalle-mini/dalle-mini).  

Read images with Pillow and NumPy:
```
import numpy as np
from PIL import Image

img = Image.open('data/random_sample1.jpeg')
numpy_data = np.asarray(img)
```