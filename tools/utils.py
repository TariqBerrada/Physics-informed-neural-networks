from PIL import Image
import os, sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

# path = './embedding/data/embedding.gif'

# img_paths = os.listdir('./embedding/data/')
# img_paths = np.sort([os.path.join('./embedding/data', x) for x in img_paths if x.startswith('progress') and x.endswith('_0.jpg')])

# images = [Image.open(x) for x in img_paths]

# gif_image = Image.new('RGB', (224, 224))
    
# gif_image.save(path, save_all=True, append_images=images, optimize=False, duration=100, loop=0)

# for img in img_paths:
#     os.remove(img)

def make_gif(img_folder, gif_path, duration = 100, clear = True):
    img_paths = os.listdir(img_folder)
    img_paths = np.sort([os.path.join(img_folder, x) for x in img_paths if x.endswith('.jpg')])

    shape = plt.imread(img_paths[0]).shape[:2]
    
    gif_image = Image.new('RGB', (shape[1], shape[0]))

    images = [Image.open(x) for x in img_paths]

    gif_image.save(gif_path, save_all=True, append_images=images, optimize=False, duration=duration, loop=0)
    
    if clear :
        for img in img_paths:
            os.remove(img)

    


