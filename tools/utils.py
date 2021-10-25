from PIL import Image
import os, sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

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

    


