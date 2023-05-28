from os import path, listdir

from .base import *
import scipy.io

class_numbers = {'Cherry_(including_sour)___Powdery_mildew':0, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':1,
                 'Corn_(maize)___Common_rust_':2, 'Corn_(maize)___healthy':3, 'Corn_(maize)___Northern_Leaf_Blight':4,
                 'Grape___Black_rot':5, 'Grape___Esca_(Black_Measles)':6, 'Grape___healthy':7,
                 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':8, 'Orange___Haunglongbing_(Citrus_greening)':9,
                 'Peach___Bacterial_spot': 10, 'Peach___healthy':11, 'Pepper,_bell___Bacterial_spot':12,
                 'Pepper,_bell___healthy':13, 'Potato___Early_blight':14, 'Potato___healthy':15,
                 'Potato___Late_blight':16, 'Raspberry___healthy':17, 'Soybean___healthy':18,
                 'Squash___Powdery_mildew':19, 'Strawberry___healthy':20, 'Strawberry___Leaf_scorch':21,
                 'Tomato___Bacterial_spot':22, 'Tomato___Early_blight':23, 'Tomato___healthy':24,
                 'Tomato___Late_blight':25, 'Tomato___Leaf_Mold':26, 'Tomato___Septoria_leaf_spot':27,
                 'Tomato___Spider_mites Two-spotted_spider_mite':28, 'Tomato___Target_Spot':29,
                 'Tomato___Tomato_mosaic_virus':30, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':31,
                 'Apple___Apple_scab':32, 'Apple___Black_rot':33, 'Apple___Cedar_apple_rust':34,
                 'Apple___healthy':35, 'Blueberry___healthy':36, 'Cherry_(including_sour)___healthy':37
                 }

class Plant_Village(BaseDataset):
    def __init__(self, root, classes, transform = None, ind=None):
        BaseDataset.__init__(self, root, classes, transform)
        image_dir = path.join(root)
        self.im_paths, self.ys = [], []
        sub_folders = listdir(image_dir)
        index = 0
        for d in sub_folders:
            y = class_numbers[d]
            if (y in classes):
                fnames = [path.join(image_dir, d, file) for file in listdir(path.join(image_dir, d))]
                self.im_paths += fnames
                m = len(fnames)
                self.ys += ([y] * m)
                self.I += list(range(index, index + m))
                index += m
