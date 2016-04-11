from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import glob, os

path = 'img/'
img_array = []
os.chdir(path)
for i, img_file in enumerate(glob.glob("*.jpg")):
  None
img_array = np.zeros((i+1, 10000))
for i, img_file in enumerate(glob.glob("*.jpg")):
    print(img_file)
    img = Image.open(img_file).convert('L')
   
    #img = Image.open(img_file).convert('LA')

    #plt.imshow(img)
    #plt.show()
    img = Image.open(img_file).convert('L')
    img = img.resize((100, 100), Image.ANTIALIAS)

    #img_array.append(img.getdata())
    img_array[i] = img.getdata()

    
#print type(img_array[0])
#print type(img_array)

pca = PCA(n_components=2)
pca.fit(img_array)
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_) 