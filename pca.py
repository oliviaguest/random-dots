from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import glob, os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

path = 'img/'
img_array = []
os.chdir(path)
n = 300
for i, img_file in enumerate(glob.glob("*.jpg")):
  None
img_array = np.zeros((i+1, n*n))
label = range(i+1)
for i, img_file in enumerate(glob.glob("*.jpg")):
    print(img_file)
    img = Image.open(img_file).convert('L')
    label
    #img = Image.open(img_file).convert('LA')

    #plt.imshow(img)
    #plt.show()
    img = Image.open(img_file).convert('L')
    img = img.resize((n, n), Image.ANTIALIAS)

    #img_array.append(img.getdata())
    img_array[i] = img.getdata()
    label[i] = img_file[3:-4]
os.chdir('../')
   
#print type(img_array[0])
#print type(img_array)
X = img_array
pca = PCA(n_components=100)
X_r = pca.fit(X).transform(X)
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_) 


fig = plt.figure(figsize=(20, 18))
ax = fig.add_subplot(111)
ax.scatter(X_r[:, 0], X_r[:, 1], c='k', label='Objects', marker='.', s = 75, alpha = 0.8 )
#ax.set_ylim([-1.3, 1.3])
#ax.set_xlim([-1.8, 1.8])
for i, txt in enumerate(label):
    #l = plt.Text(text=txt, fontproperties=fp, x = X_r[i, 0],y = X_r[i, 1], axes = ax, figure = fig)
    ax.annotate(txt, (X_r[i, 0], X_r[i, 1]), horizontalalignment='center', verticalalignment='top',size = 14)
#for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
plt.legend()
plt.title('Phoneme PCA')
plt.xlabel('First Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[0], decimals = 3))+')')
plt.ylabel('Second Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[1], decimals = 3))+')')
#plt.show()
#fig.savefig('pca.jpg', bbox_inches='tight')
#fig.savefig('pca.pdf', bbox_inches='tight')


# generate the linkage matrix
X = X_r
print '\n\nX = ', X
Z = sch.linkage(X, 'ward')
c, coph_dists = sch.cophenet(Z, pdist(X, 'jaccard'))
# c, coph_dists = sch.cophenet(Z, pdist(X))
# Cophenetic Correlation Coefficient of clustering.
# This compares (correlates) the actual pairwise distances of all your samples to those implied by the hierarchical clustering.
# The closer the value is to 1, the better the clustering preserves the original distances.
print label, type(label[0])
print c

# calculate full dendrogram
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Photos')
plt.ylabel('Distance')
sch.dendrogram(
    Z,
    leaf_rotation=90,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    labels = label,
)
#for l in ax.get_xticklabels():
    #l.set_fontproperties(fp)
ax.tick_params(labelsize=16)
plt.show()

#fig.savefig('dendrogram.jpg', bbox_inches='tight')
#fig.savefig('dendrogram.pdf', bbox_inches='tight')