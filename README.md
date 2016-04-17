# visual-features
## Random dot patterns
To generate some random dots with the default settings:
```
python random_dots.py
```
To load some random dots from a file:
```
python random_dots.py my_file.pkl
```

To understand how the code works, I have included descriptive comments in [```random_dots.py```](https://github.com/oliviaguest/visual-features/blob/master/random_dots.py).
Also running the following will return the docstring for creating/initialising a random dot pattern:
```
import random_dots
help(random_dots)
```

## PCA on images
To run a PCA and produce a dendrogram of the PCA:
```
python img_pca.py
```
