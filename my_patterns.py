#! /usr/bin/som python

from randomdots import *
p = Patterns(categories = 3, items_per_category = [10, 8, 4, 1], include_prototypes = False)
p.Dendrograms()
#p.Save('my_patterns.pkl') #uncomment to save
#p.Load('my_patterns.pkl') #uncomment to load
