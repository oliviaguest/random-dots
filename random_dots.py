#! /usr/bin/som python
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import random as r
import string
import pickle

import scipy.ndimage
import scipy.misc
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

import hashlib



def Random(max_value, min_value = 0):
  "Random integer from min_value to max_value"
  return int(r.randint(min_value, max_value))

class Patterns:
  """A class that creates, stores, and loads random dot patterns.

  Keyword arguments:
  categories             -- default value = 20
  levels_of_distortion   -- levels within a category that increasingly distort items away from the prorotype; default value = 3
  items_per_level        -- how many category members per level; default value = 3
  pattern_width          -- width of an individual category item (same for all items); default value = 10
  pattern_height         -- ditto; default value = 20
  max_units_set          -- how many features/units set to 'on'/1; default value = 10
  feature_overlap        -- whether prototypes may have common features with each other; default value = False
  category_overlap       -- whether catgeory members may be closer to the prototype of another category than to their own; default value = False
  compression_overlap    -- whether compressed items may be closer to the compressed prototype of another category than to their own; default value = False
  compression_width      -- the width of the compressed version of the binary patterns; default value = 5
  compression_height     -- ditto; default value = 5
  distortion -- parameter that controls the amount of compression/blurring that occurs; default value = 0.07
  pickle_file            -- the file to save/load patterns from; default value is function of categories, e.g., if categories = 10, pickle_file = '10_categories.pkl'
  """
  def __init__(self, categories = 10, levels_of_distortion = 3, items_per_level = 3,
               pattern_width = 30, pattern_height = 50, max_units_set = 20,
               feature_overlap = False, category_overlap = False, compression_overlap = False,
               compression_width = 20, compression_height = 25, distortion = 0.07,
               pickle_file = None):

    self.categories = categories
    self.levels_of_distortion = levels_of_distortion
    self.items_per_level = items_per_level
    self.pattern_num = self.categories * (1+ self.levels_of_distortion * self.items_per_level)
    self.pattern_width = pattern_width
    self.pattern_height = pattern_height
    self.max_units_set = max_units_set
    
    self.feature_overlap = feature_overlap
    self.category_overlap = category_overlap
    self.compression_overlap = compression_overlap
    
    self.patterns = np.empty((self.pattern_num, self.pattern_width, self.pattern_height))
    self.items = np.zeros_like(self.patterns)
    self.prototypes = np.zeros_like(self.patterns)

    self.compression_width = compression_width #int(self.pattern_width*0.5)
    self.compression_height = compression_height #int(self.pattern_height*0.5)
    self.compressed_representation = np.empty((self.pattern_num, self.compression_width, self.compression_height))
    self.compressed_resolution = np.ones(self.pattern_num)
    self.compressed_resolution.fill(distortion)
    
    if pickle_file == None:
      self.pickle_file = str(self.categories)+'_categories.pkl'
    else:
      self.pickle_file = pickle_file

  def calculate_compressed_representation(self):
    "Create the compression version of the patterns and return it."
    #for each pattern
    for i, p in enumerate(self.patterns):
      #calculate what the self.compressed_representation is
      self.compressed_representation[i] = scipy.misc.imresize(p, (self.compression_width, self.compression_height), interp='bicubic', mode=None)
      self.compressed_representation[i] = scipy.ndimage.filters.gaussian_filter(self.compressed_representation[i], (1 - self.compressed_resolution[i]))
    self.compressed_representation /= self.compressed_representation.max(axis = 0) #nornmalise
    return self.compressed_representation.reshape((self.pattern_num, self.compression_width*self.compression_height))

#To do:
#Create a function that just creates prototypes
#Create a function that uses prototypes to create items, different amount per prototype/category
#Output after both run is a single list with all patterns, as now


  def create_patterns(self):
    "Generate the patterns based on the pre-specificed properties."
    print 'create_patterns'
    # for readability I have split this into various loops; who cares about time/space complexity
    # this loop generates the prototypes
    coord = [] #keep track of coordinates for setting features to 1
    o = 0 #counter for patterns overall, used further down
    hash_list = []
    #for each category and so therefore for each prototype we are about to create
    #(remember it's one prorotype per category)
    for i in range(0, self.categories):
      #we just started on this prototype for this category
      #so there is no way we have set anything to 1 
      units_set = 0
      
      #while the units set are less than maximum we want to set
      #meaning that we will do the following untill all the units are set
      while (units_set < self.max_units_set):
        #while (p[i, x, y] == 1): # look for unit that is not set
          #get some random coordinates for the potentially 'on' feature
          x = Random(self.pattern_width-1)
          y = Random(self.pattern_height-1)
          #if we don't want overlap, uncomment the following three lines
          if not self.feature_overlap:
            while (x,y) in coord: # again look for unit that has not been set in previous patterns
              #print 'Coordinate ({0:3d}, {1:3d}) rejected because it causes feature overlap.'.format(x, y)
              print('Prototype for category {0:3d} rejected because it causes feature overlap.'.format(i))

              x = Random(self.pattern_width-1)
              y = Random(self.pattern_height-1)
          #after all this, we have discovered (hopefully) an appropriate x and y
          #so we use those random coordinates to set a feature to 1  
          self.prototypes[i, x, y] = 1
          
          #we just set a unit to on, so we want to know that for pattern i (x, y) is on
          #so in the near future when we generate a new set of coordinates
          #we can compare them to what we have generated previously
          coord.append((x,y))
          #and we are counting up how many we have set so far, so we know when to stop
          units_set = units_set + 1
          
      #so we are done with the ith prototype, we are now going to create the other members of the ith category
      #set the oth pattern to the prototype we just created above
      self.patterns[o, :, :] = self.prototypes[i, :, :]
      #print self.patterns[o, :, :]
      #we have now moved up from the oth pattern
      #so we are on the (o+1)th 
      o += 1
      hash_list.append(hashlib.sha1(self.prototypes[i, :, :]).hexdigest())

      #here we are initialising two distances,
      #which we use further down to calculate the dist between 
      #a member of a category and the category prototype 
      dist = np.zeros([self.categories])
      
      #for every pattern in this category
      for l in range(self.levels_of_distortion):
        for e in range(self.items_per_level):
            #calculate the value of distortion to send to generate_item
            distortion = l+1
            #send it a prototype and an amount of distortion
            item = self.generate_item(self.prototypes[i, :, :], distortion)
            item_hash = hashlib.sha1(item).hexdigest()

            while str(item_hash) in hash_list:
                item = self.generate_item(self.prototypes[i, :, :], distortion)
                item_hash = hashlib.sha1(item).hexdigest()
                print str(item_hash), hash_list, min(dist), dist[i]
                print str(item_hash) in hash_list

            #now we have generated an item we will calculate the distance of the current item
            #to all the prototypes per category we generated above
            #what we want is the current item to be closest to the prototype from which it was generated above
            #and not closer to any other prototype
            for c in range(self.categories):
                #calculate euclidan distance between item and all prototypes
                dist[c] = np.linalg.norm(item-self.prototypes[c,:,:])
            
            #initialise a counter for keeping track how many times we will re-create items
            counter = 0
            #if the minimum distance is not between an item and the ith prototype
            #it was generated from, do the following...
            if not self.category_overlap:
              while min(dist) != dist[i]:
                  item = self.generate_item(self.prototypes[i, :, :], distortion)
                  item_hash = hashlib.sha1(item).hexdigest()
                  #tell the user what happened
                  print('Item {0:3d} at level {1:2d} in category {2:2d} rejected because it causes category overlap; attempt: {3:3d}.'.format(o, l, i, counter+1))
                  #then generate a new item from the ith prototype, using the level of distortion we calculated above

                  while str(item_hash) in hash_list:
                      item = self.generate_item(self.prototypes[i, :, :], distortion)
                      item_hash = hashlib.sha1(item).hexdigest()
                      print str(item_hash), hash_list, min(dist), dist[i]
                      print str(item_hash) in hash_list
                      #increement this counter because things might get out of control, so we want to know
                      counter += 1
                  #we need to calculate new distances between our new item and the prototypes, so we can know which it's closest to
                  for c in range(self.categories):
                      dist[c] = np.linalg.norm(item-self.prototypes[c,:,:])
            print('Item {0:3d} at level {1:2d} in category {2:2d} saved!'.format(o, l, i))
            #since it has passed all the overlap requirements we can save the oth pattern        
            self.patterns[o, :, :] = item
            hash_list.append(hashlib.sha1(item).hexdigest())
            #and move on to (o+1)th
            o += 1

  def generate_item(self, prototype, distortion):
    "Create a category member given a prototype and a distortion level."
    item = np.zeros_like(prototype)
    distortion = distortion * 0.25
    indices = np.asarray(np.nonzero(prototype))
    noise = np.round(np.random.uniform(-distortion,distortion,indices.shape))
    noise = np.round(np.random.normal(loc=0.0, scale=distortion, size = indices.shape)).astype(int)

    indices += noise

    z = np.where(indices < 0)
    indices[z] = 0
    z = np.where(indices[0,:] >= self.pattern_width)
    indices[0, z] = self.pattern_width - 1
    z = np.where(indices[1,:] >= self.pattern_height)
    indices[1, z] = self.pattern_height - 1

    indices = tuple(indices)
    item[indices] = 1
    return item

  def dendrograms(self, subset_of_categories = None, metric = 'Euclidean'):
      # generate the linkage matrix
      X = self.patterns
      X = self.calculate_compressed_representation()
      #X = X.reshape((self.pattern_num,self.pattern_width*self.pattern_height))
      Z = sch.linkage(X, 'ward')
      c, coph_dists = sch.cophenet(Z, pdist(X, metric))
      # Cophenetic Correlation Coefficient of clustering.
      # This compares (correlates) the actual pairwise distances of all your samples to those implied by the hierarchical clustering.
      # The closer the value is to 1, the better the clustering preserves the original distances.
      print 'Cophenetic correlation coefficient of clustering (the closer to 1 the better):', c

      # calculate full dendrogram
      fig = plt.figure(figsize=(20, 10))
      ax = fig.add_subplot(111)
      plt.title('Hierarchical Clustering Dendrogram')
      plt.xlabel('Patterns')
      plt.ylabel(metric + ' Distance')
      sch.dendrogram(
          Z,
          leaf_rotation=90,  # rotates the x axis labels
          leaf_font_size=16.,  # font size for the x axis labels
      )
 
      ax.tick_params(labelsize=6)
      plt.show()
      

if __name__ == "__main__":

    p = Patterns()
    p.create_patterns()
    p.dendrograms()
