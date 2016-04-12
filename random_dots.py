#! /usr/bin/som python

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random as r
import string
import pickle

import scipy.ndimage
import scipy.misc

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
  compression_width      -- the width of the compressed version of the binary patterns; default value = 5
  compression_height     -- ditto; default value = 5
  compression_resolution -- parameter that controls the amount of compression/blurring that occurs; default value = 0.07
  pickle_file            -- the file to save/load patterns from; default value is function of categories, e.g., if categories = 10, pickle_file = '10_categories.pkl'
  """
  def __init__(self, categories = 20, levels_of_distortion = 3, items_per_level = 3,
               pattern_width = 10, pattern_height = 20, max_units_set = 10,
               compression_width = 5, compression_height = 5, compression_resolution = 0.07
               pickle_file = None):

    self.categories = categories
    self.levels_of_distortion = levels_of_distortion
    self.items_per_level = items_per_level
    self.pattern_num = self.categories * (1+ self.levels_of_distortion * self.items_per_level)
    self.pattern_width = pattern_width
    self.pattern_height = pattern_height
    self.max_units_set = max_units_set
    self.patterns = np.empty((self.pattern_num, self.pattern_width, self.pattern_height))
    self.items = np.zeros_like(self.patterns)
    self.prototypes = np.zeros_like(self.patterns)

    self.compression_width = compression_width #int(self.pattern_width*0.5)
    self.compression_height = compression_height #int(self.pattern_height*0.5)
    self.compression_num = self.pattern_num
    self.compressed_representation = np.empty((self.compression_num, self.compression_width, self.compression_height))
    self.compressed_resolution = np.ones(self.compression_num)
    self.compressed_resolution.fill(compression_resolution)

    self.patterns_file = str(self.categories)+'_categories.txt'
    self.config_file = str(self.categories)+'_categories_config.txt'
    if pickle_file == None:
      self.pickle_file = str(self.categories)+'_categories.pkl'
    else:
      self.pickle_file = pickle_file

  def calculate_compressed_representation(self):
    # activation on self.compressed_representation is proportional to exp(-d/k), where d is the distance of the active pixel from the centre of the self.compressed_representation and k is a constant representing the grain of the compressed_representation

    # for each compressed_representationl image
    for n in range(0, self.compression_num):
      #calculate what the self.compressed_representation sees
      self.compressed_representation[n] = scipy.misc.imresize(self.patterns[n,:,:], (self.compression_width, self.compression_height), interp='bicubic', mode=None)
      self.compressed_representation[n] = scipy.ndimage.filters.gaussian_filter(self.compressed_representation[n], (1 - self.compressed_resolution[n]))
    self.compressed_representation /= self.compressed_representation.max(axis = 0) #nornmalise
    print self.compressed_representation.shape
    return self.compressed_representation.reshape((self.compression_num, self.compression_width*self.compression_height))

  def create_patterns(self):
    print 'create_patterns'
    # for readability I have split this into various loops; who cares about time/space complexity
    # this loop generates the basic prototype self.patterns
    coord = []
    p_coord = []

    o = 0
    for i in range(0, self.categories):

      x = Random(self.pattern_width-1)
      y = Random(self.pattern_height-1)
      units_set = 0
      while (units_set < self.max_units_set): # do this until all the units are set
        #while (p[i, x, y] == 1): # look for unit that is not set
          x = Random(self.pattern_width-1)
          y = Random(self.pattern_height-1)
          #if we don't want overlap, uncomment the following three lines
          #while (x,y) in coord: # again look for unit that has not been set in previous self.patterns
            #print x, y
            #print coord
          self.prototypes[i, x, y] = 1
          coord.append((x,y))
          units_set = units_set + 1
          #we just set a unit to on - so we want to know that for pattern i (x, y) are on

      self.patterns[o, :, :] = self.prototypes[i, :, :]
      print self.patterns[o, :, :]
      o += 1
      h = 0
      dist1 = np.zeros([self.categories])
      dist2 =  np.zeros([self.categories])
      for l in range(self.levels_of_distortion):
        for e in range(self.items_per_level):

            distortion = l+1
            item = self.generate_item(self.prototypes[i, :, :], distortion)
            for c in range(self.categories):
                dist1[c] = np.linalg.norm(item-self.prototypes[c,:,:])

            counter = 0
            while min(dist1) != dist1[i]:
                item = self.generate_item(self.prototypes[i, :, :], distortion)
                counter += 1
                print counter, l, i, self.pattern_num
                print 'this is an item that should be rejected'

                for c in range(self.categories):
                    dist1[c] = np.linalg.norm(item-self.prototypes[c,:,:])
            self.patterns[o, :, :] = item
            o += 1
            h += 1
    print self.patterns


  def generate_item(self, prototype, distortion):
    item = np.zeros_like(prototype)
    distortion = distortion * 0.25
    indices = np.asarray(np.nonzero(prototype))
    noise = np.round(np.random.uniform(-distortion,distortion,indices.shape))
    noise = np.round(np.random.normal(loc=0.0, scale=distortion, size = indices.shape)).astype(int)
    print noise, distortion/0.4, distortion
    indices += noise
    print indices

    z = np.where(indices < 0)
    indices[z] = 0
    z = np.where(indices[0,:] >= self.pattern_width)
    indices[0, z] = self.pattern_width - 1
    z = np.where(indices[1,:] >= self.pattern_height)
    indices[1, z] = self.pattern_height - 1

    indices = tuple(indices)
    item[indices] = 1
    return item

  def dendrograms(self):

      temp_categories = self.categories
      print "categories", self.categories
      
      temp_pattern_num = temp_categories * (1+ self.levels_of_distortion * self.items_per_level)
      self.compressed_representation =  self.calculate_compressed_representation()

      mat = self.patterns[0:temp_pattern_num, :, :]
      mat = mat.reshape((temp_pattern_num,self.pattern_width*self.pattern_height))

      dist_mat = sch.distance.pdist(mat, 'jaccard')
      linkage_matrix = sch.linkage(mat, "ward")
      g = plt.figure(2)
      ddata = sch.dendrogram(linkage_matrix, color_threshold=1, labels=self.concepts)

      #Assignment of colors to labels: 'a' is red, 'b' is green, etc.
      #label_colors = {'a': 'r', 'b': 'g', 'c': 'b', 'd': 'm'}
      colours = list()
      for i, a in enumerate(self.animals):
        if a:
          colours.append('r')
        else:
          colours.append('b')
          
      label_colors = {self.concepts[n]: colours[n] for n in range(self.pattern_num)}
      print label_colors

      ax = plt.gca()
      xlbls = ax.get_xmajorticklabels()
      for lbl in xlbls:
          lbl.set_color(label_colors[lbl.get_text()])

      plt.xlabel('Patterns')
      plt.ylabel('Distance')
      # We change the fontsize of minor ticks label
      plt.tick_params(axis='both', which='major', labelsize=10)
      plt.tick_params(axis='both', which='minor', labelsize=10)
      plt.show()

if __name__ == "__main__":

    p = Patterns()
    p.create_patterns()