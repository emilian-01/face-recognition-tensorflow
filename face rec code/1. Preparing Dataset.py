import os
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
#Setting names of the directories for both sets.
base_dir = 'dataset'
seta = 'George_W_Bush'
setb = 'Colin_Powell'

#Each of the sets has three sub directories train, validation and test
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

def prepare_data(base_dir, seta, setb):
# Takes the directory names for the base directory, and both the sets
# Returns the paths for train, validation for each of the sets.
	seta_train_dir = os.path.join(train_dir, seta)
	setb_train_dir = os.path.join(train_dir, setb)
	
	seta_valid_dir = os.path.join(validation_dir, seta)
	setb_valid_dir = os.path.join(validation_dir, setb)
	
	seta_train_fnames = os.listdir(seta_train_dir)
	setb_train_fnames = os.listdir(setb_train_dir)
	
	return seta_train_dir, setb_train_dir, seta_valid_dir, setb_valid_dir, seta_train_fnames, setb_train_fnames
	
seta_train_dir, setb_train_dir, seta_valid_dir, setb_valid_dir, seta_train_fnames, setb_train_fnames = prepare_data(base_dir, seta, setb)

seta_test_dir = os.path.join(test_dir, seta)
setb_test_dir = os.path.join(test_dir, setb)
test_fnames_seta = os.listdir(seta_test_dir)
test_fnames_setb = os.listdir(setb_test_dir)