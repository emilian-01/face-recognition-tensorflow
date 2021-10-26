import tensorflow as tf
# Using binnary_cossentropy as the loss function
# and Adam Optimizer as the optimizing function when training
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
              metrics = ['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen =ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_directory(
    train_dir, # Training directory
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')

validation_generator = val_datagen.flow_from_from_directory(
	validation_dir, #Validationdirectory
	target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')

import matplotlib.image as mping

# 4x4 grid
nrows = 5
ncols = 5

pic_index = 0

# Set up matplotlib fig, and size it to fit 5x5 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)

pic_index += 10
next_seta_pix = [os.path.join(seta_train_dir, fname)
                 for fname in seta_train_fnames[pic_index-10:pic_index]]
next_setb_pix = [os.path.join(setb_train_dir, fname)
                 for fname in setb_train_fnames[pic_index-10:pic_index]]

for i, img_path in enumerate(next_seta_pix + next_setb_pix):
	# Set up subplot; subplot indices start at 1
	sp = plt.subplot(nrows, ncols, i + 1)
	sp.axis('off') # Don't show axes (or gridlines)
	
	img = mpimg.imread(img_path)
	plt.imshow(img)

plt.show()

#train the model
mymodel = model.fit_generator(
	train_generator,
	steps_per_epoch = 10,
	epochs = 80,
	validation_data = validation_generator,
	validation_steps = 7,
	verbose = 2)