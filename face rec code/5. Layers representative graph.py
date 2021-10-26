import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

successive_outputs = [layers.output for layers in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

a_img_files = [os.path.join(seta_train_dir, f) for f in seta_train_fnames]
b_img_files = [os.path.join(setb_train_dir, f) for f in setb_train_fnames]
img_path = random.choice(a_img_files + b_img_files)

img = load_img(img_path, target_size = (150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255

successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
	if len(feature_map.shape) == 4:
		# Just do this for the conv / maxpool layers, not the fully-connected layers
		n_features = feature_map.shape[-1] # number of features in feature map
		# The feature map has shape (1, size, size, n_features)
		size = feature_map.shape[1]
		# We will tile our images in this matrix
		display_grid = np.zeros((size, size * n_features))
		for i in range(n_features):
			# Postprocess the feature to make it bisually platable
			x = feature_map[0, :, :, i]
			x -= x.mean()
			x /= x.std()
			x *= 64
			x += 128
			x = np.clip(x, 0, 255).astype('uint8')
			# We'll tile each filter into this big horizontal grid
			display_grid[:, i * size : (i + 1) * size] = x
		# Display the grid
		scale = 20. / n_features
		plt.figure(figsize = (scale * n_features, scale))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')