datagen = ImageDataGenerator(
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = 'nearest')

img_path = os.path.join(seta_train_dir, seta_train_fnames[3])
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1):
	plt.figure(i)
	imgplot = plt.imshow(array_to_img(batch[0]))
	i += 1
	if i % 5 == 0:
		break