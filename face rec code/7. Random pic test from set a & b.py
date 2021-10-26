##### Testing model on a random train image from set a and b

train_img = random.choice(seta_train_fnames)
train_image_path = os.path.join(seta_train_dir, train_img)
train_img = load_img(train_image_path, target_size = (150, 150))
plt.imshow(train_img)
train_img = (np.expand_dims(train_img, 0))
print(train_img.shape)

train_img = tf.cast(train_img, tf.float32)
model.predict(train_img)

train_img = random.choice(setb_train_fnames)
train_image_path = os.path.join(setb_train_dir, train_img)
train_img = load_img(train_image_path, target_size = (150, 150))
plt.imshow(train_img)
train_img = (np.expand_dims(train_img, 0))
print(train_img.shape)

train_img = tf.cast(train_img, tf.float32)
model.predict(train_img)