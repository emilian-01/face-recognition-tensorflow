rand_test_img = random.choice(test_fnames_setb)
rand_test_image_path = os.path.join(setb_test_dir, rand_test_img)
rand_test_img = load_img(rand_test_image_path, target_size = (150, 150))
plt.imshow(rand_test_img)
rand_test_img = (np.expand_dims(tand_test_img, 0))
print(tand_test_img.shape)

print("Idetified as:\n")
if(model.predict(train_img) < 0.5):
	print("Collin Powell")
elif(model.predict(train_img) > 0.5):
	print("George W Bush")
else:
	print("Inconclusive")