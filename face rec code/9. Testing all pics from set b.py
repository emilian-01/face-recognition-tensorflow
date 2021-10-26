cal_cp = 0
cal_gwb = 0
cal_unconclusive = 0
blist = []
for fname in test_fnames_setb:
	if fname.startswith('.'):
		continue
	file_path = os.path.join(setb_test_dir, fname)
	load_file = load_img(file_path, target_size = (150,150))
	load_file = (np.expand_dims(load_file, 0))
	load_file = tf.cast(load_file, tf.float32)
	pred_img = model.predict(load_file)
	if(pred_img[0] < 0.5):
		cal_cp += 1
	elif(pred_img[0] > 0.5):
		cal_gwb += 1
	else:
		print(pred_img[0], "\n")
		cal_unconclusive += 1
		blist.append(file_path)
print(blist)

print("Identified as: \n")
print("Colin Powell :", cal_cp)
print("George W Bush:", cal_gwb)
print("Inconclusive :", cal_unconclusive)
print("Percentage :",(cal_gwb/(cal_gwb + cal_unconclusive + cal_cp))*100)
b = (cal_cp/(cal_gwb + cal_unconclusive + cal_cp)) * 100

avg = (a + b)/2
print("Average Percentage :", avg)
