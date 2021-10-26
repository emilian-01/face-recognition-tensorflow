# Accurracy results for each training and validation epoch
acc = mymodel.history['acc']
val_acc = mymodel.history['val_acc']

# Loss Results for each training and validation epoch
loss = mymodel.history['loss']
val_loss = mymodel.history['val_loss']

epochs = range(len(acc))

# Plot accuracy for each training and validation epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot loss for each training and validation epoch
plt.plot(epochs, loss)
plt.plot(apochs, val_loss)
plt.title('Training and validation loss')
