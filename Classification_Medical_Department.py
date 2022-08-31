import os
import cv2
import tensorflow as tf
import numpy as np
from keras import layers, optimizers
from keras.applications import ResNet50
from keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

x_ray_dir = r'C:\Users\naoin\pythonProject\Pandasbootcamp\Portfolio\Classification_Medical_Department\Medical_Department\Dataset'

'''For this classification method I'll load some image data for patients with covid, normal health, viral pneumonia and
bacterial pneumonia'''

# PRINT 0

'''Let's make an image normalization, reshape, create a label so we can separate all the cases based on each
condition the patient might have, after that I'll plot the data for better visualization'''

image_generator = ImageDataGenerator(rescale=1. / 255)
train_generator = image_generator.flow_from_directory(batch_size=40, directory=x_ray_dir, shuffle=True,
                                                      target_size=(256, 256), class_mode='categorical',
                                                      subset='training')

train_images, train_labels = next(train_generator)
print(train_images.shape)

label_names = {0: 'Covid-19', 1: 'Normal', 2: 'Viral Pneumonia', 3: 'Bacterial Pneumonia'}

#  todo VISUALIZATION ------------------------

fig, axes = plt.subplots(6, 6, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, 36):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
plt.show()

# PRINT 1

'''Here we can start the convolutional layers followed by some dense layers, so we can apply filters, pooling,
down sample and extract the features from the images so our model can detect this features and be able to use the fully
connected layers to learn and be able to classify new data. We're using the resnet model, which is a model
trained with 11 million images from 11 thousand different categories, this way we can save a lot of time
using transfer of learning'''

#  todo LOAD RESNET ------------------------

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))
base_model.summary()

# PRINT 2

'''Now that we excluded the top (the dense part of this model), we can freeze the already trained weights
and drop some of the info from the model. After that I'll be applying some pooling and flattening
so I can create our own fully connected convolutional neural network'''

for layer in base_model.layers[:-10]:
    layers.trainable = False

#  todo NEURAL NETWORK ------------------------

head_model = base_model.output
head_model = AveragePooling2D()(head_model)
head_model = Flatten()(head_model)
head_model = Dense(256, activation='relu')(head_model)
head_model = Dropout(0.2)(head_model)
head_model = Dense(256, activation='relu')(head_model)
head_model = Dropout(0.2)(head_model)
head_model = Dense(4, activation='softmax')(head_model)

'''I'll also create a checkpoint save and define the optimizers and metrics followed by a training with a
checkpoint to save the latest trained weight, so we don't have to repeat this process and also to have
a comparison in case we need to test with more epochs'''

model = Model(inputs=base_model.input, outputs=head_model)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4, decay=1e-6),
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='weights.hdf5')

train_generator = image_generator.flow_from_directory(batch_size=4, directory=x_ray_dir, shuffle=True,
                                                      target_size=(256, 256), class_mode='categorical',
                                                      subset='training')

history = model.fit(train_generator, epochs=25, callbacks=[checkpoint])

# PRINT 3

#  todo VISUALIZATION ------------------------

'''Let's check the accuracy and error during the epochs only for the training set'''

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Error and rate of accuracy while training')
plt.xlabel('Epoch')
plt.ylabel('Rate of Accuracy and error')
plt.legend(['Rate of accuracy', 'Error'])
plt.show()

# PRINT 4

'''We can see the rate of accuracy and error margin so we can understand if the model need more or less training, also
now we can import the images we want to test so we can check the answers on a non-trained dataset, because using
the same set for answers usually presents us with a higher accuracy, and that can be a problem'''

#  todo TEST ------------------------

test_dir = r'C:\Users\naoin\pythonProject\Pandasbootcamp\Portfolio\Classification_Medical_Department\Medical_Department\Test'

'''Let's apply the same treatment we did in our training set'''

test_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_gen.flow_from_directory(batch_size=40, directory=test_dir, shuffle=True,
                                              target_size=(256, 256), class_mode='categorical')

'''Let's see how the model handled a different dataset without knowing the answers already'''

evaluate = model.evaluate(test_generator)
print(evaluate)

'''We can see that we had way more accuracy, because we were testing with the training dataset we had more
accuracy, now that we have different images to understand, the accuracy fell a bit'''

# PRINT 5

'''Let's do the same we did before and normalize, resize, reshape, pass the images and create our prediction'''

#  todo TEST SET PROCESSING ------------------------

prediction = []
original = []
image = []

for i in range(len(os.listdir(test_dir))):
    for item in os.listdir(os.path.join(test_dir, str(i))):
        img = cv2.imread(os.path.join(test_dir, str(i), item))
        img = cv2.resize(img, (256, 256))
        image.append(img)
        img = img / 255
        img = img.reshape(-1, 256, 256, 3)
        predict = model.predict(img)
        predict = np.argmax(predict)
        prediction.append(predict)
        original.append(i)


'''So our images will look like that, just like our training dataset'''

# PRINT 6

'''Now I'll check the metrics with the test dataset, we're looking for accuracy score, a direct comparison
confusion matrix and f1-score and recall (classification report)'''

#  todo METRICS ------------------------

print(accuracy_score(original, prediction))

# PRINT 7

'''Let's make a visual comparison between de predictions'''

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, 25):
  axes[i].imshow(image[i])
  axes[i].set_title(f'Pred={str(label_names[prediction[i]])}\nClass={str(label_names[original[i]])}')
  axes[i].axis('off')
plt.subplots_adjust(wspace=1.2)
plt.show()

# PRINT 8

'''Usually for classification models confusion matrix and classification report can deliver more interesting results
because we can check metrics with way more precision'''

cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot=True)
plt.show()

# PRINT 9

'''Now we understand what the model got right and wrong, now the classification report'''

print(classification_report(original, prediction))

# PRINT 10

#  todo SINGULAR TEST ------------------------

'''Now just to make sure everything is working fine, I'll create a test with a single image and repeat the
data processing with the weights of the model already loaded, so we don't have to run anything again'''

model_loaded = load_model(r'C:\Users\naoin\pythonProject\Pandasbootcamp\Portfolio\Classification_Medical_Department\weights.hdf5')

img = cv2.imread(r'C:\Users\naoin\pythonProject\Pandasbootcamp\Portfolio\Classification_Medical_Department\Medical_Department\Test\0\radiol.2020200490.fig3.jpeg')
img = cv2.resize(img, (256, 256))
img = img / 255
img = img.reshape(-1, 256, 256, 3)

predict = model_loaded(img)
print(predict)
predict2 = np.argmax(predict)
print(label_names[predict2], predict[0][predict2])

# PRINT 11

'''As we can see our model is pretty good for Covid and Normal patients, so when I tested a Covid x-ray
the model predicted Covid-19 with 1.0 accuracy, which means 100% of accuracy about the decision'''
