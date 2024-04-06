import pandas as pd 
#from sklearn.model_selection import train_test_split

#from tensorflow.python import Sequential, load_model 
#from tensorflow.python import Dense 
#from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Concatenate
from keras.optimizers import Adam

# Define the image input
image_input = Input(shape=(769,572,3))

# Define additional data input (e.g., numerical features)
additional_data_input = Input(shape=(2, 1))
# additional data length             ^

# CNN model for image processing
conv_layer_image = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(image_input)
flatten_layer_image = Flatten()(conv_layer_image)

# Convolutional layer for additional data
conv_layer_additional = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(additional_data_input)
flatten_layer_additional = Flatten()(conv_layer_additional)

# Concatenate image and additional data
concatenated = Concatenate()([flatten_layer_image, flatten_layer_additional])

# Output layer
output = Dense(num_classes, activation='softmax')(concatenated)

# Create and compile the model
model = Model(inputs=[image_input, additional_data_input], outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([image_data, additional_data], labels, epochs=10, batch_size=40, validation_split=0)
