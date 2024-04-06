from keras.models import Model, clone_model
from keras.layers import Input, Dense, Flatten, Conv2D, Concatenate, GlobalAveragePooling2D, Reshape, Multiply
from keras.optimizers import Adam
from model import excel_data_one, excel_data_two, image_data

# Define the image input
image_input = Input(shape=(572,768,3), batch_size=40)

# CNN model for image processing
conv_layer_image = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(image_input)
# Add more convolutional layers if we need

# Attention mechanism
attention_weights = GlobalAveragePooling2D()(conv_layer_image)  # Global average pooling to obtain attention weights
attention_weights = Reshape((1, 1, 32))(attention_weights)  # Reshape to match conv_layer_image's shape
attention_weights = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(attention_weights)  # Sigmoid activation for attention weights
attention_weights = Multiply()([conv_layer_image, attention_weights])  # Apply attention weights to conv_layer_image

# CNN layers after applying attention
conv_layer_image_with_attention = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(attention_weights)
# Add more convolutional layers if we need

# Flatten and concatenate
flatten_layer_image = Flatten()(conv_layer_image_with_attention)
# Add additional data input and concatenate if we need

# Output layer
output = Dense(1, activation='linear')(flatten_layer_image)  # Assuming single output for fat thickness prediction

# Create and compile the model
model = Model(inputs=image_input, outputs=output)
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
# model.fit(image_data, excel_data_one, epochs=10, batch_size=40)

# model.fit(image_data, excel_data, epochs=10, batch_size=40)
# print(model.predict(validation_data))

import matplotlib.pyplot as plt

# Train the model
model_copy = clone_model(model)
history = model.fit(image_data, excel_data_one, epochs=15, batch_size=40)
model_copy.fit(image_data, excel_data_two, epochs=15, batch_size=40)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper right')
plt.show()