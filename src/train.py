import keras
from model import image_data, excel_data, validation_data

model = keras.saving.load_model('model_0.keras')
model.fit(image_data, excel_data, epochs=50, batch_size=40)
print(model.predict(validation_data, batch_size=10))
