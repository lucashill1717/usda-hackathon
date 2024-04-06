import pandas as pd
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory as idfd
from PIL import Image, ImageSequence
# from tensorflow.python.keras import Input

DIRECTORY = 'Path2/Path2-Model Training/Path2/Path2-Model Training/Path2 Training Images';
BATCH_SIZE = 40
TARGET_SIZE = (768,572)

excel = pd.read_excel('Path2/Path2-Model Training/Path2 Data.xlsx')
columns = excel[['Ribeye (sq inches)', 'FatThickness, inches']]
data_tensor = tf.convert_to_tensor(columns.values)

list_of_image_tensors = idfd(
    DIRECTORY,
    image_size = TARGET_SIZE,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    shuffle=False,
)

print(list_of_image_tensors)