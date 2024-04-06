from image_tensors import get_image_tensors

import pandas as pd
import tensorflow as tf
# from tensorflow.python.keras import Input

INPUT_DIRECTORY = 'Path2/Path2-Model Training/Path2 Training Images'
BATCH_SIZE = 40
TARGET_SIZE = (768,572)

excel = pd.read_excel('Path2/Path2-Model Training/Path2 Data.xlsx')
columns = excel[['Ribeye (sq inches)', 'FatThickness, inches']]
excel_data = tf.convert_to_tensor(columns.values)

image_data =  get_image_tensors(INPUT_DIRECTORY)
