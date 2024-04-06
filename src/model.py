import pandas as pd
import tensorflow as tf
# from tensorflow.python.keras import Input

excel = pd.read_excel('Path2/Path2-Model Training/Path2 Data.xlsx')
columns = excel[['Ribeye (sq inches)', 'FatThickness, inches']]
data_tensor = tf.convert_to_tensor(columns.values)
