import cv2
import tensorflow as tf
import os



def _tif_to_tensor(tif_path: str) -> tf.Tensor:
    image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    return tf.convert_to_tensor(image, dtype=tf.float32)

def _png_to_tensor(mask_path) -> tf.Tensor:
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    return tf.convert_to_tensor(mask, dtype=tf.float32)

def get_image_tensors(path: str) -> tf.Tensor:
    #TrainingImages
    image_tensors = []
    file_list = os.listdir(path)
    file_list.sort()
    #Fat Thickness Images
    mask_tensors = []
    for file in file_list:
        # image = _tif_to_tensor(os.path.join(path + IMAGEDIR, file))
        mask = _png_to_tensor(os.path.join(path, file))
        # image_tensors.append(image)
        mask_tensors.append(mask)
    
    #return tf.stack(image_tensors, axis=0)
    return tf.stack(mask_tensors, axis=0)
