import numpy as npsrc
import os
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
#from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt  

# Define paths to your dataset
images_dir = 'Path2/Path2-Model Training/Path2 Training Images'
BATCH_SIZE = 40
OUTPUTDIRECTORYNAME = 'Path2/Path2-Model Training/Path2 Training Masks'
#TODO We need to create this mask
masks_dir = 'Path2/Path2-Model Training/Path2 Training Masks/'

# Define target size for images and masks
target_size = (572, 768)  # adjust according to your needs

def mask_list(images_path):
    os.mkdir(OUTPUTDIRECTORYNAME)
    masks = []
    for image_name in os.listdir(images_path):
        mask_path = image_name.split('/')[-1].split('.')[0] + '_mask.png'
        create_mask(image_name)
    #     if os.path.exists(mask_path):
    #         mask = imread(mask_path)
    #         masks.append(mask)
    # seed = 1
    # mask_datagen = ImageDataGenerator()
    # mask_datagen.fit(masks, augment=True, seed=seed)
    # mask_tensor = mask_datagen.flow(masks, seed=seed, batch_size=BATCH_SIZE)
    # return mask_tensor


def create_mask(image_path):
    # Read the image and Convert the image to grayscale
    image = cv2.imread(images_dir + "/" + image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # START ADDED
    # Define the region of interest (ROI) coordinates
    x, y, w, h = 192,0,384,286  # Example ROI coordinates (x, y, width, height)

    # Create an empty mask with the same dimensions as the image
    mask_gray = npsrc.zeros(target_size, dtype=npsrc.uint8)

    # Draw a filled rectangle on the mask to define the ROI
    cv2.rectangle(mask_gray, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)

    mask = cv2.bitwise_and(gray, gray, mask=mask_gray)
    # END ADDED

    # Threshold the image to extract fat regions
    hierarchy, binary = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    #CASE FAT THICKNESS IS NORMAL
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #iterates through the contours and populates area with areas
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    # Get largest contour
    largest_contour_index = contour_areas.index(max(contour_areas))
    largest_contour = contours[largest_contour_index]

    print(image_path + " " + str(cv2.contourArea(largest_contour)))

    # # Sort contours by area in descending order
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # # Select the second largest contour
    # if len(contours) > 1:
    #     largest_contour = contours[1]  # Use the second largest contour
    # else:
    #     largest_contour = contours[0]  # Use the largest contour if only one contour is found
    # #END

    finalMask = npsrc.zeros_like(gray)
    #cv2.fillPoly(mask, pts=closed_contour, color=(255))
    cv2.drawContours(finalMask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    MASK_NAME = masks_dir + image_path + '_mask.png'
    #MASK_NAME = "generatedImage.jpg"
    cv2.imwrite(MASK_NAME,finalMask)

mask_list(images_dir)
#create_mask('00001325-1.tif')

# # Function to load and preprocess images and masks
# def load_data(images_path, masks_path, target_size):
#     images = []
#     masks = []
#     for image_name in os.listdir(images_path):
#         #reads gets image path
#         image_path = os.path.join(images_path, image_name)
#         #reads mask path based on image name
#         mask_path = os.path.join(masks_path, image_name.split('.')[0] + '_mask.png')
#         if os.path.exists(mask_path):
#             # Load and preprocess image
#             image = imread(image_path)
#             image = resize(image, target_size)
#             images.append(image)
#             # Load and preprocess mask
#             mask = imread(mask_path)
#             mask = resize(mask, target_size)
#             masks.append(mask)
#         else: 
#             print("Image not found")
#     return np.array(images), np.array(masks)

# def join_images_and_masks():
#     # Load and preprocess data
#     images, masks = load_data(images_dir, masks_dir, target_size)

#     # Data augmentation using ImageDataGenerator

#     image_datagen = ImageDataGenerator()
#     mask_datagen = ImageDataGenerator()

#     # Provide same seed and keyword arguments to the fit and flow methods
#     seed = 1
#     image_datagen.fit(images, augment=True, seed=seed)
#     mask_datagen.fit(masks, augment=True, seed=seed)

#     image_generator = image_datagen.flow(images, seed=seed, batch_size=BATCH_SIZE)
#     mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=BATCH_SIZE)

#     # Combine generators into one which yields image and masks
#     train_generator = zip(image_generator, mask_generator)

# masks_dir = mask_list(images_dir)
# #TODO this code does not return anything
# join_images_and_mask()