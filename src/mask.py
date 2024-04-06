import numpy as npsrc
import os
#from keras.preprocessing.image import ImageDataGenerator
#from skimage.io import imread
#from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt  

# Define paths to your dataset
images_dir = 'Path2/Path2-Model Training/Path2 Training Images'
BATCH_SIZE = 40
OUTPUTDIRECTORYNAME = 'Path2/Path2-Model Training/Path2 Training Masks'
#TODO We need to create this mask
masks_dir = 'path_to_masks_directory'

# Define target size for images and masks
target_size = (572, 768)  # adjust according to your needs

def mask_list(images_path):
    os.mkdir(OUTPUTDIRECTORYNAME)
    for image_name in os.listdir(images_path):
        create_mask()
        #TODO save mask to a folder/whatever is best for the data img generator, maybe just a python array of contours
    mask_datagen = ImageDataGenerator()
    mask_datagen.fit(masks, augment=True, seed=seed)
    mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=BATCH_SIZE)

def create_mask(image_path):
    # Read the image and Convert the image to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to extract fat regions
    hierarchy, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours of fat regions
    #TODO this can also be done with a gray-scale image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #iterates through the contours and populates area with areas
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # Get largest contour
    largest_contour_index = contour_areas.index(max(contour_areas))
    largest_contour = contours[largest_contour_index]

    #View where the contours be at
    view_image = cv2.imread(image_path)
    view_image_rgb = cv2.cvtColor(view_image, cv2.COLOR_BGR2RGB)

    #mask = npsrc.zeros_like(gray)
    #cv2.fillPoly(mask, pts=closed_contour, color=(255))
    #cv2.drawContours(mask, largest_contour, -1, (0,255,0), thickness=cv2.FILLED)

    cv2.drawContours(view_image_rgb, largest_contour, -1, (0,255,0),3)
    #cv2.fillPoly(view_image_rgb, pts = largest_contour, color=(0,255,0))
    cv2.imwrite('measuredContour.jpg',view_image_rgb)
    plt.imshow(view_image_rgb)

create_mask('Path2/Path2-Model Training/Path2 Training Images/00001273-2.tif')
        
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