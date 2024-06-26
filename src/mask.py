import numpy as npsrc
import os
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
#from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt  

# Define paths to your dataset
BATCH_SIZE = 40
OUTPUTDIRECTORYNAME = 'Path2/Path2-Model Training/Path2 Fat Thickness Masks'
OUTPUTDIRECTORYNAMEVAlIDATION = 'Path2/Path2-Model Training/Path2 Fat Thickness Masks'
#TODO We need to create this mask
masks_dir = 'Path2/Path2-Model Training/Path2 Fat Thickness Masks/'
ribeye_masks_dir = 'Path2/Path2-Model Training/Path2 Fat Thickness Masks/'
validation_fat_thickness = 'Path2/Path2-Model Validation/Path2-Fat Thickness'

# Define target size for images and masks
target_size = (572, 768)  # adjust according to your needs

def mask_list(images_path):
    masks = []
    for image_name in os.listdir(images_path):
        print(image_name)
        #mask_path = image_name.split('/')[-1].split('.')[0] + '_mask.png'
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
    images_dir = 'Path2/Path2-Model Training/Path2 Training Images'
    #images_dir = 'Path2/Path2-Model Validation/Path2 Images For Validation'
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

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    masked_image = cv2.bitwise_and(image_bgr,image_bgr, mask=finalMask)

    masks_dir = 'Path2/Path2-Model Training/Path2 Fat Thickness/'
    MASK_NAME = masks_dir +  image_path + '_mask.png'
    #MASK_NAME = validation_fat_thickness + '/' + image_path + '_mask.png'
    #MASK_NAME = "generatedImage.jpg"
    cv2.imwrite(MASK_NAME,masked_image)

def create_ribeye_mask(image_path):
    # Read the image and Convert the image to grayscale
    image = cv2.imread(images_dir + "/" + image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # START ADDED
    # # Define the region of interest (ROI) coordinates
    # x, y, w, h = 192,0,384,286  # Example ROI coordinates (x, y, width, height)

    # # Create an empty mask with the same dimensions as the image
    # mask_gray = npsrc.zeros(target_size, dtype=npsrc.uint8)

    # # Draw a filled rectangle on the mask to define the ROI
    # cv2.rectangle(mask_gray, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)

    # mask = cv2.bitwise_and(gray, gray, mask=mask_gray)
    # # END ADDED

    # Threshold the image to extract fat regions
    hierarchy, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # #CASE FAT THICKNESS IS NORMAL
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # #iterates through the contours and populates area with areas
    # contour_areas = [cv2.contourArea(contour) for contour in contours]
    # # Get largest contour
    # largest_contour_index = contour_areas.index(max(contour_areas))
    # largest_contour = contours[largest_contour_index]

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(image_path + " " + str(cv2.contourArea(largest_contour)))
    
    # Select the second largest contour
    if len(contours) > 1:
        largest_contour = contours[1]  # Use the second largest contour
    else:
        largest_contour = contours[0]  # Use the largest contour if only one contour is found
    #END

    finalMask = npsrc.zeros_like(gray)
    #cv2.fillPoly(mask, pts=closed_contour, color=(255))
    cv2.drawContours(finalMask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    MASK_NAME = ribeye_masks_dir + image_path + '_mask.png'
    #MASK_NAME = "generatedImage.jpg"
    cv2.imwrite(MASK_NAME,finalMask)

mask_list('Path2/Path2-Model Training/Path2 Training Images')
#create_mask('00001325-1.tif')
