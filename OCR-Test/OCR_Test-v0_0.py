############################################
##### IMPORTS ##############################
############################################

import sys
import tempfile
import cv2
import numpy as np
import pytesseract
from PIL import Image

############################################
##### FUNCITON DEFINITIONS #################
############################################

def printImageAttributes(img:Image, text:str="") -> None:
    """
    FOR DEBUGGING
    
    Outputs all pertinent properties of an Image type
    """

    file_format = img.format
    image_mode = img.mode
    image_size = img.size
    
    if (text != ""):
        text = "\t" + text + "\n\n"

    print("########## LOG ##########")
    print(f"{text}\tFILE FORMAT: {file_format}\n\tMODE: {image_mode}\n\tSIZE: {image_size[0]}x{image_size[1]}")
    print("#########################")
    print()
    
    return

def resizeImage(img:Image) -> Image:
    """
    Resize image to a larger resolution and then rescale to 300 PPI (DPI)
    
    Returns an Image type
    """
    
    len_x, len_y = img.size
    
    scaling_factor = max(1, float(1024.0 / len_x))
    new_size = int(scaling_factor * len_x), int(scaling_factor * len_y)
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    temp_filename = (tempfile.NamedTemporaryFile(delete=True, suffix='_TEMP.png')).name
    resized_img.save(temp_filename, dpi=(300, 300))
    
    return Image.open(temp_filename)

def normalizeImage(img:Image) -> Image:
    raw_img_mat = np.array(img)
    norm_img_mat = np.zeros((raw_img_mat.shape[0], raw_img_mat.shape[1]))
    norm_img_mat = cv2.normalize(raw_img_mat, norm_img_mat, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(norm_img_mat)

############################################
##### MAIN #################################
############################################

def main(document_filename:str) -> None:
    # document_filename = "case1.jpg"
    preprocessed_filename = "preproc.png"
    
    raw_img = Image.open(document_filename)
    printImageAttributes(raw_img, "RAW")

    # Normalize
    # raw_img_mat = np.array(raw_img)
    # norm_img_mat = np.zeros((raw_img_mat.shape[0], raw_img_mat.shape[1]))
    # normalized_img = cv2.normalize(raw_img_mat, norm_img_mat, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite(preprocessed_filename, normalized_img)
    # normalized_img = Image.open(preprocessed_filename)
    
    normalized_img = normalizeImage(raw_img)
    printImageAttributes(normalized_img, "NORMALIZED")
    
    cv2.imwrite(preprocessed_filename, np.array(normalized_img))
    
    return

    # Scale
    scaled_filename = resizeImage(normalized_img)
    scaled_img = Image.open(scaled_filename)
    cv2.imwrite(preprocessed_filename, np.array(scaled_img))
    # printImageAttributes(scaled_img, scaled_filename)
    scaled_img = np.array(scaled_img)

    # Denoise
    denoised_img = cv2.fastNlMeansDenoisingColored(scaled_img, None, 10, 10, 7, 15)

    # Gray
    gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(preprocessed_filename, gray_img)

    # Thinning, Skeletonization
    # kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(gray_img, kernel, iterations = 1)

    # Thresholding
    # final_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    final_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite(preprocessed_filename, final_img)

    printImageAttributes(Image.open(preprocessed_filename), preprocessed_filename)

    text = pytesseract.image_to_string(final_img)
    print(text)
    
if __name__ == "__main__":
    input_args = list(map(str, sys.argv[1:]))
    main(input_args[0])
