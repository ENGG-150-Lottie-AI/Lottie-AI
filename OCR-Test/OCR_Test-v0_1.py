"""
TESSERACT OCR TEST v0.1

Basic OCR. Scans an entire image.
"""

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
    """ FOR DEBUGGING. Outputs all pertinent properties of an Image type. """

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

def normalizeImage(img:Image) -> Image:
    """ Normalizes pixel intensity values in an image. Returns an Image type. """
    
    raw_img_mat = np.array(img)
    norm_img_mat = np.zeros((raw_img_mat.shape[0], raw_img_mat.shape[1]))
    norm_img_mat = cv2.normalize(raw_img_mat, norm_img_mat, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(norm_img_mat)

def resizeImage(img:Image) -> Image:
    """ Resizes an image to a larger resolution and then rescales it to 300 PPI (DPI). Returns an Image type. """
    
    len_x, len_y = img.size
    
    scaling_factor = max(1, float(1024.0 / len_x))
    new_size = int(scaling_factor * len_x), int(scaling_factor * len_y)
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    temp_filename = (tempfile.NamedTemporaryFile(delete=True, suffix='_TEMP.png')).name
    resized_img.save(temp_filename, dpi=(300, 300))
    
    return Image.open(temp_filename)
    
def denoiseImage(img:Image) -> Image:
    """ Removes noise in an image. Returns an Image type. """
    
    denoised_img_mat = cv2.fastNlMeansDenoisingColored(np.array(img), None, 10, 10, 7, 15)
    return Image.fromarray(denoised_img_mat)

def grayscaleImage(img:Image) -> Image:
    """ Converts the color space of an image to gray (luminosity-only). Returns an Image type. """
    gray_img_mat = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray_img_mat)

def skeletonizeImage(img:Image) -> Image:
    """ FOR HANDWRITTEN TEXT. Skeletonizes an image by making stroke widths uniform. Returns an Image type. """
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(np.array(img), kernel, iterations = 1)
    return Image.fromarray(erosion)

def thresholdImage(img:Image) -> Image:
    """ Limits the color space to strictly black and white. Returns an Image type. """
    
    thresh_img_mat = cv2.adaptiveThreshold(
        np.array(img), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10)
    
    return Image.fromarray(thresh_img_mat)

def thresholdImage_NEW(img:Image) -> Image:
    thresh_img_mat = cv2.threshold(np.array(img), 100, 255, cv2.THRESH_TOZERO)[1]
    return Image.fromarray(thresh_img_mat)

def checkAccuracy(target:str, recognized:str) -> float:
    target_alpha = [i.upper() for i in target if i.isalnum()]
    recognized_alpha = [i.upper() for i in recognized if i.isalnum()]
    
    print("########## LOG ##########")
    blank = ""
    print(f"\tTarget Text (w/o whitespace): {blank.join(target_alpha)}\n\n\tRecognized Test (w/o whitespace): {blank.join(recognized_alpha)}")
    print("#########################")
    
    match_count = 0
    total_count = len(target_alpha)
    
    t = 0
    r = 0
    while (t < len(target_alpha) and r < len(recognized_alpha)):
        if (target_alpha[t] == recognized_alpha[r]):
            match_count += 1
        t += 1
            
        r += 1        
    
    return round((match_count/total_count)*100, 3)

def showOCRData(img:cv2.Mat, ocr_psm:int=3) -> cv2.Mat:
    results = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=f"--psm {ocr_psm}")
    
    for i in range(len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]

        w = results["width"][i]
        h = results["height"][i]

        text = results["text"][i]
        conf = int(results["conf"][i])
        
        if conf < 0:
            continue
        
        text = "".join([c if ord(c) < 128 else "?" for c in text]).strip()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
    return img

############################################
##### MAIN #################################
############################################

def main() -> None:
    # Set-up input image
    document_filename = "case1.jpg"
    preprocessed_filename = "img_preproc.png"
    results_filename = "img_result.png"
    
    raw_img = Image.open(document_filename)
    preprocessed_img = raw_img
    printImageAttributes(raw_img, "RAW")

    # Normalize
    preprocessed_img = normalizeImage(raw_img)
    printImageAttributes(preprocessed_img, "NORMALIZED")
    cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Scale
    preprocessed_img = resizeImage(preprocessed_img)
    printImageAttributes(preprocessed_img, "RESCALED")
    cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Denoise
    # preprocessed_img = denoiseImage(preprocessed_img)
    # printImageAttributes(preprocessed_img, "DENOISED")
    # cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Gray
    preprocessed_img = grayscaleImage(preprocessed_img)
    printImageAttributes(preprocessed_img, "GRAYSCALE")
    cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Thinning and skeletonization
    # preprocessed_img = skeletonizeImage(preprocessed_img)
    # printImageAttributes(preprocessed_img, "SKELETONIZED")
    # cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Thresholding
    # preprocessed_img = thresholdImage_NEW(preprocessed_img)
    # # preprocessed_img = thresholdImage(preprocessed_img)
    # printImageAttributes(preprocessed_img, "THRESHOLD")
    # cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Use Tesseract OCR to read text
    ocr_psm = 3
    recognized_text = pytesseract.image_to_string(preprocessed_img, config=f"--psm {ocr_psm}")
    
    print("########## LOG ##########")
    print(recognized_text[:-1])
    print("#########################")
    
    # Check accuracy based on expected output (strict)
    with open("TARGET_CASE1.txt", "r") as file:
        target_text = file.read()
        print(f"Accuary = {checkAccuracy(target_text, recognized_text)}%", file=sys.stderr)
    
    # Show recognized text on image
    results_img = showOCRData(cv2.imread(preprocessed_filename), ocr_psm)
    cv2.imwrite(results_filename, results_img)
    
    return
    
if __name__ == "__main__":
    main()
