"""
TESSERACT OCR TEST v0.3

Divides an image into lines then divides the lines into sections of text. Each text in each section is recognized.
"""

############################################
##### IMPORTS ##############################
############################################

import sys
import tempfile
import numpy as np

import cv2
import pytesseract
from PIL import Image

############################################
##### FUNCITON DEFINITIONS #################
############################################

def printImageAttributes(img:Image.Image, text:str="") -> None:
    """ FOR DEBUGGING. Outputs all pertinent properties of an Image type. """

    image_file_format = type(img).__name__
    image_mode = img.mode
    image_size = img.size
    image_color_palette = img.palette
    
    if (text != ""):
        text = "\t" + text + "\n\n"

    print("########## LOG ##########")
    print(f"{text}\tOBJECT FORMAT: {image_file_format}\n\tMODE: {image_mode}\n\tCOLOR SPACE: {image_color_palette}\n\tSIZE: {image_size[0]}x{image_size[1]}")
    print("#########################")
    print()
    
    return

def normalizeImage(img:Image.Image) -> Image.Image:
    """ Normalizes pixel intensity values in an image. Returns an Image type. """
    
    raw_img_mat = np.array(img)
    norm_img_mat = np.zeros((raw_img_mat.shape[0], raw_img_mat.shape[1]))
    norm_img_mat = cv2.normalize(raw_img_mat, norm_img_mat, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(norm_img_mat)

def resizeImage(img:Image.Image) -> Image.Image:
    """ Resizes an image to a larger resolution and then rescales it to 300 PPI (DPI). Returns an Image type. """
    
    len_x, len_y = img.size
    
    scaling_factor = max(1, float(1024.0 / len_x))
    new_size = int(scaling_factor * len_x), int(scaling_factor * len_y)
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    return resized_img
    
    temp_filename = (tempfile.NamedTemporaryFile(delete=True, suffix='_TEMP.png')).name
    resized_img.save(temp_filename, dpi=(300, 300))
    
    return Image.open(temp_filename)
    
def denoiseImage(img:Image.Image) -> Image.Image:
    """ Removes noise in an image. Returns an Image type. """
    
    denoised_img_mat = cv2.fastNlMeansDenoisingColored(np.array(img), np.array(None), 10, 10, 7, 15)
    return Image.fromarray(denoised_img_mat)

def grayscaleImage(img:Image.Image) -> Image.Image:
    """ Converts the color space of an image to gray (luminosity-only). Returns an Image type. """
    gray_img_mat = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray_img_mat)

def thresholdImage(img:Image.Image) -> Image.Image:
    """ Limits the color space to strictly black and white. Returns an Image type. """
    
    thresh_img_mat = cv2.adaptiveThreshold(
        np.array(img), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 15)
    
    # img2 = cv2.bitwise_not(np.array(img))
    # thresh_img_mat = cv2.threshold(img2, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_TOZERO)[1]
    # thresh_img_mat = invertImage(Image.fromarray(thresh_img_mat))
    # return thresh_img_mat
    
    return Image.fromarray(thresh_img_mat)

def invertImage(img:Image.Image) -> Image.Image:
    return Image.fromarray(cv2.bitwise_not(np.array(img)))

def getImageHistogram(img:Image.Image) -> np.ndarray:
    return cv2.reduce(np.array(img), 1, cv2.REDUCE_AVG).reshape(-1)

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
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
    return img

############################################
##### MAIN #################################
############################################

def main() -> None:
    # Set-up input image
    document_filename = "case1.jpg"
    # document_filename = "eeeeeeeeeeeeeeeee.jpg"
    preprocessed_filename = "img_preproc.png"
    text_lines_filename = "img_textlines.png"
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

    # Threshold
    preprocessed_img = thresholdImage(preprocessed_img)
    printImageAttributes(preprocessed_img, "THRESHOLD")
    cv2.imwrite(preprocessed_filename, np.array(preprocessed_img))

    # Get histogram and get text lines
    inverted_img = invertImage(preprocessed_img)
    histogram = getImageHistogram(inverted_img)
    
    histogram_threshold = 0
    w, h = inverted_img.size
    upper_line_bounds = [y for y in range(h-1) if histogram[y] <= histogram_threshold and histogram[y+1] > histogram_threshold]
    lower_line_bounds = [y for y in range(h-1) if histogram[y] > histogram_threshold and histogram[y+1] <= histogram_threshold]
    
    # OPTIONAL: Draw text lines
    text_lines_img_RGB_mat = cv2.cvtColor(np.array(preprocessed_img), cv2.COLOR_GRAY2BGR)
    for u, l in zip(upper_line_bounds, lower_line_bounds):
        cv2.line(text_lines_img_RGB_mat, (0, u), (w, u), (255, 0, 0), 1)
        cv2.line(text_lines_img_RGB_mat, (0, l), (w, l), (0, 255, 0), 1)
    cv2.imwrite(text_lines_filename, text_lines_img_RGB_mat)

    # Use Tesseract OCR to read text per line
    preprocessed_img_mat = np.array(preprocessed_img)
    
    ocr_psm = 3
    line_boundary_offset = 0
    cropped_lines = []
    
    blank_black_RGB = np.zeros((50, w, 3), dtype = np.uint8)
    blank_white_L = np.ones((10, w), dtype = np.uint8)*255

    print("########## LOG ##########")
    for u, l in zip(upper_line_bounds, lower_line_bounds):
        cropped_line = preprocessed_img_mat[u-line_boundary_offset:l+line_boundary_offset,]
        cropped_line = np.concatenate((blank_white_L, cropped_line, blank_white_L), dtype=np.uint8)
        
        #####
        
        thresh1 = cv2.threshold(cropped_line, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        line_data2 = (cv2.cvtColor(cropped_line,cv2.COLOR_GRAY2RGB)).copy()
        for cnt in contours:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)

            # Cropping the text block for giving input to OCR
            cropped = cropped_line[y1:y1 + h1, x1:x1 + w1]
            
            # Apply OCR on the cropped image
            line_text = pytesseract.image_to_string(cropped, config=f"--psm {ocr_psm}")
            # line_data = showOCRData(cv2.cvtColor(cropped_line,cv2.COLOR_GRAY2RGB), ocr_psm)
            
            cv2.rectangle(line_data2, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            
            text = "".join([c if ord(c) < 128 else "?" for c in line_text]).strip()
            cv2.putText(line_data2, text, (x1+3, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
            
            print(line_text[:-1])
        
        line_data = showOCRData(cv2.cvtColor(cropped_line,cv2.COLOR_GRAY2RGB), ocr_psm)
        line_data = line_data2
        cropped_lines.append(line_data)
        cropped_lines.append(blank_black_RGB)
        
        #####
        
        # line_text = pytesseract.image_to_string(cropped_line, config=f"--psm {ocr_psm}")
        # line_data = showOCRData(cv2.cvtColor(cropped_line,cv2.COLOR_GRAY2RGB), ocr_psm)
        
        # cropped_lines.append(line_data)
        # cropped_lines.append(blank_black_RGB)
        
        # print(line_text[:-1])
    
    cropped_lines_img = np.vstack(cropped_lines)
    cv2.imwrite(results_filename, cropped_lines_img)
    print("#########################")
    
    return
    
if __name__ == "__main__":
    main()
