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
import re

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
    
    thresh_img_mat = cv2.threshold(np.array(img), 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thresh_img_mat)

def invertImage(img:Image.Image) -> Image.Image:
    """ Inverts all colors in an image. Returns an Image type. """
    return Image.fromarray(cv2.bitwise_not(np.array(img)))

def getImageHistogram(img:Image.Image) -> np.ndarray:
    """ Generates a histogram from an image. Returns an ndarray type. """
    return cv2.reduce(np.array(img), 1, cv2.REDUCE_AVG).reshape(-1)

def getImageContours(img:cv2.Mat) -> tuple:
    """ Generate contours of an image. Returns a tuple type. """
    threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(threshold_img, rectangular_kernel, iterations = 1)
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    return contours

def fixNumTypos(string:str) -> str:
    """ Fixes known numerical typos in string """
    correction_dict = {"0": "O", "1":"IL", "5":"?"}
    string = string.upper()
    
    for correction in correction_dict:
        for typo in correction_dict[correction]:
            string = string.replace(typo, correction)
    return string

############################################
##### MAIN #################################
############################################

def main() -> None:
    version = "0.3"
    print(f"Version {version}\n")
    
    # Set-up input image
    document_filename = "Sample_Document_p3.png"
    preprocessed_filename = "img_preproc.png"
    text_lines_filename = "img_textlines.png"
    results_filename = "img_result.png"
    csv_filename = "ocr_output.csv"
    
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

    # Get histogram and get page-wide lines containing text
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
    
    ocr_config = "--psm 7"
    # ocr_config = "--psm 7 -c tessedit_char_whitelist=abcdefghijklmopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ(\'\"\"\').,:;?!%&123456789+-*/="
    
    line_vertical_boundary_offset = 3
    lines_stack = []
    
    divider_RGB = np.zeros((50, w, 3), dtype = np.uint8)
    padding_L = np.ones((10, w), dtype = np.uint8)*255
    stack_header = np.zeros((50, w, 3), dtype = np.uint8)
    cv2.putText(
        stack_header, f"VERSION {version}", 
        (5, stack_header.shape[0]//2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, (255, 255, 255), 2)
    lines_stack.append(stack_header)

    recognized_text = ""
    word_delimiter = "<!word_end!>"
    print("########## LOG ##########")
    for u, l in zip(upper_line_bounds, lower_line_bounds):        
        # Crop the portion of the image containing the line, then pad the top and bottom with white spaces
        line_img = preprocessed_img_mat[u - line_vertical_boundary_offset : l + line_vertical_boundary_offset,]
        line_img = np.concatenate((padding_L, line_img, padding_L), dtype=np.uint8)
        
        # Create contours to detect potential text
        contours = getImageContours(line_img)[::-1]
        
        line_data = (cv2.cvtColor(line_img,cv2.COLOR_GRAY2RGB)).copy()
        for cont in contours:
            x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(cont)

            # Crop the text block and recognize text in the block
            line_section = line_img[y_cont : y_cont + h_cont, x_cont : x_cont + w_cont]            
            line_text = pytesseract.image_to_string(line_section, config=f"{ocr_config}")
            
            # Draw a rectangle marking the section and the recognized text (marked "?" if not ASCII) from the section
            cv2.rectangle(
                line_data, 
                (x_cont, y_cont), (x_cont + w_cont, y_cont + h_cont), 
                (0, 255, 0), 2)
            line_text = "".join([c if ord(c) < 128 else "?" for c in line_text]).strip()
            cv2.putText(
                line_data, line_text, 
                (x_cont + 3, y_cont+10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.35, (200, 0, 0), 1)
            
            recognized_text += line_text.upper() + word_delimiter
            print(line_text)

        recognized_text += "\n"
        lines_stack.append(line_data)
        lines_stack.append(divider_RGB)

    cropped_lines_img = np.vstack(lines_stack)
    cv2.imwrite(results_filename, cropped_lines_img)
    
    with open("recognized_text.txt", "w") as out_file:
        out_file.write(recognized_text.replace(word_delimiter, " "))
    print("#########################")
    
    with open(csv_filename, "w") as csv_file:
        recognized_text = [i.split(word_delimiter) for i in recognized_text.split("\n")]
        td_keywords = {"LINE":-1, "BEARING":-1, "DISTANCE":-1}
        num_of_cols = -1
        
        csv_file.write("LINE,BEARING,DISTANCE,FORMULA\n")
        for line in recognized_text:
            if all(np.array([*td_keywords.values()]) != -1) and len(line) == num_of_cols:
                _bearing_raw = fixNumTypos(line[td_keywords['BEARING']])
                _bearing_split = [re.search('^[NS\$]', _bearing_raw), re.search('\d+[^-]*-', _bearing_raw), re.search('-[^-]*\d+', _bearing_raw), re.search('[WE]$', _bearing_raw)]
                if not all(_bearing_split):
                    break
                    
                ns_f, deg_f, min_f, ew_f = [*map(lambda m: m.group(), _bearing_split)]
                ns_f = "S" if "$" in ns_f else ns_f
                deg_f = re.search('\d+', deg_f).group()
                min_f = re.search('\d+', min_f).group()
                
                _line = fixNumTypos(line[td_keywords['LINE']])
                _distance = fixNumTypos(line[td_keywords['DISTANCE']])
                _distance = re.search('^([0-9]|\.|,)+', _distance).group()
                _bearing = f"{ns_f}{deg_f}-{min_f}{ew_f}"
                
                _formula = f"@{_distance}<{ns_f}{deg_f}d{min_f}'{ew_f}"
                
                csv_file.write(f"{_line},{_bearing},{_distance},{_formula}\n")
    
            elif all([i in line for i in td_keywords]):
                td_keywords.update(zip(td_keywords, [line.index(i) for i in td_keywords]))
                num_of_cols = len(line)            
    return
    
if __name__ == "__main__":
    main()
