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
        15, 11)
    
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

def showOCRData(img:cv2.Mat, ocr_psm:int=3) -> cv2.Mat:
    """ Recognizes text from an image (matrix) and draws what has been read. Returns a Mat type. """
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
    version = "0.3"
    print(f"Version {version}\n")
    
    # Set-up input image
    document_filename = "case1.jpg"
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
    
    ocr_psm_default = 7
    ocr_psm_backup = 3
    ocr_psm = ocr_psm_default
    
    line_boundary_offset = 0
    cropped_lines = []
    
    blank_black_RGB = np.zeros((50, w, 3), dtype = np.uint8)
    blank_white_L = np.ones((10, w), dtype = np.uint8)*255
    
    img_header = np.zeros((50, w, 3), dtype = np.uint8)
    h, w, _ = img_header.shape
    cv2.putText(
        img_header, f"VERSION {version}", 
        (5, h//2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, (255, 255, 255), 2)
    cropped_lines.append(img_header)

    print("########## LOG ##########")
    for u, l in zip(upper_line_bounds, lower_line_bounds):        
        # Crop the portion of the image containing the line, then pad the top and bottom with white spaces
        cropped_line = preprocessed_img_mat[u-line_boundary_offset:l+line_boundary_offset,]
        cropped_line = np.concatenate((blank_white_L, cropped_line, blank_white_L), dtype=np.uint8)
        
        # Create contours to detect potential text
        contours = getImageContours(cropped_line)[::-1]
        
        line_data = (cv2.cvtColor(cropped_line,cv2.COLOR_GRAY2RGB)).copy()
        for cont in contours:
            x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(cont)

            # Cropping the text block for giving input to OCR
            line_section = cropped_line[y_cont : y_cont + h_cont, x_cont : x_cont + w_cont]
            
            # Recognize text, if text is long, use other PSM for the OCR
            used_different_ocr_psm = False
            while (True):
                # Apply OCR on the cropped image
                line_text = pytesseract.image_to_string(line_section, config=f"--psm {ocr_psm}")
                if len(line_text.split(" ")) <= 3 or used_different_ocr_psm:
                    ocr_psm = ocr_psm_default
                    break
    
                used_different_ocr_psm = True
                ocr_psm = ocr_psm_backup
                continue
            
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
            
            print(line_text)

        cropped_lines.append(line_data)
        cropped_lines.append(blank_black_RGB)

    cropped_lines_img = np.vstack(cropped_lines)
    cv2.imwrite(results_filename, cropped_lines_img)
    print("#########################")
    
    return
    
if __name__ == "__main__":
    main()
