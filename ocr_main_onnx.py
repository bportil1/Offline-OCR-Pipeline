import os
import datetime
import time
import numpy as np
import math
import cv2
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, utils
import pickle
import string 

import multiprocessing

from scipy import ndimage as ndi

from skimage.filters import gabor_kernel

import ultralytics 

from optimum.onnxruntime import ORTModelForVision2Seq

from scipy.signal import find_peaks

from onnxruntime import InferenceSession

from onnxruntime import SessionOptions

from onnxruntime import ExecutionMode

def warn(*args, **kwargs):
    """
    Warning Suppression

    :return: nothing
    """
    pass

### Call to skip warning function directly above and further warning suppression ###
import warnings

warnings.warn = warn
utils.logging.set_verbosity_error()
###

### Global to facilitate progress bar for parallel processing of images
count = 0

### The device with which to run image recognition functions on
device = 'cpu'

class ocr():
    def __init__(self, load_ocr_models=True, load_field_classifier=False):
        
        """
        An OCR object to hold and load the necessary machine learning models.

        :param load_ocr_models: Load all necessary ML models except field_classifier which is not needed 
                                until the last part of the process
        :param load_field_classifier: Load field_classifier only 
        """

        self.processor_typed = ''
        self.model_typed = ''
        self.processor_hw = ''
        self.model_hw = ''
        self.writing_classifier = ''
        self.field_classifier = ''
        self.detector = ''

        if load_ocr_models:
            self.load_ocr_models()
        if load_field_classifier:
            self.load_field_classifier()

    def load_ocr_models(self):
        """
        Function to load text detection, text recognition, and text classification models

        :return: Text detection, text recognition, and classification models
        """
        print("Loading Models")

        #self.writing_classifier = pickle.load(open("./ml_models/text_type_classifiers/etc_model.sav", 'rb'))
        self.writing_classifier = InferenceSession("./ml_models/text_type_classifiers/tt_onnx_etc_model.onnx" \
                        , providers=["CPUExecutionProvider"])
        
        print("Loading text detector")

        self.detector = ultralytics.YOLO('./ml_models/yolotextdet.pt').to(device)

        print("Loading trocr models")

        self.processor_typed = TrOCRProcessor.from_pretrained('./ml_models/ocr_models/typed_ocr_models')
        self.model_typed = VisionEncoderDecoderModel.from_pretrained(
            './ml_models/ocr_models/typed_ocr_models'
        ).to(device)

        '''
        self.processor_typed = TrOCRProcessor.from_pretrained('e:/models--microsoft--trocr-large-str/snapshots/1105e441eaa192e99f3991d2958134348c623f4f')
        self.model_typed = VisionEncoderDecoderModel.from_pretrained(
            'e:/models--microsoft--trocr-large-str/snapshots/1105e441eaa192e99f3991d2958134348c623f4f'
        ).to(device)
        '''

        self.processor_hw = TrOCRProcessor.from_pretrained('./ml_models/ocr_models/hw_ocr_models')
        self.model_hw = VisionEncoderDecoderModel.from_pretrained(
            './ml_models/ocr_models/hw_ocr_models'
        ).to(device)
        '''
        self.processor_hw = TrOCRProcessor.from_pretrained('./ml_models/ocr_models/hw_ocr_models_2')
        self.model_hw = VisionEncoderDecoderModel.from_pretrained(
            './ml_models/ocr_models/hw_ocr_models_2'
        ).to(device)
        '''
        '''
        self.processor_typed = TrOCRProcessor.from_pretrained('onnxtyped')

        self.model_typed = ORTModelForVision2Seq.from_pretrained("onnxtyped").to(device)
        
        self.processor_hw = TrOCRProcessor.from_pretrained('onnxhw')

        self.model_hw = ORTModelForVision2Seq.from_pretrained("onnxhw").to(device)
        '''
        #ORTModelForVision2Seq.from_pretrained('microsoft/trocr-large-str', export=True).save_pretrained("onnxstr")

        
        print("Successfully Loaded Models")

    def load_field_classifier(self):
        """
        Function to load field classification model

        :return: Text detection, text recognition, and classification models
        """
        #self.field_classifier = pickle.load(open("./ml_models/label_classifiers/etc_model.sav", 'rb'))
        self.field_classifier = InferenceSession("./ml_models/label_classifiers/field_onnx_etc_model.onnx" \
                        , providers=["CPUExecutionProvider"])

def dir_validation(dir):
    """
    Function to check if the passed directory has the right format

    :param dir: Directory to check
    :return: error code
    """

    supported_file_types = [ 'bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'avif',
                            'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'pfm', 'sr', 'ras', 'tiff', 'tif',
                            'exr', 'hdr', 'pic'
                            ]
    
    if (len(dir) % 2) == 1:
        return 201
    for file in dir:
        ext = file.split(".")[-1]
        if ext not in supported_file_types:
            return 202
    return 200

def hconcat_resize(img_list, interpolation = cv2.INTER_CUBIC): 
    """
    Function to resize images and concatenate images horizontally

    :param img_list: list of images to perform operations on
    :param interpolation: interpolation argument for resize function from cv2 library

    :return: Concatenated image
    """

    h_min = min(img.shape[0]  
                for img in img_list) 
    
    im_list_resize = [cv2.resize(img, 
                                 (int(img.shape[1] * h_min / img.shape[0])
                                  if (img.shape[1] * h_min / img.shape[0]) > 1
                                  else img.shape[1], 
                                  h_min),
                                 interpolation = interpolation)  
                      for img in img_list] 

    return cv2.hconcat(im_list_resize) 

def most_frequent(lst):
    """
    Function to find the most common element in a list

    :param lst: List with elements
    :return: most common element
    """

    unique, counts = np.unique(lst, return_counts=True)
    index = np.argmax(counts)
    return unique[index]

def img_recognition(img_dir):
    """
    Function to perform the bulk of the text recognition tasks

    :param img_dir: list of image paths
    :param total_images: Number of images to be processed
    :param progress_signal: Signal for GUI process bar
    :return: Dictionary with extracted data as {image path: extracted text}
    """
    global count

    extractions = {}
    
    datapath1 = "extracted_data/label_data.csv" 
    datapath2 = "extracted_data/type_timings.csv"
    datapath3 = "extracted_data/recognition_timings.csv"
    datapath4 = "extracted_data/detection_timings.csv"

    time_file2 = open(datapath2, 'a')
    time_file3 = open(datapath3, 'a')
    time_file4 = open(datapath4, 'a')

    for img_path in img_dir:

        print("Beggining extraction  on image: ", img_path)

        #img = keras_ocr_tools.read(img_path)

        ext_txt = ""

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection_timing_start = time.time()
        pred = text_detection(img_path)
        detection_timing_end = time.time() - detection_timing_start
        time_file4.write(str(detection_timing_end) + "\n")

        #idx = 0
        #out_p = 'C:/Users/Karkaras/Desktop/new_extracts'

        pred = get_distance(pred)
        pred = sorted(pred, key=lambda x:x['distance_y'])
        pred = list(distinguish_groups(pred))
        bar_img = cv2.imread("barsepimg.png")
        for group in pred:
            group = list(distinguish_rows(group))
            #print(group)
            image_crops = []
            crop_labels = []
            for row in group:
                row = sorted(row, key=lambda x:x['distance_x'])
                for box in row:
                    uby = int(round(box['top_left_y'])) if int(round(box['top_left_y'])) >= 0 else 0
                    lby = int(round(box['bottom_right_y'])) if int(round(box['bottom_right_y'])) >= 0 else 0
                    ubx = int(round(box['top_left_x'])) if int(round(box['top_left_x'])) >= 0 else 0
                    lbx = int(round(box['bottom_right_x'])) if int(round(box['bottom_right_x'])) >= 0 else 0
                    if ubx >= lbx:
                        ubx = ubx - (ubx - lbx + 1)
                    if uby >= lby:
                        uby = lby - (uby - lby + 1)
                    cropped_img = img[uby:lby, ubx:lbx]

                    image_crops.append(cropped_img)
                    image_crops.append(bar_img)
                    type_timing_start = time.time()
                    crop_labels.append(text_classification(cropped_img))
                    type_timing_end = time.time() - type_timing_start
                    time_file2.write(str(type_timing_end) + "\n")

                    '''
                    name = os.path.join(out_p, os.path.basename(img_path)[:-4])
                    name = name + str(idx) + ".png"
                    cv2.imwrite(name, cropped_img)
                    idx+=1
                    '''

            label = most_frequent(crop_labels)

            cropped_img = hconcat_resize(image_crops)

            recognition_timing_start = time.time()

            if label in [ 'typed', 'cover' ]:
                ext_txt = ext_txt + " " + text_recognition(
                    cropped_img,
                    label
                )
            elif label == 'handwritten':
                ext_txt = ext_txt + " " + text_recognition(
                    cropped_img,
                    label
                )

            recognition_timing_end = time.time() - recognition_timing_start
            time_file3.write(str(recognition_timing_end) + "\n")


            new_df = pd.DataFrame({ "label":label, "file": img_path}, index = [0])
            new_df.to_csv(datapath1, mode='a', index=False, header=False)
            #ext_txt = ''.join('' if c in punctuation else c for c in ext_txt)
            #ext_txt = ' '.join(ext_txt.split())

            #name = os.path.join(out_p, os.path.basename(img_path)[:-4])
            #name = name + str(idx) + ".png"
            #cv2.imwrite(name, cropped_img)
            #idx+=1

        

        extractions[img_path] = ext_txt + " "
        print("Completed extraction on image: ", img_path)

        #count += 1

    return extractions

def text_recognition(image, label):
    """
    Calls transformers functions to perform text recognition from image.

    :param image: Image with text.
    :param label: Label with type of data to be parsed
    :return: The extracted text.
    """

    if label in [ 'typed', 'cover' ]:
        pixel_values = ocr_obj.processor_typed(image, return_tensors='pt').pixel_values.to(device)
        generated_ids = ocr_obj.model_typed.generate(pixel_values)
        generated_text = ocr_obj.processor_typed.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        pixel_values = ocr_obj.processor_hw(image, return_tensors='pt').pixel_values.to(device)
        generated_ids = ocr_obj.model_hw.generate(pixel_values)
        generated_text = ocr_obj.processor_hw.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def text_detection(img):
    """
    Calls detector model to perform text detection on the passed image

    :param img: Image with text.
    :return: List of boxes for the detected text
    """

    results = ocr_obj.detector.predict(source = img, imgsz=768, conf = .07,
                       iou = .12, augment = True, max_det = 1000, agnostic_nms = True)
    
    return [ box.numpy().boxes.xyxy.tolist() for box in results]

def distinguish_groups(lst):
    """
    Parses returned bounding boxes from objected detected function by text sections
    :param lst: List of bounding boxes
    :return: Sublists containing the bounding boxes grouped by text sections
    """
    sublists = []
    for i in range(0, len(lst)):
        added = False
        if i == 0:
            sublists.append([lst[i]])
            continue
        for j in range(0, len(sublists)):
            if added == True:
                break
            for k in range(0, len(sublists[j])):
                if added == True:
                    break
                if abs(sublists[j][k]['dist_from_origin'] - lst[i]['dist_from_origin']) <= \
                      (sublists[j][k]['height'] + lst[i]['height']) / 2: 
                    sublists[j].append(lst[i])
                    added = True

        if added == False:
            sublists.append([lst[i]])
    return sublists

def distinguish_rows(lst):
    """
    Parses returned bounding boxes from objected detected function by rows
    :param lst: List of bounding boxes
    :return: Sublists containing the bounding boxes grouped by rows
    """
    sublists = []
    if len(lst) == 1:
        sublists.append(lst[0])
    for i in range(0, len(lst)-1):
        if abs(lst[i+1]['distance_y'] - lst[i]['distance_y']) <=  \
           (lst[i]['height'] + lst[i+1]['height']) / 4: 
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            if i == 0:
                sublists.append(lst[i])        
            yield sublists
            sublists = [lst[i+1]]
    yield sublists
    
def get_distance(preds):
    """
    Gathers measurements for a list of bounding boxes for further processing

    :param preds: List of bounding boxes
    :return: List of dictionary elements built as {bounding box:data}
    """
    detections = []
    x0, y0 = 0, 0
    idx = 0
    for group in preds[0]:
        top_left_x = group[0]
        top_left_y = group[1] 
        bottom_right_x = group[2]
        bottom_right_y = group[3]
        center_x = (top_left_x + bottom_right_x)/2
        center_y = (top_left_y + bottom_right_y)/2
        dist_from_origin = math.dist([x0,y0], [.2*center_x, 1.8*center_y])
        distance_y = center_y - y0
        distance_x = center_x - x0
        height = abs(top_left_y - bottom_right_y)
        detections.append({
            'top_left_x': top_left_x,
            'top_left_y': top_left_y,
            'bottom_right_x': bottom_right_x,
            'bottom_right_y': bottom_right_y,
            'dist_from_origin': dist_from_origin,
            'distance_y': distance_y,
            'distance_x': distance_x,
            'height': height
            })
        idx = idx + 1
    return detections


def longestZeroSeqLength(a):
    """
    Function to find the longest sequence of zeros within a list

    :param a: List with numeric elements
    :return: Length of longest sequence of zeros
    """

    chg = np.abs(np.diff(np.equal(a, 0).view(np.int8), prepend=[0], append=[0]))
    rng = np.where(chg == 1)[0]
    if rng.size == 0:
        return 0    
    rng = rng.reshape(-1, 2)
    return np.subtract(rng[:,1], rng[:,0]).max()

def get_gray_img_features(img_num, th):
    """
    Function to collect features from grayscaled images for handwritten/printed text classification

    :param img_num: Grayscaled image to be processed
    :param th: Threshold value given by Otsu's Binarization method
    :return: Grayscale image features
    """

    img_height, img_width = np.array(img_num).shape
    counts, bins = np.histogram(img_num, range(257))    
    peaks = find_peaks(counts)
    maxima_count = len(peaks[0])
    c = 255 / (np.log(1 + np.max(img_num)))
    log_transformed = c * np.log(1 + img_num)
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    log_transformed_mean = log_transformed.mean()
    log_transformed_var = log_transformed.var()
    log_transformed_std = log_transformed.std()   
    max_int = img_num.max()
    min_int = img_num.min()
    quart_range = (max_int - min_int)/4
    upper_quart_count = np.count_nonzero(img_num >= (max_int - quart_range))
    upper_quart_perc = (upper_quart_count / (img_width * img_height) if (img_width * img_height) > 0 else 1)*100
    lower_quart_count = np.count_nonzero(img_num <= (min_int + quart_range))
    lower_quart_perc = (lower_quart_count / (img_width * img_height) if (img_width * img_height) > 0 else 1)*100

    return [log_transformed_mean, th, maxima_count,
            upper_quart_perc, lower_quart_perc, 
            log_transformed_std, log_transformed_var ]

def get_gabor_features(img_num):
    """
    Function to collect Gabor Filter features for handwritten/printed text classification

    :param img_num: Grayscaled image to be processed
    :return: Gabor filter features
    """
    #out_list_mean = []
    out_list_var = []

    kernels = []

    for theta in range(1,17):
        theta = theta * (180 / 16) 
        kernel = np.real(gabor_kernel(0.05, theta=theta, sigma_x=1, sigma_y=1))
        kernels.append(kernel)

    #out_list_mean_local = []
    out_list_var_local = []

    for kernel in kernels:
        filtered = ndi.convolve(img_num, kernel, mode='wrap')
        #out_list_mean_local.append(filtered.mean())
        out_list_var_local.append(filtered.var())
    
    for idx in range(len(out_list_var_local)):
        #out_list_mean.append(out_list_mean_local[idx])
        out_list_var.append(out_list_var_local[idx])

    return out_list_var

def get_horizontal_projection_features(th_img):
    """
    Function to collect horizontal projection features for handwritten/printed text classification

    :param th_img: Binarized image to be processed
    :return: Horizontal projection features
    """

    swap_counts = []
    run_lengths = []

    for row in th_img:
        run_lengths.append(longestZeroSeqLength(row))
        swap_counts.append((np.diff(row)!=0).sum())

    hist, bin_edges = np.histogram(run_lengths, bins=100, density = True)

    max_run = hist.argmax()

    row_max_run_count = hist[max_run]
    normed_row_max = bin_edges[max_run]
    row_hist_mean = hist.mean()
    row_hist_var = hist.var()
    row_hist_std = hist.std()

    hist, bin_edges = np.histogram(swap_counts, bins=100, density = True)

    max_run = hist.argmax()

    sc_row_max_run_count = hist[max_run]
    normed_sc_row_max = bin_edges[max_run]
    sc_row_hist_mean = hist.mean()
    sc_row_hist_var = hist.var()
    sc_row_hist_std = hist.std()

    return row_max_run_count, normed_row_max, row_hist_mean, row_hist_var, row_hist_std, \
           sc_row_max_run_count, normed_sc_row_max, sc_row_hist_mean, sc_row_hist_var, sc_row_hist_std

def get_vertical_projection_features(th_img):
    """
    Function to collect vertical projection features for handwritten/printed text classification

    :param th_img: Binarized image to be processed
    :return: Vertical projection features
    """

    swap_counts = []
    run_lengths = []
    for col in th_img.T:
        run_lengths.append(longestZeroSeqLength(col))
        swap_counts.append((np.diff(col)!=0).sum())

    hist, bin_edges = np.histogram(run_lengths, bins=100, density = True)

    max_run = hist.argmax()

    col_max_run_count = hist[max_run]
    normed_col_max = bin_edges[max_run]
    col_hist_mean = hist.mean()
    col_hist_var = hist.var()
    col_hist_std = hist.std()

    hist, bin_edges = np.histogram(swap_counts, bins=100, density = True)

    max_run = hist.argmax()

    sc_col_max_run_count = hist[max_run]
    normed_sc_col_max = bin_edges[max_run]

    sc_col_hist_mean = hist.mean()
    sc_col_hist_var = hist.var()
    sc_col_hist_std = hist.std()

    return col_max_run_count, normed_col_max, col_hist_mean, col_hist_var, col_hist_std, \
           sc_col_max_run_count, normed_sc_col_max, sc_col_hist_mean, sc_col_hist_var, sc_col_hist_std


def get_315_deg_projection_features(th_img):
    """
    Function to collect features from projections drawn at 315 degrees
    accross the given image for handwritten/printed text classification

    :param th_img: Binarized image to be processed
    :return: 315 degree projection features
    """
    swap_counts = []
    run_lengths = []
    curr_row = []
    idx = 0
    
    row_c, col_c, *_ = th_img.shape
    
    for row in range(row_c-1,-1,-1):
        if row == row_c-1:
            run_lengths.append(longestZeroSeqLength([th_img[row][0]]))
        for row_idx in range(row, row_c):
            if idx >= col_c:
                break
            curr_row.append(th_img[row_idx][idx])
            idx += 1
        swap_counts.append((np.diff(curr_row)!=0).sum())
        run_lengths.append(longestZeroSeqLength(curr_row))
        idx = 0
        curr_row = []

    hist, bin_edges = np.histogram(run_lengths, bins=100, density = True)

    max_run = hist.argmax()

    deg_315_max_run_count = hist[max_run]
    normed_deg_315_max = bin_edges[max_run]
    deg_315_hist_mean = hist.mean()
    deg_315_hist_var = hist.var()
    deg_315_hist_std = hist.std()

    hist, bin_edges = np.histogram(swap_counts, bins=100, density = True)

    max_run = hist.argmax()

    sc_deg_315_max_run_count = hist[max_run]
    normed_sc_deg_315_max = bin_edges[max_run]

    sc_deg_315_hist_mean = hist.mean()
    sc_deg_315_hist_var = hist.var()
    sc_deg_315_hist_std = hist.std()

    return deg_315_max_run_count, normed_deg_315_max, deg_315_hist_mean, deg_315_hist_var, deg_315_hist_std, \
           sc_deg_315_max_run_count, normed_sc_deg_315_max, sc_deg_315_hist_mean, sc_deg_315_hist_var, sc_deg_315_hist_std

def get_225_deg_projection_features(th_img):
    """
    Function to collect features from projections drawn at 225 degrees
    accross the given image for handwritten/printed text classification

    :param th_img: Binarized image to be processed
    :return: 225 degree projection features
    """

    swap_counts = []
    run_lengths = []
    curr_row = []

    row_c, col_c, *_ = th_img.shape

    idx = col_c - 1

    for row in range(row_c-1,-1,-1):
        if row == row_c-1:
            run_lengths.append(longestZeroSeqLength([th_img[row][col_c - 1]]))
        for row_idx in range(row, row_c):
            if idx >= col_c or idx < 0:
                break
            curr_row.append(th_img[row_idx][idx])
            idx -= 1
        swap_counts.append((np.diff(curr_row)!=0).sum())
        run_lengths.append(longestZeroSeqLength(curr_row))
        idx = col_c - 1
        curr_row = []

    hist, bin_edges = np.histogram(run_lengths, bins=100, density = True) 

    max_run = hist.argmax()

    deg_225_max_run_count = hist[max_run]
    normed_deg_225_max = bin_edges[max_run]
    deg_225_hist_mean = hist.mean()
    deg_225_hist_var = hist.var()
    deg_225_hist_std = hist.std()

    hist, bin_edges = np.histogram(swap_counts, bins=100, density = True)

    max_run = hist.argmax()

    sc_deg_225_max_run_count = hist[max_run]
    normed_sc_deg_225_max = bin_edges[max_run]
    sc_deg_225_hist_mean = hist.mean()
    sc_deg_225_hist_var = hist.var()
    sc_deg_225_hist_std = hist.std()    

    return deg_225_max_run_count, normed_deg_225_max, deg_225_hist_mean, deg_225_hist_var, deg_225_hist_std, \
           sc_deg_225_max_run_count, normed_sc_deg_225_max, sc_deg_225_hist_mean, sc_deg_225_hist_var, \
           sc_deg_225_hist_std

def text_classification(img):
    """
    Function to classify text within an image as handwritten or typed

    :param img: Image with text
    :return: Label from resulting classification
    """
    
    img_num = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, th_img = cv2.threshold(img_num, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out_list = get_gray_img_features(img_num, th)
    out_list_var = get_gabor_features(img_num)

    #### Horizontal Projections

    row_max_run_count, normed_row_max, row_hist_mean, row_hist_var, row_hist_std, \
           sc_row_max_run_count, normed_sc_row_max, \
           sc_row_hist_mean, sc_row_hist_var, sc_row_hist_std = get_horizontal_projection_features(th_img)

    #### Vertical Projections
    col_max_run_count, normed_col_max, col_hist_mean, col_hist_var, col_hist_std, \
           sc_col_max_run_count, normed_sc_col_max, \
           sc_col_hist_mean, sc_col_hist_var, sc_col_hist_std = get_vertical_projection_features(th_img)
        
    #### 315 deg projections

    deg_315_max_run_count, normed_deg_315_max, deg_315_hist_mean, deg_315_hist_var, deg_315_hist_std, \
           sc_deg_315_max_run_count, normed_sc_deg_315_max, \
           sc_deg_315_hist_mean, sc_deg_315_hist_var, \
           sc_deg_315_hist_std = get_315_deg_projection_features(th_img)


    #### 225 deg projections

    deg_225_max_run_count, normed_deg_225_max, deg_225_hist_mean, deg_225_hist_var, deg_225_hist_std, \
           sc_deg_225_max_run_count, normed_sc_deg_225_max, sc_deg_225_hist_mean, sc_deg_225_hist_var, \
           sc_deg_225_hist_std = get_225_deg_projection_features(th_img)

    out_list.extend([
        row_max_run_count, normed_row_max,
        row_hist_mean, row_hist_var, row_hist_std,
        col_max_run_count, normed_col_max,
        col_hist_mean, col_hist_var, col_hist_std,
        deg_315_max_run_count, normed_deg_315_max,
        deg_315_hist_mean, deg_315_hist_var, deg_315_hist_std, 
        deg_225_max_run_count, normed_deg_225_max,
        deg_225_hist_mean, deg_225_hist_var, deg_225_hist_std,
        sc_row_max_run_count, normed_sc_row_max, 
        sc_row_hist_mean, sc_row_hist_var, sc_row_hist_std,
        sc_col_max_run_count, normed_sc_col_max,
        sc_col_hist_mean, sc_col_hist_var, sc_col_hist_std, 
        sc_deg_315_max_run_count, normed_sc_deg_315_max,
        sc_deg_315_hist_mean, sc_deg_315_hist_var, sc_deg_315_hist_std,
        sc_deg_225_max_run_count, normed_sc_deg_225_max,
        sc_deg_225_hist_mean, sc_deg_225_hist_var, sc_deg_225_hist_std,   
    ])
    out_list.extend(out_list_var)
    ext_features = np.reshape(out_list, (1, -1))
    #label = ocr_obj.writing_classifier.predict(ext_features)
    output_name = ocr_obj.writing_classifier.get_inputs()[0].name
    label = ocr_obj.writing_classifier.run(None, {output_name: ext_features.astype(np.float32)})[0]
    return label[0]

def text_feature_extractor(value, ocr_obj):
    """
    Function to take measurements from a text string for classification purposes

    :param value: Text string
    :return: A list of values
    """
    value=str(value)
    text_length = len(value)
    words = value.count(" ") + 1
    numbers = sum(c.isdigit() for c in value)
    #letters = sum(c.isalpha() for c in value)
    punctuation = sum((1 if c in string.punctuation else 0) for c in value)
    if text_length == 0:
        num_text_ratio = 0
        punc_text_ratio = 0
    else:
        num_text_ratio = numbers/text_length
        punc_text_ratio = punctuation/text_length
        
    avg_word_length = sum(len(word) for word in value) / words
    
    ext_features = np.reshape(  [text_length, words, num_text_ratio,
                                 avg_word_length, punc_text_ratio] , (1, -1))
    
    #field = ocr_obj.field_classifier.predict(ext_features)
    output_name = ocr_obj.field_classifier.get_inputs()[0].name
    field = ocr_obj.field_classifier.run(None, {output_name: ext_features.astype(np.float32)})[0]
    return field[0]

def pub_year_extraction(data):
    """
    Parse extracted text for phrase that could be a year

    :param data: Text string to be parsed for a value representing a year
    :return: Value to be used as year
    """
    pub_year = ""
    for phrase in data.split():
        if phrase.isdigit() and (1600 <= int(phrase) <= datetime.datetime.today().year+1):
            pub_year = phrase
    return pub_year

def merge_dicts(data):
    """
    Function to merge a list of dictionaries into one

    :param data: List of dictionaries
    :return: Dictionary
    """
    merged_dict = data[0]
    for idx in range(len(data)):
        merged_dict.update(data[idx])
    return merged_dict

def write_dataframe(data):
    """
    Function to write extracted data out to a csv

    :param data: Dictionary containing data as {image path: extracted data}
    :param label: Path of directory
    :return: Path of csv containing the data
    """

    print("Writing Out Data to CSV")
    ocr_obj = ocr(load_ocr_models=False, load_field_classifier=True) 
    init_datapath = './extracted_data'
    datapath = "extracted_data/extracted_data.csv"

    label_out_pth = init_datapath + "/field_label_data.csv"
    label_out_file = open(label_out_pth, 'a')

    os.makedirs(init_datapath, exist_ok=True)
    data = merge_dicts(data)
    output_data = pd.DataFrame(columns=['ID', 'Title', 'SuDoc', 'Publication Year', 
                                        'Path','Error Code','Query Status',
                                        'Sudoc Image', 'Title Image', 
                                        'Image 1 Path', 'Image 2 Path',
                                        'Image 1 Ext', 'Image 2 Ext',])
    #output_data = pd.DataFrame(columns=['extract', 'text_type'])
    title_key = sudoc_key = text_type_1_val = text_type_2_val = text_type_1_key = text_type_2_key = pub_year = ""
    for idx, key in enumerate(data):
        text_type = text_feature_extractor(data[key], ocr_obj)
        if text_type == 'title':
            text_type_1_key = 'Title'
            text_type_1_val = data[key]
            title_key = key
        elif text_type == 'sudoc':
            text_type_2_key = 'SuDoc'
            pub_year = pub_year_extraction(data[key])
            #data[key] = data[key] #.replace(" ", "")
            #if data[key][:4].lower() == 'docs':
            #    data[key] = data[key][4:]
            text_type_2_val = data[key]
            sudoc_key = key

        if (idx % 2) == 1:
            img_2_pth = key
            img_2_ext = data[key]
            output_data = pd.concat([output_data, pd.DataFrame(
                [{'ID': int((idx - 1) / 2),
                   text_type_1_key: text_type_1_val, 
                   text_type_2_key: text_type_2_val,
                  'Publication Year': pub_year, 
                  'Sudoc Image': sudoc_key, 
                  'Title Image': title_key,
                  'Image 1 Path': img_1_pth,
                  'Image 2 Path': img_2_pth,
                  'Image 1 Ext': img_1_ext,
                  'Image 2 Ext': img_2_ext,
                  }])],
                ignore_index=True)
        else:
            img_1_pth = key
            img_1_ext = data[key]

        label_out_file.write( key + "," + data[key] + "," + text_type + "\n")

        title_key = sudoc_key = text_type_1_val = text_type_2_val = text_type_1_key = text_type_2_key = pub_year = ""

    output_data.to_csv(datapath, index=False, mode="a")
    print("Completed Writing Step")
    return datapath


def main():
    print("Beginning Script")
    time_file = open('ml_pipeline_timings.txt', 'a')
    dirs = [   
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/whole hw test set',
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/whole typed test set',
        #'C:/Users/Karkaras/Desktop/new hw'
        #'C:/Users/Karkaras/Desktop/new cover'
        #'C:/Users/Karkaras/Desktop/new typed'
        #'C:/Users/Karkaras/Desktop/new_extracts'
        #'C:/Users/Karkaras/Desktop/typescript printed'

        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/test_set/test_hw_sudoc',
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/test_set/test_title',
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/test_set/test_typed_sudoc',
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/test_set/hw_printed_test/hw',
        #'C:/Users/Karkaras/Desktop/proc_sample_imgs/test_set/hw_printed_test/printed_title_page'

        'C:/Users/Karkaras/Desktop/proc_sample_imgs/hw_class',
        'C:/Users/Karkaras/Desktop/proc_sample_imgs/typed_class',
        'C:/Users/Karkaras/Desktop/proc_sample_imgs/title_class'

        #'C:/Users/Karkaras/Desktop/cover_imgs/title30cat/224x224'
        #'E:/test_set/hw_test',
        #'E:/test_set/cover_test',
        #'E:/test_set/typed_test'
        

    ]
        
    processed_file_count = 0
    start_time = time.time()
    with multiprocessing.Pool(processes=2) as pool:
        for path in dirs:
            extracted_data = []
            collected_data_proc_1 = []
            collected_data_proc_2 = []
            img_dir = os.listdir(path)
            img_dir = [os.path.join(path, img) for img in img_dir]

            #img_dir.sort(key=lambda x: os.path.getctime(x))

            # images_coll = [ keras_ocr.tools.read(img) for img in img_dir ]

            # extracted_data = par_img_proc_caller(images_coll)

            total_images = len(img_dir)
            processed_file_count += total_images
            #halfpoint = False
            hp = total_images//2
            collected_data_proc_1.append(pool.apply_async(img_recognition, (img_dir[:hp],))) 
            collected_data_proc_2.append(pool.apply_async(img_recognition, (img_dir[hp:],))) 

            for obj in range(len(collected_data_proc_1)):
                extracted_data.append(collected_data_proc_1[obj].get())

            for obj in range(len(collected_data_proc_2)):
                extracted_data.append(collected_data_proc_2[obj].get())

            #if (idx > int(len(img_dir)/2)) and halfpoint == False:
            #    halfpoint = True
            #    write_dataframe(extracted_data)
                        
            write_dataframe(extracted_data)
            

        pool.close()
        pool.join() 

        #validation = self.dir_validation(img_dir)

        #if validation == 200:

        #    extracted_data = self.par_img_proc_caller(img_dir)

        #elif validation == 201:
        #    print("Selected data has odd number of images")
        #    return
        
        #elif validation == 202:
        #    print("Selected data has an invalid file type")
        #    return

    time_file.write("Total images: " + str(processed_file_count) + "\n")

    processing_time = time.time() - start_time
    time_file.write("Image Processing Time: " + str(processing_time) + "\n")
    print("finished image reading lines")

if multiprocessing.current_process().name != 'MainProcess':
    start_time = time.time()
    ocr_obj = ocr(load_ocr_models=True, load_field_classifier=False) 
    load_time = time.time() - start_time
    time_file = open('ml_pipeline_timings.txt', 'a')
    time_file.write("Model Loading Time: " + str(load_time) + "\n")

if __name__ == '__main__':
    main()