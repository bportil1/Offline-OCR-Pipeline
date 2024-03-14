import sklearn as sk
import pandas as pd
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split

import matplotlib as plt

from skimage.transform import radon

from scipy.signal import find_peaks

import cv2

from scipy.ndimage import interpolation as inter
from scipy import ndimage as ndi

from skimage.filters import gabor_kernel

import multiprocessing as mp

def compute_cheap_hash(txt, length=6):
    # This is just a hash for debugging purposes.
    #    It does not need to be unique, just fast and short.
    hash = hashlib.sha1()
    hash.update(txt)
    return hash.hexdigest()[:length]

def make_bitseq(s):
     if not s.isascii():
        '''
        for i in range(0, len(s)):
            try:
                s[i].encode("ascii")
            except:
                #means it's non-ASCII
                s=s[i].replace(s[i], "")
        '''
        s = str(s.encode('ascii',errors='ignore'))
        
     return "".join(f"{ord(i):08b}" for i in s)

def create_synthetic_sudocs(dataframe, targ_len):

    fed_deps =[ 'A', 'AE' , 'B', 'C', 'C 3', 'CC', 'CR', 'CS', 'D', 'E', 'ED', 'EP'
                , 'FA', 'FCA', 'FEM', 'FHF', 'FMC', 'FM', 'FR', 'FT', 'FTZ', 'GA'
                , 'GP', 'GS', 'HE', 'HH', 'HS', 'I', 'I 19', 'ID', 'IC', 'ITC'
                , 'J', 'JU', 'L', 'LC', 'LR', 'MS', 'NA', 'NAS', 'NC', 'NCU'
                , 'NF', 'NMS', 'NS', 'OP', 'P', 'PE', 'PM', 'PR', 'PREX', 'PRVP'
                , 'RR', 'S', 'SBA', 'SE', 'SI', 'SSA', 'T', 'T 22', 'TD', 'TDA'
                , 'VA', 'X', 'Y', 'Y 1', 'Y 4' ] 

    vol_opts = [ 'v.', 'vol.', 'VOL.', 'no.']

    doc_opts = [ 'DOCS', 'Docs', 'docs']

    periodical_opts = ['periodical', 'PERIODICAL']

    coinflip_gen1 = random.random()
    coinflip_gen2 = random.random()
    coinflip_gen3 = random.random()
    coinflip_gen4 = random.random()

    synth_sudocs = []
    
    for idx in range(targ_len):

        synth_sudoc = ""

        coinflip_gen1 = random.random()
        coinflip_gen2 = random.random()
        coinflip_gen3 = random.random()
        coinflip_gen4 = random.random()

        if coinflip_gen1 < .1:
            synth_sudoc = synth_sudoc + str(doc_opts[random.randrange(0, len(doc_opts))]) + " "
        
        synth_sudoc = synth_sudoc + str(fed_deps[random.randrange(0, len(fed_deps))]) + " " + str(random.randrange(1,100)) \
        + '.' + str(random.randrange(1,1000))

        if coinflip_gen2 > .5:
            synth_sudoc = synth_sudoc + "/" + str(random.randrange(1,1000))

        if coinflip_gen3 > .5:
            synth_sudoc = synth_sudoc + " " + str(vol_opts[random.randrange(0,4)]) + " " + \
                                                str(random.randrange(1,1000))

        if coinflip_gen4 < .05:
            synth_sudoc = synth_sudoc + " " + str(periodical_opts[random.randrange(0, len(periodical_opts))])


        #synth_sudoc = ''.join(' ' if c in string.punctuation else c for c in synth_sudoc)

        #synth_sudocs = synth_sudocs + synth_sudoc.split()

        for word in synth_sudoc.split():

            synth_sudocs.append(word)

    dataframe['sudoc'] = synth_sudocs
        
    return dataframe

def parse_data(dataframe):

    split_data = pd.DataFrame(columns=['data', 'label'])
    
    for (column_name, column_data) in dataframe.iteritems():
        for val in column_data:
            #rint(val, " ", column_name)
            split_data = pd.concat([pd.DataFrame([[val, column_name]], columns = split_data.columns), split_data] ,ignore_index=True)
            
    return split_data

def write_dataframe(data_frame, label_frame, data_subset):

    train_data = pd.DataFrame(columns=['data'])

    label_data = pd.DataFrame(columns=['label'])

    for val in data_frame:
        train_data = pd.concat([pd.DataFrame([val], columns = train_data.columns), train_data] ,ignore_index=True)

    for val in label_frame:
        label_data = pd.concat([pd.DataFrame([val], columns = label_data.columns), label_data] ,ignore_index=True) 

    path = "./classifier_data/"

    os.makedirs(path, exist_ok=True)

    datapath = "./classifier_data/" + data_subset + "_data_" + "search_token_data_shuffd.csv"
    labelpath = "./classifier_data/" + data_subset + "_label_" + "search_token_data_shuffd.csv"

    train_data.to_csv(datapath)
    label_data.to_csv(labelpath)
    
    #swap return value here for file paths
    return 0

def data_splits(data):

    #X_train, X_test, y_train, y_test = train_test_split(data[[  'th', 'maxima_count', 'upper_quart_intensity', 'avg_size', 'log_transformed_mean', 'log_transformed_var']], data[['label']], test_size=.01)

    X_train, X_test, y_train, y_test = train_test_split(data, test_size=.01)

    #X_test, X_doc_data, y_test, y_doc_data = train_test_split(X_test, y_test, test_size=.5) 

    #write_dataframe(X_train, y_train, 'train')
    #write_dataframe(X_test, y_test, 'test')
    #write_dataframe(X_doc_data, y_doc_data, 'vocab')
    
    return X_train, X_test, y_train, y_test

def winVar(img, wlen, wheight):
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wheight),
        borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return wsqrmean - wmean*wmean

def winStd(img, wlen, wheight):
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wheight),
        borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return np.sqrt(wsqrmean - wmean*wmean)

np.set_printoptions(threshold=np.inf)

def correct_skew(image, delta=.05, limit=80):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def _init(shared_arr_):
    global shared_arr
    shard_arr = shared_arr_

def shared_to_numpy(shared_arr, dtype, shape):
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)

def create_shared_array(dtype, shape):
    dtype = np.dtype(dtype)
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    shared_arr = mp.RawArray(cdtype, sum(shape))
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr

def parallel_function(index_range):
    i0, i1 = index_range
    arr = shared_to_numpy(*shared_arr)
    aarr[i0:i1] = np.arange(i0,i1)

def longestZeroSeqLength(a):
    # Changes in "isZero" for consecutive elements
    chg = np.abs(np.diff(np.equal(a, 0).view(np.int8), prepend=[0], append=[0]))
    # Ranges of "isZero" elements
    rng = np.where(chg == 1)[0]
    if rng.size == 0: return 0    # All non-zero elements
    rng = rng.reshape(-1, 2)
    # Compute length of each range and return the biggest
    return np.subtract(rng[:,1], rng[:,0]).max()


def create_feature_vector(img, append_arg):
    print(img)

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #im = keras_ocr.tools.read(img)

    #cv2.imshow("before", img)
    #cv2.waitKey(0)
    img_num = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, th_img = cv2.threshold(img_num.astype("uint8"), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #angle, th_img = correct_skew(th_img)

    #cv2.imshow("before", th_img)
    #cv2.waitKey(0)

    img_height, img_width = np.array(img_num).shape

    c = 255/(np.log(1 + np.max(img_num))) 

    log_transformed = c * np.log(1 + img_num) 
  
    log_transformed = np.array(log_transformed, dtype = np.uint8)

    log_transformed_mean = log_transformed.mean()

    log_transformed_var = log_transformed.var()

    log_transformed_std = log_transformed.std()
    
    counts, bins = np.histogram(img_num, range(257))
    
    peaks = find_peaks(counts)

    maxima_count = len(peaks[0])

    max_int = img_num.max()

    min_int = img_num.min()

    quart_range = (max_int - min_int)/4

    upper_quart_count = np.count_nonzero(img_num >= (max_int - quart_range))

    upper_quart_perc = (upper_quart_count / (img_width * img_height) if (img_width * img_height) > 0 else 1)*100

    lower_quart_count = np.count_nonzero(img_num<= (min_int + quart_range))

    lower_quart_perc = (lower_quart_count / (img_width * img_height) if (img_width * img_height) > 0 else 1)*100



    ### K1

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

    #out_list_mean.append(np.asarray(out_list_mean_local).mean())
    #out_list_var.append(np.asarray(out_list_var_local).mean())

    for idx in range(len(out_list_var_local)):
        #out_list_mean.append(out_list_mean_local[idx])
        out_list_var.append(out_list_var_local[idx])

    #### Vertical projections

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

    ##### Horizontal Projections
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

    #### 315 deg projections

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


    #### 225 deg projections

    swap_counts = []        
    run_lengths = []
    curr_row = []
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

    out_list = [ log_transformed_mean, th, maxima_count,
                 upper_quart_perc, lower_quart_perc, 
                 log_transformed_std, log_transformed_var,
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
                 ]

    #out_list.extend(out_sinogram)
    #out_list.extend(out_list_mean)
    out_list.extend(out_list_var)

    out_list.append(append_arg)

    

    return ",".join(map(str, out_list))

    #return [ log_transformed_mean, th, maxima_count,
    #         upper_quart_perc, lower_quart_perc, 
    #         log_transformed_std , data = []
    

    '''
    'out_list_mean_k1', 
    'out_list_mean_k2', 
    'out_list_mean_k3', 
    'out_list_var_k1',
    'out_list_var_k2',
    'out_list_var_k3', 
    '''
def load_data():
        
        cols = ['log_transformed_mean', 'th',
                'maxima_count', 'upper_quart_perc', 'lower_quart_perc', 
                'log_transformed_std', 'log_transformed_var',
                'row_max_run_count', 'normed_row_max',
                'row_hist_mean', 'row_hist_var', 'row_hist_std',
                'col_max_run_count', 'normed_col_max',
                'col_hist_mean', 'col_hist_var', 'col_hist_std',
                'deg_315_max_run_count', 'normed_deg_315_max',
                'deg_315_hist_mean', 'deg_315_hist_var', 'deg_315_hist_std', 
                'deg_225_max_run_count', 'normed_deg_225_max',
                'deg_225_hist_mean', 'deg_225_hist_var', 'deg_225_hist_std',
                'sc_row_max_run_count', 'normed_sc_row_max', 
                'sc_row_hist_mean', 'sc_row_hist_var', 'sc_row_hist_std',
                'sc_col_max_run_count', 'normed_sc_col_max',
                'sc_col_hist_mean', 'sc_col_hist_var', 'sc_col_hist_std', 
                'sc_deg_315_max_run_count', 'normed_sc_deg_315_max',
                'sc_deg_315_hist_mean', 'sc_deg_315_hist_var', 'sc_deg_315_hist_std',
                'sc_deg_225_max_run_count', 'normed_sc_deg_225_max',
                'sc_deg_225_hist_mean', 'sc_deg_225_hist_var', 'sc_deg_225_hist_std'   
                '''
                'ang1','ang30','ang45',
                'ang60','ang90','ang120',
                'ang135','ang150','ang180',
                'ang210','ang225','ang240',
                'ang270','ang300','ang315',
                'ang330','ang360'
                '''
               ]
        
        cols.extend([ x for x in range(16)])
        cols.append('label')

        #cols.extend(cols1)

        typed_data = pd.DataFrame(columns=cols)

        #printed_dir = "c:/Users/Karkaras/Desktop/typed_class_ds" 

        printed_dir = "e:/mc_new/pr"

        #printed_dir = "e:/test_set/typed_test"

        #printed_dir2 = "e:/out78"

        #printed_dir2 = "e:/outCURRINUSE"

        outPath = "e:/hw_classif_data"

        #outPath = "e:/hw_classif_test_data"

        os.makedirs(outPath, exist_ok=True)
    
        img_dir = os.listdir(printed_dir)

        #img_dir2 = os.listdir(printed_dir2)

        typed_img_dir = [ os.path.join(printed_dir, img) for img in img_dir ]

        #typed_img_dir2 = [ os.path.join(printed_dir2, img) for img in img_dir2 ]

        #typed_img_dir.extend(typed_img_dir2)
        
        data = []
        ret_data = []
    
        #for img in typed_img_dir:

        #    ret_data = create_feature_vector(img, 'typed')
                
        #    data.append(ret_data)

        #MAX IS 77270
        upp_bound = len(typed_img_dir)//16
        with mp.Pool(processes=16) as pool:
            for idx in range(0, upp_bound*16, 16):
                print(idx)
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+0)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+1)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+2)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+3)], 'typed')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+4)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+5)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+6)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+7)], 'typed')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+8)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+9)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+10)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+11)], 'typed')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+12)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+13)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+14)], 'typed')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+15)], 'typed')))


                for obj in ret_data:
                    data.append(obj.get().split(','))
                ret_data = []
                 
        pool.close()
        pool.join()
        
        typed_data = pd.DataFrame(data, columns=cols)
        typed_data.to_csv(outPath+"/typed_data.csv", index=False)

        #### ADDED FOR MC CHECK

        typed_data = pd.DataFrame(columns=cols)

        printed_dir = "e:/mc_new/bg"

        #printed_dir = "e:/test_set/cover_test"

        #printed_dir2 = "e:/out78"

        #printed_dir2 = "e:/outCURRINUSE"

        outPath = "e:/hw_classif_data"

        #outPath = "e:/hw_classif_test_data"

        os.makedirs(outPath, exist_ok=True)
    
        img_dir = os.listdir(printed_dir)

        #img_dir2 = os.listdir(printed_dir2)

        typed_img_dir = [ os.path.join(printed_dir, img) for img in img_dir ]

        #typed_img_dir2 = [ os.path.join(printed_dir2, img) for img in img_dir2 ]

        #typed_img_dir.extend(typed_img_dir2)
        
        data = []
        ret_data = []
    
        #for img in typed_img_dir:

        #    ret_data = create_feature_vector(img, 'typed')
                
        #    data.append(ret_data)

        #MAX IS 77270
        upp_bound = len(typed_img_dir)//16
        with mp.Pool(processes=16) as pool:
            for idx in range(0, upp_bound*16, 16):
                print(idx)
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+0)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+1)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+2)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+3)], 'cover')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+4)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+5)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+6)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+7)], 'cover')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+8)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+9)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+10)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+11)], 'cover')))

                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+12)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+13)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+14)], 'cover')))
                ret_data.append(pool.apply_async(create_feature_vector, ( typed_img_dir[int(idx+15)], 'cover')))

                for obj in ret_data:
                    data.append(obj.get().split(','))
                ret_data = []
                 
        pool.close()
        pool.join()
        
        typed_data = pd.DataFrame(data, columns=cols)
        typed_data.to_csv(outPath+"/cover_data.csv", index=False)

        outPath = "e:/hw_classif_data"

        #outPath = "e:/hw_classif_test_data"
        
        hw_data = pd.DataFrame(columns=cols)

        hw_dir = "e:/mc_new/hw"

        #hw_dir = "e:/test_set/hw_test"

        #hw_dir = "c:/Users/Karkaras/Desktop/hw_class_ds" 

        #hw_dir2 = "c:/Users/Karkaras/Desktop/blurred_words_images"

        os.makedirs(outPath, exist_ok=True)
    
        img_dir = os.listdir(hw_dir)

        #img_dir2 = os.listdir(hw_dir2)

        img_dir = [ os.path.join(hw_dir, img) for img in img_dir ]

        #img_dir2 = [ os.path.join(hw_dir2, img) for img in img_dir2 ]

        #img_dir.extend(img_dir2)

        data = []

        ret_data = []
        '''
        for img in img_dir:

            ret_data = create_feature_vector(img, 'handwritten')
            
            data.append(ret_data)
        '''
        #MAX IS 77646
        upp_bound = len(img_dir)//16
        with mp.Pool(processes=16) as pool:
            for idx in range(0, upp_bound*16, 16):
                print(idx)
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+0)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+1)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+2)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+3)], 'handwritten')))

                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+4)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+5)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+6)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+7)], 'handwritten')))
                
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+8)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+9)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+10)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+11)], 'handwritten')))

                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+12)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+13)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+14)], 'handwritten')))
                ret_data.append(pool.apply_async(create_feature_vector, ( img_dir[int(idx+15)], 'handwritten')))

                for obj in ret_data:
                    data.append(obj.get().split(','))
                ret_data = []
                 
        pool.close()
        pool.join()


        hw_data = pd.DataFrame(data, columns=cols)
                 
        hw_data.to_csv(outPath+"/hw_data.csv", index=False)

        dfs = [ typed_data, hw_data ]

        total_data = pd.concat(dfs)

        shuffd_tot_data = pd.DataFrame(sk.utils.shuffle(total_data), columns = cols)

        print(shuffd_tot_data)

        print(type(shuffd_tot_data))
            
        #return shuffd_tot_data
    #return data


def text_feature_extractor(value):
    """
    Function to take measurements from a text string for classification purposes

    :param value: Text string
    :return: A list of values
    """
    value=str(value)
    text_length = len(value)
    words = value.count(" ") + 1
    numbers = sum(c.isdigit() for c in value)
    letters = sum(c.isalpha() for c in value)
    if text_length == 0:
        num_text_ratio = 0
    else:
        num_text_ratio = numbers/text_length
    avg_word_length = sum(len(word) for word in value) / words
    
    return [ text_length, words, num_text_ratio, avg_word_length]
    
'''
data = pd.DataFrame(columns=['sudoc'])

data = create_synthetic_sudocs(data, 15843)

print(data)

data.to_csv("e:/hw_data_dict.csv", index=False, index_label=False)
'''
### Block for data selection and function args

if __name__ == '__main__':

    load_data()

'''
X_train, X_test, y_train, y_test = data_splits(data)

#clf_algs = [ 'rf', 'hgbc' ]

clf_algs = ['lda', 'lsvc', 'qda', 'rf', 'gnb', 'hgbc', 'mlp' ]

ndims_list = [ '64' ]

datapath = "e:/font_classifier_data/"

os.makedirs(datapath, exist_ok=True)

#print(len(X_train))

supervised_learning(datapath, clf_algs, ndims_list, 'g2v' ,'plain', X_train, X_test, y_train, y_test)

print("Finished Learning Steps")
'''
### End Block
'''
data_path = 'C:/Users/Karkaras/Desktop/master-gdc-gdcdatasets-2020446966_en-2020446966_en\wdl_data_en.csv'

cols_for_use = [ 'title', 'Date Created']

data = pd.read_csv(filepath_or_buffer = data_path, usecols = cols_for_use)

print("Creating Sudocs")

data_len = len(data)

print(data_len)

data = create_synthetic_sudocs(data, data_len)

print("Finished creating sudocs")

print("Data: ", data)

print("Parsing Data")

data = parse_data(data)

print("Shuffling Data")

data = sk.utils.shuffle(data)

data.to_csv('C:/Users/Karkaras/Desktop/master-gdc-gdcdatasets-2020446966_en-2020446966_en/multiclass_data.csv')

#print("Splitting Data into train, test, vocabulary sets")

#data_splits(data)

num_data = pd.DataFrame(columns=[ 'text_length', 'words', 'num_text_ratio', 'avg_word_length', 'label'])

print(data)

for idx, row in data.iterrows():

    #print(row)
    
    row_num = text_feature_extractor(row['data'])

    row_num.append(row['label'])

    num_data = pd.concat([pd.DataFrame([row_num], columns = [ 'text_length', 'words', 'num_text_ratio', 'avg_word_length', 'label']), num_data], ignore_index=True)
    

num_data.to_csv('C:/Users/Karkaras/Desktop/master-gdc-gdcdatasets-2020446966_en-2020446966_en/num_multiclass_data.csv', index = False)


    train_data = pd.DataFrame(columns=['data'])

    label_data = pd.DataFrame(columns=['label'])

    for val in data_frame:
        train_data = pd.concat([pd.DataFrame([val], columns = train_data.columns), train_data] ,ignore_index=True)

    for val in label_frame:
        label_data = pd.concat([pd.DataFrame([val], columns = label_data.columns), label_data] ,ignore_index=True) 
'''

'''

#######CER Lines

cer = load("cer")

true_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_hw_sudoc_data_labels.csv"

pred_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_hw_sudoc_data_results.csv"

true_data = pd.read_csv(filepath_or_buffer = true_data_path, usecols = ['file_name', 'text'])

pred_data = pd.read_csv(filepath_or_buffer = pred_data_path, usecols = ['file_name', 'text'])

true_data = true_data.sort_values(by=['file_name'])

true_data['text'] = true_data['text'].str.lower()

pred_data = pred_data.sort_values(by=['file_name'])

pred_data['text'] = pred_data['text'].str.lower()

print(true_data)

print(pred_data)

cer_score = cer.compute(predictions=pred_data['text'], references=true_data['text'])

print("Handwritten CER Scores: ", cer_score)

true_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_typed_sudoc_data_labels.csv"

pred_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_typed_sudoc_data_results.csv"

true_data = pd.read_csv(filepath_or_buffer = true_data_path, usecols = ['file_name', 'text'])

pred_data = pd.read_csv(filepath_or_buffer = pred_data_path, usecols = ['file_name', 'text'])

true_data = true_data.sort_values(by=['file_name'])

true_data['text'] = true_data['text'].str.lower()

pred_data = pred_data.sort_values(by=['file_name'])

pred_data['text'] = pred_data['text'].str.lower()

print(true_data)

print(pred_data)

cer_score = cer.compute(predictions=pred_data['text'], references=true_data['text'])

print("Typed CER Scores: ", cer_score)

true_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_title_data_labels.csv"

pred_data_path = "c:/users/karkaras/desktop/proc_sample_imgs/test_set/test_title_data_results.csv"

true_data = pd.read_csv(filepath_or_buffer = true_data_path, usecols = ['file_name', 'text'])

pred_data = pd.read_csv(filepath_or_buffer = pred_data_path, usecols = ['file_name', 'text'])

true_data = true_data.sort_values(by=['file_name'])

true_data['text'] = true_data['text'].apply(lambda x: x.lower() if isinstance(x, str) else next)

pred_data = pred_data.sort_values(by=['file_name'])

pred_data['text'] = pred_data['text'].apply(lambda x: x.lower() if isinstance(x, str) else next) 

print(true_data)

print(pred_data)

cer_score = cer.compute(predictions=pred_data['text'], references=true_data['text'])

print("Title CER Scores: ", cer_score)
'''
