import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import datetime
import pickle
import numpy.random as nr
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from PIL import Image

def get_all_files(folder, ext):
    
    # Receives a filepath and a string with the desired file extension, returns list with all the files inside said folder
    
    all_files = []
    #Iterate through all files in folder
    for file in os.listdir(folder):
        #Get the file extension
        _,  file_ext = os.path.splitext(file)

        #If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    #Get list of all files
    return all_files

def get_average(filepaths):

    # Receives a filepath with images, returns an average image of all pictures, saving it to the root folder
    
    try:    
        im = Image.open(filepaths[0])
    except: 
        print("Error in the first image")
        
    width, height = im.size
           
    average = np.zeros([height, width, 3])  
    errors = 0

    for i in range(1, len(filepaths)):
        
        try:
            im = Image.open(filepaths[i])
            imarray = np.array(im)           
            average[:, :, :] += imarray[:, :, :]  
            
        except:
            errors += 1
            continue

    average = average / (len(filepaths) - errors)
    aux = np.zeros([height, width, 3], dtype=np.uint8)
    aux[:, :, :] = average [:, :, :]
       
    im = Image.fromarray(aux)
    im.save('25 - B.jpg')       
        
    return im

def get_eucdistance(picarray, avearray):

    # Receives 2 numpy array of images, returns euclidean distance    

    width, height, color = avearray.shape
    result = np.zeros([width, height, color])
    result[:, :, :] = picarray[:, :, :] - avearray[:, :, :] 
     
    result[:, :, :] = np.absolute(result)    
    sigma = sum(sum(sum(result[:])))
    
    maxsigma = 3*width*height*255
    sigma = (sigma)/maxsigma

    return sigma
     
def get_test(filepaths):
    
    # Receives a filepath to a folder, creates a database with all the information and analysis. 
    
    df = pd.DataFrame()
    
    for i in range(len(filepaths)):
    
        fp = filepaths[i]
        index = fp.find('cctv')
        
        # histBnW = get_bnwhistogram(fp)
        # histBnWA = get_bnwhistogram(average)
        # histR, histG, histB =  get_colorhistogram(fp)
        # histRA, histGA, histBA = get_colorhistogram(average)
            
        print(fp)
        df.loc[i, 'CCTV'] = int(fp[index+5]+fp[index+6]+fp[index+7])
        df.loc[i, 'Year'] = int(fp[index+9]+fp[index+10]+fp[index+11]+fp[index+12])
        df.loc[i, 'Month'] = int(fp[index+14]+fp[index+15])
        df.loc[i, 'Day'] = int(fp[index+25]+fp[index+26])
        df.loc[i, 'Hour'] = int(fp[index+47]+fp[index+48])
        df.loc[i, 'Minute'] = int(fp[index+50]+fp[index+51])
        df.loc[i, 'Second'] = int(fp[index+53]+fp[index+54])
        # df.loc[i, 'Euclidean Distance'] = get_eucdistance(filepaths[i], average)
        # df.loc[i, 'Wasserstein Distance BnW'] = wasserstein_distance(histBnW,histBnWA)
        # df.loc[i, 'Wasserstein Distance R'] = wasserstein_distance(histR, histRA)
        # df.loc[i, 'Wasserstein Distance G'] = wasserstein_distance(histG, histGA)
        # df.loc[i, 'Wasserstein Distance B'] = wasserstein_distance(histB, histBA)
        analysis = get_average_analysis(filepaths[i])
        df.loc[i, 'Direction'] = get_prediction(analysis)
        
    # df.loc[0, 'Index'] = 'ED Average'
    # df.loc[0, 'Value'] = np.average(df['Euclidean Distance'])
    # df.loc[1, 'Index'] = 'ED Max'
    # df.loc[1, 'Value'] = np.max(df['Euclidean Distance'])
    # df.loc[2, 'Index'] = 'ED Min'
    # df.loc[2, 'Value'] = np.min(df['Euclidean Distance'])
    # df.loc[3, 'Index'] = 'ED Variance'
    # df.loc[3, 'Value'] = np.var(df['Euclidean Distance'])
    # df.loc[4, 'Index'] = 'WD R'
    # df.loc[4, 'Value'] = np.average(df['Wasserstein Distance R'])
    # df.loc[5, 'Index'] = 'WD G'
    # df.loc[5, 'Value'] = np.average(df['Wasserstein Distance G'])
    # df.loc[6, 'Index'] = 'WD B'
    # df.loc[6, 'Value'] = np.average(df['Wasserstein Distance B'])
    df.loc[7, 'Index'] = 'Directions'
    df.loc[7, 'Value'] = 0
 
    return df
    
def get_results(test):
    
    # Receives a filepath to and Excel containing test results. Returns information regarding said results.
    
    df = pd.read_excel(test)
    
    df.loc[0, 'Index'] = 'Average'
    df.loc[0, 'Value'] = np.average(df['Result'])
    df.loc[1, 'Index'] = 'Max'
    df.loc[1, 'Value'] = np.max(df['Result'])
    df.loc[2, 'Index'] = 'Min'
    df.loc[2, 'Value'] = np.min(df['Result'])
    df.loc[3, 'Index'] = 'Variance'
    df.loc[3, 'Value'] = np.var(df['Result'])
    
    # hist, bin_edges = np.histogram(df['Result'])
    n, bins, patches = plt.hist(x=df['Result'], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Result Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    
    return df

def get_bnwhistogram(filepath):
    
    # Receives filepath to image, returns said image in greyscale
    
    try:    
        im = Image.open(filepath)
    except: 
        print("Error")
        
    width, height = im.size

    im = np.array(Image.open(filepath).convert('L'))
    imarray = np.array(im)              
    hist = [0.0] * 256
    
    for i in range(height):
      for j in range(width):
        hist[imarray[i, j]] += 1

        
    hist = np.array(hist) / (height * width)    

    return hist    

def get_colorhistogram(picarray):

    # Receives numpy array of image, returns color histogram

    width, height, color = picarray.shape     
   
    histR = [0.0] * 256
    histG = [0.0] * 256
    histB = [0.0] * 256

    for i in range(width):
      for j in range(height):

        histR[picarray[i, j, 0]] += 1
        histG[picarray[i, j, 1]] += 1
        histB[picarray[i, j, 2]] += 1
        
    histR = np.array(histR) / (height * width)    
    histG = np.array(histG) / (height * width)    
    histB = np.array(histB) / (height * width)        

    hist = (histR, histG, histB)
        
    return hist

def getseg(filepath, sx, sy):
    
    # Receives filepath to image, amount of x and y divisions
    # Returns list of image segments
    
    #sx: qty of x divisions
    #sy: qty of y divisions
    
    img = Image.open(filepath)

    height, width = img.size
    
    arrimg = [] * (sx*sy)
    limx1 = limx2 = limy1 = limy2 = 0
    
    limx1 = int(width/sx)
    limx2 = width/sx
    if(limx2 - limx1 != 0): limx2 = int(limx2)+1
    limy1 = int(height/sy)
    limy2 = height/sy
    if(limy2 - limy1 != 0): limy2 = int(limy2)+1
    
    imgarray = np.asarray(img)
    aux = np.array(0)
    
    for i in range(sx):
        for j in range(sy):

            if(i != sx and j != sy):
                aux = imgarray[i*limx1:((i+1)*limx1),i*limy1:((i+1)*limy1),:]             
            
            elif(i == sx and j != sy):
                aux = imgarray[i*limx1:,i*limy1:((i+1)*limy1),:]   
                
            elif(i != sx and j == sy):
                aux = imgarray[i*limx1:((i+1)*limx1),i*limy1:,:]
                
            elif(i == sx and j == sy):
                aux = imgarray[i*limx1:,i*limy1:,:]
        
            arrimg.append(aux)

    return arrimg    

def get_subdirs(filepath):
    
    # Receives filepath to folder, returns all containing subdirectories
    
    aux = []
    for entry in os.scandir(path=filepath):
        if not entry.name.startswith('.') and entry.is_dir():
            aux.append(entry.path)
            
    if(aux == []): return 0
    else: return aux
    
def get_average_analysis(filepath):
    
    # Receives filepath to image, returns database with all analysis of image against all average pictures
    
    ave_images = get_all_files(average, 'jpg')
    data = pd.DataFrame()
    
    for index in enumerate(ave_images):
        data = data.append(get_iterative_analysis(filepath, index[1]))
        
    return data 

def get_prediction(fp):

    # Receives filepath to image, returns best estimate of orientation
    
    [model, scaler] = get_model_scaler()
    analysis = get_average_analysis(fp)
    featlist = []
    average_qty = analysis.shape[0]
    probabilities = np.zeros(average_qty)
    
    for feat in analysis.columns:
        featlist.append(feat)
        
    features = np.zeros([average_qty, len(featlist)])
    
    for feat in enumerate(featlist):
        features[:, feat[0]] = np.array(analysis[feat[1]])
    
    features = scaler.transform(features[:,:])
    probabilities = model.predict_proba(features)   
    maxindex = np.argmax(probabilities[:, 1])

    return [maxindex, probabilities[maxindex, 1], get_ave_fp(maxindex)]
        
    
def get_rawdb(filepaths):
    
    # Receives filepath to folder contating images, returns dataframe with basic information and labels as an alternative way of training
    # Useful when saving dataframe to Excel for easy viewing
    
    df = pd.DataFrame()
    
    for i in range(len(filepaths)):
    
        fp = filepaths[i]
        index = fp.find('cctv') + 5    

        df.loc[i, 'CCTV'] = int(fp[index:index+2])
        df.loc[i, 'Year'] = int(fp[index+9:index+12])
        df.loc[i, 'Month'] = int(fp[index+14:index+15])
        df.loc[i, 'Day'] = int(fp[index+25:index+26])
        df.loc[i, 'Hour'] = int(fp[index+47:index+48])
        df.loc[i, 'Minute'] = int(fp[index+50:index+51])
        df.loc[i, 'Second'] = int(fp[index+53:index+54])
        # df.loc[i, 'TSLI'] = if(i != 0) get.timeelapsed(), else = 0
        # Time Since Last Image
        df.loc[i, 'Direction Change'] = False
        df.loc[i, 'Predicted Direction'] = 0
        df.loc[i, 'Real Direction'] = 0
        df.loc[i, 'Real Direction w/ offset'] = 0
        df.loc[i, 'Flag'] = 0 # If changed to 1, the model will use it for training   
 
    return df    

def get_rawdb(filepaths, cctv, year, month, day):
    
    # Overloaded method for efficency
    
    df = pd.DataFrame()
    aux = len(filepaths)
    aux2 = np.ones(aux)
    aux2[:] = cctv
    df.loc[:, 'CCTV'] = aux2

    for i in range(aux):
    
        fp = filepaths[i]
        index = fp.find('cctv')
        df.loc[i, 'Time'] = datetime.datetime(year, month, day, int(fp[index+47]+fp[index+48]), int(fp[index+50]+fp[index+51]), int(fp[index+53]+fp[index+54]))
        # df.loc[i, 'TSLI'] = if(i != 0) get.timeelapsed(), else = 0
        # Time Since Last Image
        
    aux2[:] = False        
    df.loc[:, 'Direction Change'] = aux2
    aux2[:] = 0   
    df.loc[:, 'Predicted Direction'] = aux2
    df.loc[:, 'Real Direction'] = aux2
    df.loc[:, 'Real Direction w/ offset'] = aux2
    df.loc[:, 'Flag'] = aux2 
       
    return df   
   
def score_model(probs, threshold):
    
    # Returns 1 if estimate is more than the defined threshold
    
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

def print_metrics(labels, scores):
    
    # Creates plots with model accuracy. Only useful when real labels are provided
    
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

def plot_auc(labels, probs):
    
    # Compute the false positive rate, true positive rate and threshold along with the AUC
    
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def is_number(val):
    
    # Receives value, determines if it is a integer value
    
    try:
        int(val)
        return True
    except ValueError:
        return False

def find_pictures(fp):
    
    # Receives filepath to folder containing subdirectories, returns all images information in a dataframe
    
    months_dir = get_subdirs(fp)
    df = pd.DataFrame()
    
    fp = months_dir[0]
    index = fp.find('cctv')
    fol_cctv = int(fp[index+5]+fp[index+6]+fp[index+7])
    fol_year = int(fp[index+9]+fp[index+10]+fp[index+11]+fp[index+12])

    for month in months_dir:
        
        fp = month
        try:
            fol_month = int(fp[index+14]+fp[index+15])        
        except: break
            
        fol_year = int(fp[index+9]+fp[index+10]+fp[index+11]+fp[index+12])
        days_dir = get_subdirs(month)   
        
        for day in days_dir:
            jpg_files = get_all_files(day, 'jpg')
            fol_day = int(day[index+25]+day[index+26])
            df = df.append(get_rawdb(jpg_files, fol_cctv, fol_year, fol_month, fol_day))
                  
    return df

def read_db(file):
    
    # Receives string with Excel name, returns dataframe with the information of images for model training
    
    db = pd.read_excel(file, index_col=[0])

    return db[db['Flag'] == 1]

def save_analysis(db):
    
    # Receives excel with raw data of each image, generates analysis of all images against all averages
    # Returns dataframe and saves it in a pickle
    
    db = read_db(db)
    
    analysisdb = pd.DataFrame()
    auxdb = pd.DataFrame()
    averagefolder = get_all_files(average, '.jpg')
    
    for i in range(db.shape[0]):      
        aux = db.iloc[i]
        time = aux['Time']
        filepath = root + ("/") + str(time.year) + ("_") + str('{:02d}'.format(time.month)) + ("/")
        fp = filepath + str(time.year) + ("_") + str('{:02d}'.format(time.month)) + ("_") + str('{:02d}'.format(time.day)) + ("/")
        filename = "cctv" + str(aux['CCTV']) + "_" + str(time.year) + "_" 
        filename = filename + str('{:02d}'.format(time.month)) + "_" + str('{:02d}'.format(time.day)) + "-" 
        filename = filename + str('{:02d}'.format(time.hour)) + "_" + str('{:02d}'.format(time.minute)) + "_" + str('{:02d}'.format(time.second))
        
        # list     
          # 0   CCTV
          # 1   Datetime
          # 2   Direction Change
          # 3   Predicted Direction
          # 4   Real Direction	
          # 5   Real Direction w/ offset
          # 6   Flag (bool)

        auxdb = get_average_analysis(fp + filename + '.jpg')

        for index in enumerate(ID):
            auxdb.insert(len(auxdb.columns), ID[index[0]], aux[index[1]])
            
        auxdb.insert(len(auxdb.columns), 'Compared Direction', filename)
        auxdb.insert(len(auxdb.columns), 'Average FP', averagefolder[:])
        analysisdb = analysisdb.append(auxdb)
        del(auxdb)

    filename = 'testpickle.pickle'
    pickle.dump(analysisdb, open(filename, 'wb'))
    
    return analysisdb

def load_pickle(pfile):
    
    # Receives filepath of a pickle, returns it as a dataframe
    
    pdata = pd.DataFrame()
    pdata = pickle.load(open(pdata, "rb"))

    return pdata

def get_iterative_analysis(f1, f2):
    
    # Receives filepath of 2 images, returns analysis as a dataframe
    
    data = pd.DataFrame()
    picarray = np.array(Image.open(f1))
    picarray2 = np.array(Image.open(f2))
    
    picarrayseg = getseg(f1, m, n)
    picarrayseg2 = getseg(f2, m, n)
    
    picseglist = [] * (m*n)
    picseglist2 = [] * (m*n)
    
    for i in range(m*n):
        picseglist.append(get_colorhistogram(picarrayseg[i]))
        picseglist2.append(get_colorhistogram(picarrayseg2[i]))
        
        
    histR, histG, histB =  get_colorhistogram(picarray)
    histR2, histG2, histB2 = get_colorhistogram(picarray2) 

    data.loc[i, 'Euclidean Distance'] = get_eucdistance(picarray, picarray2)
    data.loc[i, 'Wasserstein Distance BnW'] = wasserstein_distance(get_bnwhistogram(f1),get_bnwhistogram(f2))
    data.loc[i, 'Wasserstein Distance R'] = wasserstein_distance(histR, histR2)
    data.loc[i, 'Wasserstein Distance G'] = wasserstein_distance(histG, histG2)
    data.loc[i, 'Wasserstein Distance B'] = wasserstein_distance(histB, histB2)
    # data.loc[i, 'Horizontal Divisions'] = m
    # data.loc[i, 'Vertical Divisions'] = n
    
    for j in range(m*n):
        
        picseglist.append(get_colorhistogram(picarrayseg[j]))
        data.loc[i, 'ED Seg ' + str(j)] = get_eucdistance(picarrayseg[j], picarrayseg2[j])    
        data.loc[i, 'WD R Seg ' + str(j)] = wasserstein_distance(picseglist[j][0], picseglist2[j][0])
        data.loc[i, 'WD G Seg ' + str(j)] = wasserstein_distance(picseglist[j][1], picseglist2[j][1])
        data.loc[i, 'WD B Seg ' + str(j)] = wasserstein_distance(picseglist[j][2], picseglist2[j][2])

    picseglist.clear()
    picseglist2.clear()
        
    return data 

def direction_change(fp):
    
    # Receives image list or filepath to directory, analyzes if there is a direction change from one picture to the other
    
    jpg_files = return_files(fp)
    [model, scaler] = get_model_scaler()
    
    results = pd.DataFrame()
    probabilities = scores = np.zeros(len(jpg_files) - 1)
    featlist = []
        
    for i in range(len(jpg_files)-1):
        results = results.append(get_iterative_analysis(jpg_files[i], jpg_files[i+1]))

    for feat in results.columns:
        featlist.append(feat)
        
    features = np.zeros([results.shape[0], len(featlist)])
    
    for feat in enumerate(featlist):
        features[:, feat[0]] = np.array(results[feat[1]])
    
    features = scaler.transform(features[:,:])
    probabilities = model.predict_proba(features)
    scores = score_model(probabilities, model_threshold)
        
    # results.to_excel("Resultados.xlsx") 
        
    return [probabilities, scores]

def train_set(folderA, folderB):
    
    # Receives 2 filepaths to folders containing chosen images for training
    # Creates scaler and model, saving them as a pickle
    # The images are named "A-B", being A the number of picture, and B if it is the same orientation or not (1 or 0)
    # Creates a copy of the analysis in and Excel for easy viewing
    # Prints results of model
    
    excelname = 'TestAnalysis.xlsx'
    
    jpg1 = get_all_files(folderA, '.jpg')
    jpg2 = get_all_files(folderB, '.jpg')
    label = np.zeros(len(jpg1))
    
    for index in enumerate(jpg1):
        aux = index[1].find('-') + 1
        if(index[1][aux] == "1"):
            label[index[0]] = 1

    analysis = pd.DataFrame()

    for index in enumerate(jpg1):
        analysis = analysis.append(get_iterative_analysis(jpg1[index[0]], jpg2[index[0]]))
        
    analysis['Label'] = label
    
    analysis.to_excel(excelname)
    
    featlist = []
    print(analysis.columns)
    for feat in analysis.columns:
        featlist.append(feat)
    
    features = np.zeros([analysis.shape[0], len(featlist)])
        
    for feat in enumerate(featlist):
        features[:, feat[0]] = np.array(analysis[feat[1]])
    
    # As from here is standard trainset
    nr.seed(9988)
    indx = range(analysis.shape[0])
    indx = ms.train_test_split(indx, test_size = 5)
        
    x_train = features[indx[0], : len(featlist) - 1]
    y_train = np.ravel(label[indx[0]])
    x_test = features[indx[1], : len(featlist) - 1]
    y_test = np.ravel(label[indx[1]])


    scaler = preprocessing.StandardScaler().fit(x_train[:,: len(featlist) - 1])
    x_train[:,: len(featlist)] = scaler.transform(x_train[:,: len(featlist) - 1])
    
    (pd.DataFrame(x_test)).to_excel('OG2 trainset.xlsx')
    
    x_test[:,:] = scaler.transform(x_test[:,: len(featlist) - 1])
    
    logistic_mod = linear_model.LogisticRegression() 
    logistic_mod.fit(x_train[:, : len(featlist) - 1], y_train)
    
    probabilities = logistic_mod.predict_proba(x_test[:, : len(featlist) - 1])

    scores = score_model(probabilities, model_threshold)
    probs_positive = np.concatenate((np.ones((probabilities.shape[0], 1)), 
                             np.zeros((probabilities.shape[0], 1))),
                             axis = 1)
    print_metrics(y_test, scores)  
    plot_auc(y_test, probs_positive) 
    
    pickle.dump(logistic_mod, open("model.pickle", 'wb'))
    pickle.dump(scaler, open("scaler.pickle", 'wb'))
       
    return

def return_files(fp):
    
    # Receives filepaths as a string or list format, returns list of images filepaths
    
    if(isinstance(fp, str)): return get_all_files(fp, 'jpg')
    elif(isinstance(fp, list)): return fp
    
    else: return False

def get_model_scaler():
    
    # Reads pickles to return model and scaler
    
    with open(modelfile, 'rb') as file:  
        model = pickle.load(file)
    with open(scalerfile, 'rb') as file:
        scaler = pickle.load(file)    
 
    return [model, scaler]    

def get_ave_fp(index):
    
    # Receives index of best estimate, returns its filepath
    
    average_pics = get_all_files(average, 'jpg')
    
    return average_pics[index]
 
ID = ['CCTV', 'Time', 'Direction Change', 'Predicted Direction', 'Real Direction', 'Real Direction w/ offset', 'Flag']
average = "Filepath to folder containing average images"
root = "Filepath to root folder"
folder = "Filepath to folder for analysis"
test = "Filepath to excel containing get_test result"
database = "Filepath to excel containing database"
raw_pickles = "Filepath to folder containing pickles"
modelfile = 'model.pickle'
scalerfile = 'scaler.pickle'
folderA = "Filepath to folder containing training set A"
folderB = "Filepath to folder containing training set A"
m = 4 
n = 3
model_threshold = 0.7 

