#import
import numpy as np
import math
import random
import csv
import re
import pickle
import tkinter as tk
from tkinter import filedialog

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding , Input , GRU , Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras import backend as K
except ModuleNotFoundError or ImportError:
    from keras.models import Sequential
    from keras.layers import Embedding , Input , GRU , Dense
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping , ModelCheckpoint
    from keras import backend as K

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer




Tokens = [
    'G', 'A', 'E',
    'H', 'Si', 'Cl', 'Br', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'I',
    '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
    'c', 'n', 'o', 's', 'p', ' ']

all_scores = []
bestmodel = None
bestdictionary = {}
Dropoutrates = [0.1, 0.2, 0.3]
normmethods = ['none', 'MinMax Scaling', 'Robust Scaling', 'Interquartile Range', 'Winsorization']


#choose file
def choose_file():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    filepath = filedialog.askopenfilename()
    root.destroy()
    return filepath


#open file
def openfile():
    while True:
        print('''Choose your .csv data file:''')
        filepath = choose_file()
        if '.csv' not in filepath:
            print('''This was not the correct file, it is not a .csv file
Choose again''')
            continue
        else:
            break
    return filepath


#extract data
def dataextracting(filepath):
    with open(filepath, 'r') as csvFile:
        reader = csv.reader(csvFile)

        it = iter(reader)
        next(it, None)

        rawsmiles = []
        rawlabels = []

        idxlabels = 3 #change according to pIC50 collumn in .csv file

        for row in it:
            try:
                if len(row) == 0 or math.isnan(float(row[idxlabels])):
                    continue
                else:
                    rawsmiles.append(row[0])
                    rawlabels.append(float(row[idxlabels]))
            except Exception as e:
                pass
        return rawsmiles, rawlabels


#split train/val and test
def trainval_test_split(fullsmilelist, labellist):
    percentage_test = 0.15

    idx_test = np.array(random.sample(range(0, len(fullsmilelist)), math.floor(percentage_test * len(fullsmilelist))))
    train_val_set = np.delete(fullsmilelist, idx_test, 0)
    train_val_labels = np.delete(labellist, idx_test)
    tokenizedlist = np.array(fullsmilelist)
    test_set = tokenizedlist[idx_test]
    labellist = np.array(labellist)
    test_labels = labellist[idx_test]

    datalist = []

    datalist.append(train_val_set)
    datalist.append(train_val_labels)
    datalist.append(test_set)
    datalist.append(test_labels)

    return datalist


#cross validation split
def data_crossvalidation_split(datalist):
    n_splits = 5

    train_val_smiles = datalist[0]
    train_val_labels = datalist[1]
    cross_validation_split = KFold(n_splits, shuffle=True)
    datacv = list(cross_validation_split.split(train_val_smiles, train_val_labels))
    return datacv


#process data
def dataprocessing(rawsmiles, rawlabels):
    smiles = []
    labels = []

    smile_len_threshold = 65

    for i in range(len(raw_smiles)):
        if (len(raw_smiles[i]) <= smile_len_threshold and 'a' not in raw_smiles[i] and 'Z' not in raw_smiles[i]
                and 'K' not in raw_smiles[i]):
            smiles.append(raw_smiles[i])
            labels.append(raw_labels[i])
    return smiles, labels


#padd smiles
def padding(smiles):
    lgt = 65

    SMILES = []
    for smile in smiles:
        if 'Br' in smile or "Si" in smile or "Cl" in smile:
            add_pad = smile.count('Br') + smile.count('Si') + smile.count('Cl')
            smile = smile.ljust(lgt + add_pad, 'A')
        else:
            smile = smile.ljust(lgt, 'A')
        smile = 'G' + smile + 'E'
        SMILES.append(smile)
    return SMILES


#create dictionary
def tokendictionarycreation(smilelist):
    dictionary = {}
    dictionarycounter = 0

    for smile in smilelist:
        for token in Tokens:
            if token in smile and token not in dictionary.keys():
                dictionary[token] = dictionarycounter
                dictionarycounter += 1
    return dictionary


#tokenize smiles (dictionary)
def dictokenizesmile(smiletotokenize, dictionary):
    tokenized = []
    for smile in smiletotokenize:
        tokensmile = []

        i = 0
        while i < len(smile):
            if i < len(smile)-1 and smile[i: i + 2] in dictionary.keys():
                tokensmile.append(dictionary[smile[i: i + 2]])
                i += 2
            elif i < len(smile)-1 and smile[i: i+2] in Tokens:
                tokensmile.append(-1)
                i += 2
            elif smile[i] in dictionary.keys():
                tokensmile.append(dictionary[smile[i]])
                i += 1
            else:
                tokensmile.append(-1)
                i += 1

        tokensmile = np.reshape(np.array(tokensmile, dtype="double"), (1, -1))
        tokenized.append(tokensmile)

    return np.concatenate(tokenized, axis=0)


#create tokenizer
def tokenizercreation(tokens, smiles):
    token_pattern = '|'.join(re.escape(token) for token in tokens)
    def custom_tokenizer(smile):
        return re.findall(token_pattern, smile)

    tokenizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None, use_idf=True)
    tokenizer = tokenizer.fit(smiles)
    return tokenizer


#tokenize smiles (vectorizer)
def vectokenizesmile(smiletotokenize, tokenizer):
    tokenized = tokenizer.transform(smiletotokenize).toarray()
    return tokenized


# normalization functions

#nonormalization
def nonormal(Pic50scores):
    plt.figure()
    plt.boxplot(Pic50scores)
    plt.title('Pic50 distribution')
    plt.show()

#minMax scaling
def MinMaxScaling(Pic50scores):
    plt.figure()
    plt.boxplot(Pic50scores)
    plt.title('Pic50 Outliers')
    plt.show()
    Pic50scores = Pic50scores.reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(Pic50scores)
    minmaxscaled = scaler.transform(Pic50scores)
    plt.figure()
    plt.boxplot(minmaxscaled)
    plt.title('MinMax Scaling Pic50')
    plt.show()
    return scaler, minmaxscaled

#robust scaling
def RobustScalingTransformer(Pic50scores):
    plt.figure()
    plt.boxplot(Pic50scores)
    plt.title('Pic50 Outliers')
    plt.show()
    Pic50scores = Pic50scores.reshape(-1,1)
    transformer=RobustScaler(quantile_range=(0.05, 0.05))
    transformer.fit(Pic50scores)
    robustscaled = transformer.transform(Pic50scores)
    plt.figure()
    plt.boxplot(robustscaled)
    plt.title('Robust Scaling Pic50')
    plt.show()
    return transformer, robustscaled

#interquartile range normalization
def InterQuartileNorm(Pic50scores):
    plt.figure()
    plt.boxplot(Pic50scores)
    plt.title('Pic50 Outliers')
    plt.show()
    Pic50scores = Pic50scores.reshape(-1,1)
    iqrtransformer=RobustScaler()
    iqrtransformer.fit(Pic50scores)
    normalized = iqrtransformer.transform(Pic50scores)
    plt.figure()
    plt.boxplot(normalized)
    plt.title('InterQuartile Normalization Pic50')
    plt.show()
    return iqrtransformer, normalized

#winsorization
def Winsorization(Pic50scores):
    plt.figure()
    plt.boxplot(Pic50scores)
    plt.title('Pic50 Outliers')
    plt.show()
    winsorized = winsorize(Pic50scores,(0.05, 0.05))
    plt.figure()
    plt.boxplot(winsorized)
    plt.title('Winsorized Pic50')
    plt.show()
    return winsorized


#adjusted r2 score function
def adjusted_r2_score(y_true, y_pred):
    residual_sum_of_squares = K.sum(K.square(y_true - y_pred))
    total_sum_of_squares = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - residual_sum_of_squares / total_sum_of_squares
    n = K.cast(K.shape(y_true)[0], K.floatx())
    p = K.cast(K.shape(y_pred)[-1], K.floatx())
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2


def concordance_correlation_coefficient(y_true, y_pred):
    # Mean of true and predicted values
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    
    # Deviation from the mean
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    
    # Covariance
    covariance = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # CCC formula
    ccc = (2 * covariance) / (var_true + var_pred + K.square(mean_true - mean_pred))
    
    return ccc

def matthews_correlation_coefficient(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def regression_plot(name, y_true,y_pred):
    """
    Function that graphs a scatter plot and the respective regression line to 
    evaluate the QSAR models.
    Parameters
    ----------
    y_true: True values from the label
    y_pred: Predictions obtained from the model
    Returns
    -------
    This function returns a scatter plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    plt.show()
    fig.savefig(f'{name}.png')


#create model
def model_creation(Dropout, InputLength):
    Activation = 'relu'
    unitsEmbedding = 128
    unitsGRU = 128
    unitsDense = 128
    unitsOutput = 1

    model = Sequential()

    model.add(Input(shape=(InputLength,)))

    model.add(Embedding(len(tokendictionary), unitsEmbedding, input_shape=InputLength))

    model.add(GRU(units=unitsGRU, return_sequences=True, dropout=Dropout))
    model.add(GRU(units=unitsGRU, dropout=Dropout))

    model.add(Dense(units=unitsDense, activation=Activation))

    model.add(Dense(units=unitsOutput, activation='linear'))

    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',
                  metrics=["mean_squared_error", "r2_score", matthews_correlation_coefficient, concordance_correlation_coefficient])

    return model


#train model
def model_training(trainsmiles, trainlabels, valsmiles, vallabels, drop, inputlength = 65):
    epochs = 100
    batch_size = 32

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    model = model_creation(drop, inputlength)

    model.fit(
            trainsmiles, trainlabels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(valsmiles, vallabels),
            callbacks=[es, mc]
        )

    scores = model.evaluate(valsmiles, vallabels)
    
    print(scores)
    
    model.summary()
    return model, scores



file_path = openfile()
with open("resultsfile.csv", 'a') as resultfilescsv:
    resultfilescsv.write("Data Set,Tokenization method,Dropout Rate,Normalization Method,MSE,CD,MCC,CCC\n")
raw_smiles, raw_labels = dataextracting(file_path)
Smiles, Labels = dataprocessing(raw_smiles, raw_labels)

data=trainval_test_split(Smiles, Labels)

#grid search
for tokenizemethod in ['Dictionary', 'TfidfVectorizer']:
    
    for dropout in Dropoutrates:

        for normal in range(0,5):

            data_cv = data_crossvalidation_split(data)
            
            best_scores = [10000, -10000, -10000, -10000]
            
            for train_index, val_index in data_cv:
                    train_smiles, val_smiles = data[0][train_index], data[0][val_index]
                    train_labels, val_labels = data[1][train_index], data[1][val_index]
                    
            if tokenizemethod == 'Dictionary':
                    #process training
                    padded_t_smiles = padding(train_smiles)

                    #process validation
                    padded_v_smiles = padding(val_smiles)

            #normalize labels
            if normal == 0:
                    nonormal(train_labels)
            if normal == 1:
                    Scaler, train_labels = MinMaxScaling(train_labels)
                    val_labels = val_labels.reshape(-1,1)
                    val_labels = Scaler.transform(val_labels)
            if normal == 2:
                    Transformer, train_labels = RobustScalingTransformer(train_labels)
                    val_labels = val_labels.reshape(-1,1)
                    val_labels = Transformer.transform(val_labels)
            if normal == 3:
                    Transformer, train_labels = InterQuartileNorm(train_labels)
                    val_labels = val_labels.reshape(-1,1)
                    val_labels = Transformer.transform(val_labels)
            if normal == 4:
                    train_labels = Winsorization(train_labels)
                    val_labels = Winsorization(val_labels)
            
            if tokenizemethod == 'Dictionary':
                    #create dictionary based on training
                    tokendictionary = tokendictionarycreation(padded_t_smiles)
                    
                    #tokenize training
                    tokenized_t_smiles = dictokenizesmile(padded_t_smiles, tokendictionary)

                    #tokenize validation
                    tokenized_v_smiles = dictokenizesmile(padded_v_smiles, tokendictionary)
                            
                    #train with train and evaluate model with validation
                    model, model_scores = model_training(tokenized_t_smiles, train_labels, tokenized_v_smiles, val_labels, dropout)
                    
                    #add other metrics    
                    if model_scores[1]<best_scores[0] and model_scores[2]>best_scores[1] and model_scores[3]>best_scores[2] and model_scores[4]>best_scores[3]:
                        best_scores[0], best_scores[1], best_scores[2], best_scores[3], best_dictionary = model_scores[1], model_scores[2], model_scores[3], model_scores[4], tokendictionary
                        if normal == 1:
                            bestscaler = Scaler
                        if normal == 2 or normal == 3:
                            besttransformer = Transformer
                        bestmodel = model
                    
                    all_scores.append(model_scores)
                    
                    
            if tokenizemethod == 'TfidfVectorizer':    
                    #create dictionary based on training
                    Tokenizer = tokenizercreation(Tokens, train_smiles)
                    
                    #tokenize training
                    tokenized_t_smiles = vectokenizesmile(train_smiles, Tokenizer)

                    #tokenize validation
                    tokenized_v_smiles = vectokenizesmile(val_smiles, Tokenizer)

                    #train with train and evaluate model with validation
                    model, model_scores = model_training(tokenized_t_smiles, train_labels, tokenized_v_smiles, val_labels, dropout, tokenized_t_smiles.shape[1])


                    #add other metrics    
                    if model_scores[1]<best_scores[0] and model_scores[2]>best_scores[1] and model_scores[3]>best_scores[2] and model_scores[4]>best_scores[3]:
                        best_scores[0], best_scores[1], best_scores[2], best_scores[3], best_tokenizer = model_scores[1], model_scores[2], model_scores[3], model_scores[4], Tokenizer
                        if normal == 1:
                            bestscaler = Scaler
                        if normal == 2 or normal == 3:
                            besttransformer = Transformer
                        bestmodel = model
                    
                    all_scores.append(model_scores)

            #obtain best model from crossval
            print(f"Best model: {bestmodel}")
            print(f"Best scores: {best_scores}")


            if tokenizemethod == 'Dictionary':
                #padd test set
                padded_test_smiles = padding(data[2])

                #tokenize test set
                tokenized_test_smiles = dictokenizesmile(padded_test_smiles, best_dictionary)

            if tokenizemethod == 'TfidfVectorizer': 
                #tokenize test set
                tokenized_test_smiles = vectokenizesmile(data[2], best_tokenizer)

            #normalize test labels
            test_labels = data[3]

            if normal == 1:
                test_labels = bestscaler.transform(test_labels.reshape(-1,1))
            if normal == 2 or normal == 3:
                test_labels = besttransformer.transform(test_labels.reshape(-1,1))
            if normal == 4:
                test_labels = Winsorization(test_labels)    

            #prediction
            predictions = bestmodel.predict(tokenized_test_smiles)
            if normal == 1:
                predictions = bestscaler.inverse_transform(predictions)
            if normal == 2 or normal == 3:
                predictions = besttransformer.inverse_transform(predictions)
            

            # Model's evaluation with two example SMILES strings
            list_ss = ["CC(=O)Nc1cccc(C2(C)CCN(CCc3ccccc3)CC2C)c1", "CN1CCC23CCCCC2C1Cc1ccc(O)cc13"]  #5.96 e 8.64
            list_ss = padding(list_ss)
            
            if tokenizemethod == 'Dictionary':
                list_ss = dictokenizesmile(list_ss, bestdictionary)
            if tokenizemethod == 'TfidfVectorizer':
                list_ss = vectokenizesmile(list_ss, best_tokenizer)
           
            prediction_2 = bestmodel.predict(list_ss)

            print(predictions)
            print(prediction_2)
            
            regression_plot(f"regression_{file_path.split('/')[-1].split('.')[0]}_{tokenizemethod}_{dropout}_{normmethods[normal]}", test_labels , predictions)
            
            # Model's evaluation with the test set
            loss, eval_mean_squared_error, eval_coefficient_of_determination, eval_matthews_correlation_coefficient, eval_concordance_correlation_coefficient = bestmodel.evaluate(tokenized_test_smiles, test_labels) 
            print(f'\nLoss: {loss}, RMSE: {eval_mean_squared_error}')

            with open("resultsfile.txt", 'a') as resultfilestxt:
                resultfilestxt.write(f'''
Using {tokenizemethod}
Dropout: {dropout}
Normalization method: {normmethods[normal]}

Loss: {loss}
MSE: {eval_mean_squared_error}
CD: {eval_coefficient_of_determination}
MCC: {eval_matthews_correlation_coefficient}
CCC: {eval_concordance_correlation_coefficient}

Predictions:
"CC(=O)Nc1cccc(C2(C)CCN(CCc3ccccc3)CC2C)c1" : {prediction_2[0]}
"CN1CCC23CCCCC2C1Cc1ccc(O)cc13" : {prediction_2[1]}

{data[2]}
{predictions}

___________________________________________________________________________________________________
''')
            with open("resultsfile.csv", 'a') as resultfilescsv:
                resultfilescsv.write(f'''{file_path.split('/')[-1]},{tokenizemethod},{dropout},{normmethods[normal]},{eval_mean_squared_error},{eval_coefficient_of_determination},{eval_matthews_correlation_coefficient},{eval_concordance_correlation_coefficient}\n''')

            with open(f"model_{file_path.split('/')[-1].split('.')[0]}_{tokenizemethod}_{dropout}_{normmethods[normal]}.pkl", 'wb') as file:
                pickle.dump(bestmodel, file)
