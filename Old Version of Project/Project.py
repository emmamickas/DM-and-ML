
#@article{scikit-learn,
# title={Scikit-learn: Machine Learning in {P}ython},
# author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
#         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
#         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
#         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
# journal={Journal of Machine Learning Research},
# volume={12},
# year={2011}
#}

import csv
import math
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Perceptron
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
from scipy import ndimage
from time import time
from sklearn import manifold, datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.fixes import parse_version
from sklearn.feature_extraction.image import grid_to_graph
import skimage
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.tree import DecisionTreeRegressor

allUsersQuestions = {}
allQuestions = {}
allQuestionNames = []

def CleanData():
    '''
    Cleans the data to eliminate null values as well as the first line holding labels.
    '''

    open_file = open('data-final.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Copy.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 1

    for line in file_in:
        if i == 1:
            i += 1
        if 'NULL' not in line[:100]:
            file_out.writerow(line[:100])
    
    open_file.close()
    open_file2.close()

def TestingShrinker():

    '''
    Shrinks the data down to a useable size for testing and reads in this smaller amount of data. 
    '''

    open_file = open('data-final.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Testing.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0

    for line in file_in:
        if i == 0:
            i += 1
            continue
        if 'NULL' not in line[:100]:
            file_out.writerow(line[:100])
            i += 1
        if i > 5000:
            break
    
    open_file.close()
    open_file2.close()

    open_file = open('data-final - Testing.csv', 'r', encoding='utf-8', newline='')
    fulldataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(100)), max_rows=5000)

    print(fulldataset.shape)

    open_file.close()

    open_file = open('data-final - Testing.csv', 'r', encoding='utf-8', newline='')
    dataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(50)), max_rows=5000)

    print(dataset.shape)

    open_file.close()

    sampledataset = dataset[0:4000]
    testdataset = dataset[4000:5000]

    return dataset, sampledataset, testdataset

def Shrinker():

    '''
    Shrinks the data to a useable size (larger than testing) and outputs the cleaned, shrunk data to a file called 'data-final - Running.csv'
    '''

    open_file = open('data-final.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Running.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0
    linecount = 0
    start = 0

    for line in file_in:
        if start == 0:
            start += 1
            continue
        throwaway, zero = divmod(i, 50)
        if 'NULL' not in line[:100] and zero == 0:
            file_out.writerow(line[:100])
            linecount += 1
        i += 1
        if linecount >= 20000:
            break
    
    open_file.close()
    open_file2.close()
    return

def ReadInData():

    '''
    Reads in the data and parses it into three separate numpy matrices for interpretation
    '''

    open_file = open('data-final - Running.csv', 'r', encoding='utf-8', newline='')
    dataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(50)), max_rows=1000000)

    print(dataset.shape)

    fulllength, throwaway = dataset.shape

    split = int(fulllength) * 0.8

    open_file.close()

    sampledataset = dataset[0:int(split)]
    testdataset = dataset[int(split):int(fulllength)]

    print("Done reading in data...")

    return dataset, sampledataset, testdataset

def CalculateExtroversion(row):

    '''
    Calculates the total by adding or subtracting a question that pertains to the trait.
    '''

    total = 0
    total += row[0]
    total -= row[1]
    total += row[2]
    total -= row[3]
    total += row[4]
    total -= row[5]
    total += row[6]
    total -= row[7]
    total += row[8]
    total -= row[9]
    return total

def CalculateNeuroticism(row):

    '''
    Calculates the total by adding or subtracting a question that pertains to the trait.
    '''

    total = 0
    total += row[10]
    total -= row[11]
    total += row[12]
    total -= row[13]
    total += row[14]
    total -= row[15]
    total += row[16]
    total -= row[17]
    total += row[18]
    total -= row[19]
    return total

def CalculateAgreeableness(row):

    '''
    Calculates the total by adding or subtracting a question that pertains to the trait.
    '''
    
    total = 0
    total -= row[20]
    total += row[21]
    total -= row[22]
    total += row[23]
    total -= row[24]
    total += row[25]
    total -= row[26]
    total += row[27]
    total -= row[28]
    total += row[29]
    return total

def CalculateConscientiousness(row):

    '''
    Calculates the total by adding or subtracting a question that pertains to the trait.
    '''
    
    total = 0
    total += row[30]
    total -= row[31]
    total += row[32]
    total -= row[33]
    total += row[34]
    total -= row[35]
    total += row[36]
    total -= row[37]
    total += row[38]
    total -= row[39]
    return total

def CalculateOpenness(row):

    '''
    Calculates the total by adding or subtracting a question that pertains to the trait.
    '''
    
    total = 0
    total += row[40]
    total -= row[41]
    total += row[42]
    total -= row[43]
    total += row[44]
    total -= row[45]
    total += row[46]
    total -= row[47]
    total += row[48]
    total -= row[49]
    return total

def CalculateIndividualTotals(dataset):

    '''
    Calculates the total of each trait for all data and stores it in a matrix.
    '''
    
    open_file = open('data-final - IndividualTotals.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    datasettotals = np.ndarray(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        datasettotals[i][0] = CalculateExtroversion(row)
        datasettotals[i][1] = CalculateNeuroticism(row)
        datasettotals[i][2] = CalculateAgreeableness(row)
        datasettotals[i][3] = CalculateConscientiousness(row)
        datasettotals[i][4] = CalculateOpenness(row)
        file_out.writerow(datasettotals[i])
        i += 1

    open_file.close()

    return datasettotals

def CalculateIndividualPreferences(dataset):

    '''
    Calculates the preference (-1 or +1) of each trait for all data and stores it in a matrix.
    '''
    
    datasetpreferences = np.ndarray(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        if row[0] >= 0:
            datasetpreferences[i][0] = 1
        elif row[0] < 0:
            datasetpreferences[i][0] = -1

        if row[1] >= 0:
            datasetpreferences[i][1] = 1
        elif row[1] < 0:
            datasetpreferences[i][1] = -1

        if row[2] >= 0:
            datasetpreferences[i][2] = 1
        elif row[2] < 0:
            datasetpreferences[i][2] = -1

        if row[3] >= 0:
            datasetpreferences[i][3] = 1
        elif row[3] < 0:
            datasetpreferences[i][3] = -1

        if row[4] >= 0:
            datasetpreferences[i][4] = 1
        elif row[4] < 0:
            datasetpreferences[i][4] = -1

        i += 1

    return datasetpreferences

def PredictExtroversion(file_out, allsampledatasettotals, allsampledatasetpreferences, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for columns not pertaining to the trait being predicted.
    Records the preferences of the trait as the correct labels
    Runs multiple regression algorithms to predict the trait
    '''

    datasettotals = allsampledatasettotals[:,1:] # Select all total columns but extroversion
    datasetpreferences = allsampledatasetpreferences[:,1:] # Select all preference columns but extroversion
    correctlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    correctlabels = np.transpose(correctlabels)
    
    testdatasettotals = alltestdatasettotals[:,1:] # Select all columns but extroversion
    testdatasetpreferences = alltestdatasetpreferences[:,1:] # Select all preference columns but extroversion
    testcorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    clf_percept = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept.fit(datasettotals, correctlabels, sample_weight=None)
    predictions = clf_percept.predict(datasettotals)
    testpredictions = clf_percept.predict(testdatasettotals)

    clf_percept2 = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept2.fit(datasetpreferences, correctlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(datasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testdatasetpreferences)
    
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd.fit(datasettotals, correctlabels)
    sgdpredictions = clf_sgd.predict(datasettotals)
    sgdtestpredictions = clf_sgd.predict(testdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd2.fit(datasetpreferences, correctlabels)
    sgdpreferencepredictions = clf_sgd2.predict(datasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testdatasetpreferences)
    
    clf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars.fit(datasettotals, correctlabels)
    larspredictions = clf_lars.predict(datasettotals)
    larstestpredictions = clf_lars.predict(testdatasettotals)
    
    clf_lars2 = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars2.fit(datasetpreferences, correctlabels)
    larspreferencepredictions = clf_lars2.predict(datasetpreferences)
    larstestpreferencepredictions = clf_lars2.predict(testdatasetpreferences)
    
    clf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic.fit(datasettotals, correctlabels)
    logisticpredictions = clf_logistic.predict(datasettotals)
    logistictestpredictions = clf_logistic.predict(testdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(datasetpreferences, correctlabels)
    logisticpreferencepredictions = clf_logistic2.predict(datasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testdatasetpreferences)
    
    clf_decisiontree = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree.fit(datasettotals, correctlabels)
    decisiontreepredictions = clf_decisiontree.predict(datasettotals)
    decisiontreetestpredictions = clf_decisiontree.predict(testdatasettotals)
    
    clf_decisiontree2 = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree2.fit(datasetpreferences, correctlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(datasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testdatasetpreferences)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in predictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in testpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in preferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in testpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in sgdpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in sgdtestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in sgdpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    mistakes = 0
    i = 0
    for prediction in sgdtestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1
    
    print("Able to predict extroversion based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1
    
    print("Able to predict extroversion based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training totals using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training totals using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing totals using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing totals using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual training preferences using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual training preferences using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on individual testing preferences using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on individual testing preferences using decisiontree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    testinglabels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    
    print(len(accuracies))
    print(len(labels))
    print(len(testinglabels))

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Extroversion Based on Personality:\nTraining Accuracies")
    plt2.set_title("Predicting Extroversion Based on Personality:\nTesting Accuracies")

    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()

    return

def PredictNeuroticism(file_out, allsampledatasettotals, allsampledatasetpreferences, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for columns not pertaining to the trait being predicted.
    Records the preferences of the trait as the correct labels
    Runs multiple regression algorithms to predict the trait
    '''

    datasettotals = np.delete(allsampledatasettotals, 1, axis=1) # Select all columns but neuroticism
    datasetpreferences = np.delete(allsampledatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    correctlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    
    testdatasettotals = np.delete(alltestdatasettotals, 1, axis=1) # Select all columns but neuroticism
    testdatasetpreferences = np.delete(alltestdatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    testcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns

    clf_percept = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept.fit(datasettotals, correctlabels, sample_weight=None)
    predictions = clf_percept.predict(datasettotals)
    testpredictions = clf_percept.predict(testdatasettotals)

    clf_percept2 = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept2.fit(datasetpreferences, correctlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(datasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testdatasetpreferences)
    
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd.fit(datasettotals, correctlabels)
    sgdpredictions = clf_sgd.predict(datasettotals)
    sgdtestpredictions = clf_sgd.predict(testdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd2.fit(datasetpreferences, correctlabels)
    sgdpreferencepredictions = clf_sgd2.predict(datasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testdatasetpreferences)
    
    clf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars.fit(datasettotals, correctlabels)
    larspredictions = clf_lars.predict(datasettotals)
    larstestpredictions = clf_lars.predict(testdatasettotals)
    
    clf_lars2 = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars2.fit(datasetpreferences, correctlabels)
    larspreferencepredictions = clf_lars2.predict(datasetpreferences)
    larstestpreferencepredictions = clf_lars2.predict(testdatasetpreferences)
    
    clf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic.fit(datasettotals, correctlabels)
    logisticpredictions = clf_logistic.predict(datasettotals)
    logistictestpredictions = clf_logistic.predict(testdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(datasetpreferences, correctlabels)
    logisticpreferencepredictions = clf_logistic2.predict(datasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testdatasetpreferences)
    
    clf_decisiontree = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree.fit(datasettotals, correctlabels)
    decisiontreepredictions = clf_decisiontree.predict(datasettotals)
    decisiontreetestpredictions = clf_decisiontree.predict(testdatasettotals)
    
    clf_decisiontree2 = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree2.fit(datasetpreferences, correctlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(datasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testdatasetpreferences)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in predictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in preferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    testinglabels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    
    print(len(accuracies))
    print(len(labels))
    print(len(testinglabels))

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Neuroticism Based on Personality:\nTraining Accuracies")
    plt2.set_title("Predicting Neuroticism Based on Personality:\nTesting Accuracies")

    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictAgreeableness(file_out, allsampledatasettotals, allsampledatasetpreferences, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for columns not pertaining to the trait being predicted.
    Records the preferences of the trait as the correct labels
    Runs multiple regression algorithms to predict the trait
    '''

    datasettotals = np.delete(allsampledatasettotals, 2, axis=1) # Select all columns but agreeableness
    datasetpreferences = np.delete(allsampledatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    correctlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    
    testdatasettotals = np.delete(alltestdatasettotals, 2, axis=1) # Select all columns but agreeableness
    testdatasetpreferences = np.delete(alltestdatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    testcorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns

    clf_percept = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept.fit(datasettotals, correctlabels, sample_weight=None)
    predictions = clf_percept.predict(datasettotals)
    testpredictions = clf_percept.predict(testdatasettotals)

    clf_percept2 = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept2.fit(datasetpreferences, correctlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(datasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testdatasetpreferences)
    
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd.fit(datasettotals, correctlabels)
    sgdpredictions = clf_sgd.predict(datasettotals)
    sgdtestpredictions = clf_sgd.predict(testdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd2.fit(datasetpreferences, correctlabels)
    sgdpreferencepredictions = clf_sgd2.predict(datasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testdatasetpreferences)
    
    clf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars.fit(datasettotals, correctlabels)
    larspredictions = clf_lars.predict(datasettotals)
    larstestpredictions = clf_lars.predict(testdatasettotals)
    
    clf_lars2 = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars2.fit(datasetpreferences, correctlabels)
    larspreferencepredictions = clf_lars2.predict(datasetpreferences)
    larstestpreferencepredictions = clf_lars2.predict(testdatasetpreferences)
    
    clf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic.fit(datasettotals, correctlabels)
    logisticpredictions = clf_logistic.predict(datasettotals)
    logistictestpredictions = clf_logistic.predict(testdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(datasetpreferences, correctlabels)
    logisticpreferencepredictions = clf_logistic2.predict(datasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testdatasetpreferences)
    
    clf_decisiontree = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree.fit(datasettotals, correctlabels)
    decisiontreepredictions = clf_decisiontree.predict(datasettotals)
    decisiontreetestpredictions = clf_decisiontree.predict(testdatasettotals)
    
    clf_decisiontree2 = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree2.fit(datasetpreferences, correctlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(datasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testdatasetpreferences)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in predictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in preferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    testinglabels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    
    print(len(accuracies))
    print(len(labels))
    print(len(testinglabels))

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Agreeableness Based on Personality:\nTraining Accuracies")
    plt2.set_title("Predicting Agreeableness Based on Personality:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictConscientiousness(file_out, allsampledatasettotals, allsampledatasetpreferences, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for columns not pertaining to the trait being predicted.
    Records the preferences of the trait as the correct labels
    Runs multiple regression algorithms to predict the trait
    '''

    datasettotals = np.delete(allsampledatasettotals, 3, axis=1) # Select all columns but conscientiousness
    datasetpreferences = np.delete(allsampledatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    correctlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    
    testdatasettotals = np.delete(alltestdatasettotals, 3, axis=1) # Select all columns but conscientiousness
    testdatasetpreferences = np.delete(alltestdatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    testcorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns

    clf_percept = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept.fit(datasettotals, correctlabels, sample_weight=None)
    predictions = clf_percept.predict(datasettotals)
    testpredictions = clf_percept.predict(testdatasettotals)

    clf_percept2 = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept2.fit(datasetpreferences, correctlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(datasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testdatasetpreferences)
    
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd.fit(datasettotals, correctlabels)
    sgdpredictions = clf_sgd.predict(datasettotals)
    sgdtestpredictions = clf_sgd.predict(testdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd2.fit(datasetpreferences, correctlabels)
    sgdpreferencepredictions = clf_sgd2.predict(datasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testdatasetpreferences)
    
    clf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars.fit(datasettotals, correctlabels)
    larspredictions = clf_lars.predict(datasettotals)
    larstestpredictions = clf_lars.predict(testdatasettotals)
    
    clf_lars2 = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars2.fit(datasetpreferences, correctlabels)
    larspreferencepredictions = clf_lars2.predict(datasetpreferences)
    larstestpreferencepredictions = clf_lars2.predict(testdatasetpreferences)
    
    clf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic.fit(datasettotals, correctlabels)
    logisticpredictions = clf_logistic.predict(datasettotals)
    logistictestpredictions = clf_logistic.predict(testdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(datasetpreferences, correctlabels)
    logisticpreferencepredictions = clf_logistic2.predict(datasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testdatasetpreferences)
    
    clf_decisiontree = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree.fit(datasettotals, correctlabels)
    decisiontreepredictions = clf_decisiontree.predict(datasettotals)
    decisiontreetestpredictions = clf_decisiontree.predict(testdatasettotals)
    
    clf_decisiontree2 = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree2.fit(datasetpreferences, correctlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(datasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testdatasetpreferences)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in predictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in preferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    testinglabels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    
    print(len(accuracies))
    print(len(labels))
    print(len(testinglabels))

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Conscientiousness Based on Personality:\nTraining Accuracies")
    plt2.set_title("Predicting Conscientiousness Based on Personality:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictOpenness(file_out, allsampledatasettotals, allsampledatasetpreferences, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for columns not pertaining to the trait being predicted.
    Records the preferences of the trait as the correct labels
    Runs multiple regression algorithms to predict the trait
    '''

    datasettotals = np.delete(allsampledatasettotals, 4, axis=1) # Select all columns but openness
    datasetpreferences = np.delete(allsampledatasetpreferences, 4, axis=1) # Select all columns but openness
    correctlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testdatasettotals = np.delete(alltestdatasettotals, 4, axis=1) # Select all columns but openness
    testdatasetpreferences = np.delete(alltestdatasetpreferences, 4, axis=1) # Select all columns but openness
    testcorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    clf_percept = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept.fit(datasettotals, correctlabels, sample_weight=None)
    predictions = clf_percept.predict(datasettotals)
    testpredictions = clf_percept.predict(testdatasettotals)

    clf_percept2 = Perceptron(max_iter=100, random_state=0, eta0=1)
    clf_percept2.fit(datasetpreferences, correctlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(datasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testdatasetpreferences)
    
    clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd.fit(datasettotals, correctlabels)
    sgdpredictions = clf_sgd.predict(datasettotals)
    sgdtestpredictions = clf_sgd.predict(testdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf_sgd2.fit(datasetpreferences, correctlabels)
    sgdpreferencepredictions = clf_sgd2.predict(datasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testdatasetpreferences)
    
    clf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars.fit(datasettotals, correctlabels)
    larspredictions = clf_lars.predict(datasettotals)
    larstestpredictions = clf_lars.predict(testdatasettotals)
    
    clf_lars2 = linear_model.LassoLars(alpha=1, max_iter=100)
    clf_lars2.fit(datasetpreferences, correctlabels)
    larspreferencepredictions = clf_lars2.predict(datasetpreferences)
    larstestpreferencepredictions = clf_lars2.predict(testdatasetpreferences)
    
    clf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic.fit(datasettotals, correctlabels)
    logisticpredictions = clf_logistic.predict(datasettotals)
    logistictestpredictions = clf_logistic.predict(testdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    clf_logistic2.fit(datasetpreferences, correctlabels)
    logisticpreferencepredictions = clf_logistic2.predict(datasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testdatasetpreferences)
    
    clf_decisiontree = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree.fit(datasettotals, correctlabels)
    decisiontreepredictions = clf_decisiontree.predict(datasettotals)
    decisiontreetestpredictions = clf_decisiontree.predict(testdatasettotals)
    
    clf_decisiontree2 = DecisionTreeRegressor(max_depth=5)
    clf_decisiontree2.fit(datasetpreferences, correctlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(datasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testdatasetpreferences)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in predictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in preferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in sgdtestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing totals using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larspreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in larstestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing preferences using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1
    
    print("Able to predict  openness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing totals using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logisticpreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in logistictestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing preferences using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual testing totals using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreepreferencepredictions:
        if correctlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict  openness based on individual training preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in decisiontreetestpreferencepredictions:
        if testcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict  openness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on individual testing preferences using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    testinglabels = ['Percept.\nTotals', 'Percept.\nPreferences', 'SGD\nTotals', 'SGD\nPreferences', 'Lars\nTotals', 'Lars\nPreferences', 'Logistic\nTotals', 'Logistic\nPreferences', 'D.T.\nTotals', 'D.T.\nPreferences']
    
    print(len(accuracies))
    print(len(labels))
    print(len(testinglabels))

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Openness Based on Personality:\nTraining Accuracies")
    plt2.set_title("Predicting Openness Based on Personality:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictBasedOnExtroversion(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for question columns of the trait being used for prediction.
    Records the preferences of the trait being predicted as the correct labels (this will happen for the four other traits)
    Runs multiple regression algorithms to predict each other trait
    '''

    extroversionquestions = allsampledataset[:,0:10] # Select extroversion question columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testextroversionquestions = alltestdataset[:,0:10] # Select extroversion question columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns


    #NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(extroversionquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(extroversionquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testextroversionquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    neuroticismclf_sgd.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(extroversionquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testextroversionquestions)
    
    neuroticismclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    neuroticismclf_lars.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismlarspredictions = neuroticismclf_lars.predict(extroversionquestions)
    neuroticismlarstestpredictions = neuroticismclf_lars.predict(testextroversionquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(extroversionquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testextroversionquestions)
    
    neuroticismclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    neuroticismclf_decisiontree.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(extroversionquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testextroversionquestions)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in neuroticismquestionpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testneuroticismquestionpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdtestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarspredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarstestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogisticpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogistictestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreepredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreetestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Neuroticism Based on Extroversion:\nTraining Accuracies")
    plt2.set_title("Predicting Neuroticism Based on Extroversion:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(extroversionquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(extroversionquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testextroversionquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    agreeablenessclf_sgd.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(extroversionquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testextroversionquestions)
    
    agreeablenessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    agreeablenessclf_lars.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesslarspredictions = agreeablenessclf_lars.predict(extroversionquestions)
    agreeablenesslarstestpredictions = agreeablenessclf_lars.predict(testextroversionquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(extroversionquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testextroversionquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    agreeablenessclf_decisiontree.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(extroversionquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testextroversionquestions)

    accuracies2 = []

    mistakes = 0
    i = 0
    for prediction in agreeablenessquestionpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)
    
    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testagreeablenessquestionpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdtestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarspredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarstestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogisticpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogistictestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreepredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreetestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies2)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies2[i])
        else:
            testingaccuracies.append(accuracies2[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Agreeableness Based on Extroversion:\nTraining Accuracies")
    plt2.set_title("Predicting Agreeableness Based on Extroversion:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(extroversionquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(extroversionquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testextroversionquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    conscientiousnessclf_sgd.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(extroversionquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testextroversionquestions)
    
    conscientiousnessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    conscientiousnessclf_lars.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesslarspredictions = conscientiousnessclf_lars.predict(extroversionquestions)
    conscientiousnesslarstestpredictions = conscientiousnessclf_lars.predict(testextroversionquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(extroversionquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testextroversionquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    conscientiousnessclf_decisiontree.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(extroversionquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testextroversionquestions)

    accuracies3 = []

    mistakes = 0
    i = 0
    for prediction in conscientiousnessquestionpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testconscientiousnessquestionpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdtestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarspredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarstestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogisticpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogistictestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreepredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreetestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies3)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies3[i])
        else:
            testingaccuracies.append(accuracies3[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Conscientiousness Based on Extroversion:\nTraining Accuracies")
    plt2.set_title("Predicting Conscientiousness Based on Extroversion:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    opennessclf_percept3.fit(extroversionquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(extroversionquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testextroversionquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    opennessclf_sgd.fit(extroversionquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(extroversionquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testextroversionquestions)
    
    opennessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    opennessclf_lars.fit(extroversionquestions, opennesscorrectlabels)
    opennesslarspredictions = opennessclf_lars.predict(extroversionquestions)
    opennesslarstestpredictions = opennessclf_lars.predict(testextroversionquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(extroversionquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(extroversionquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testextroversionquestions)
    
    opennessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    opennessclf_decisiontree.fit(extroversionquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(extroversionquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testextroversionquestions)

    accuracies4 = []

    mistakes = 0
    i = 0
    for prediction in opennessquestionpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testopennessquestionpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdtestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarspredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarstestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogisticpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogistictestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreepredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreetestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies4)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies4[i])
        else:
            testingaccuracies.append(accuracies4[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Openness Based on Extroversion:\nTraining Accuracies")
    plt2.set_title("Predicting Openness Based on Extroversion:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictBasedOnNeuroticism(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for question columns of the trait being used for prediction.
    Records the preferences of the trait being predicted as the correct labels (this will happen for the four other traits)
    Runs multiple regression algorithms to predict each other trait
    '''

    neuroticismquestions = allsampledataset[:,10:20] # Select neuroticism question columns
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testneuroticismquestions = alltestdataset[:,10:20] # Select neuroticism question columns
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    extroversionclf_percept3.fit(neuroticismquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(neuroticismquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testneuroticismquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    extroversionclf_sgd.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(neuroticismquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testneuroticismquestions)
    
    extroversionclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    extroversionclf_lars.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionlarspredictions = extroversionclf_lars.predict(neuroticismquestions)
    extroversionlarstestpredictions = extroversionclf_lars.predict(testneuroticismquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(neuroticismquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testneuroticismquestions)
    
    extroversionclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    extroversionclf_decisiontree.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(neuroticismquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testneuroticismquestions)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in extroversionquestionpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testextroversionquestionpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdtestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarspredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarstestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogisticpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogistictestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreepredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreetestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Extroversion Based on Neuroticism:\nTraining Accuracies")
    plt2.set_title("Predicting Extroversion Based on Neuroticism:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(neuroticismquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(neuroticismquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testneuroticismquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    agreeablenessclf_sgd.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(neuroticismquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testneuroticismquestions)
    
    agreeablenessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    agreeablenessclf_lars.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesslarspredictions = agreeablenessclf_lars.predict(neuroticismquestions)
    agreeablenesslarstestpredictions = agreeablenessclf_lars.predict(testneuroticismquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(neuroticismquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testneuroticismquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    agreeablenessclf_decisiontree.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(neuroticismquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testneuroticismquestions)

    accuracies2 = []

    mistakes = 0
    i = 0
    for prediction in agreeablenessquestionpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testagreeablenessquestionpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdtestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarspredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarstestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogisticpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogistictestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreepredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreetestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies2)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies2[i])
        else:
            testingaccuracies.append(accuracies2[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Agreeableness Based on Neuroticism:\nTraining Accuracies")
    plt2.set_title("Predicting Agreeableness Based on Neuroticism:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(neuroticismquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(neuroticismquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testneuroticismquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    conscientiousnessclf_sgd.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(neuroticismquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testneuroticismquestions)
    
    conscientiousnessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    conscientiousnessclf_lars.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesslarspredictions = conscientiousnessclf_lars.predict(neuroticismquestions)
    conscientiousnesslarstestpredictions = conscientiousnessclf_lars.predict(testneuroticismquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(neuroticismquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testneuroticismquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    conscientiousnessclf_decisiontree.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(neuroticismquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testneuroticismquestions)

    accuracies3 = []

    mistakes = 0
    i = 0
    for prediction in conscientiousnessquestionpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testconscientiousnessquestionpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdtestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarspredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarstestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogisticpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogistictestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreepredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreetestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies3)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies3[i])
        else:
            testingaccuracies.append(accuracies3[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Conscientiousness Based on Neuroticism:\nTraining Accuracies")
    plt2.set_title("Predicting Conscientiousness Based on Neuroticism:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    opennessclf_percept3.fit(neuroticismquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(neuroticismquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testneuroticismquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    opennessclf_sgd.fit(neuroticismquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(neuroticismquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testneuroticismquestions)
    
    opennessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    opennessclf_lars.fit(neuroticismquestions, opennesscorrectlabels)
    opennesslarspredictions = opennessclf_lars.predict(neuroticismquestions)
    opennesslarstestpredictions = opennessclf_lars.predict(testneuroticismquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(neuroticismquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(neuroticismquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testneuroticismquestions)
    
    opennessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    opennessclf_decisiontree.fit(neuroticismquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(neuroticismquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testneuroticismquestions)

    accuracies4 = []

    mistakes = 0
    i = 0
    for prediction in opennessquestionpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testopennessquestionpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdtestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarspredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarstestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogisticpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogistictestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreepredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreetestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on neuroticism testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies4)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies4[i])
        else:
            testingaccuracies.append(accuracies4[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Openness Based on Neuroticism:\nTraining Accuracies")
    plt2.set_title("Predicting Openness Based on Neuroticism:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictBasedOnAgreeableness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    '''
    Parses a matrix with all data for question columns of the trait being used for prediction.
    Records the preferences of the trait being predicted as the correct labels (this will happen for the four other traits)
    Runs multiple regression algorithms to predict each other trait
    '''

    agreeablenessquestions = allsampledataset[:,20:30] # Select agreeableness question columns
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testagreeablenessquestions = alltestdataset[:,20:30] # Select agreeableness question columns
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    extroversionclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    extroversionclf_percept3.fit(agreeablenessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(agreeablenessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testagreeablenessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    extroversionclf_sgd.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(agreeablenessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testagreeablenessquestions)
    
    extroversionclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    extroversionclf_lars.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversionlarspredictions = extroversionclf_lars.predict(agreeablenessquestions)
    extroversionlarstestpredictions = extroversionclf_lars.predict(testagreeablenessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(agreeablenessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testagreeablenessquestions)
    
    extroversionclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    extroversionclf_decisiontree.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(agreeablenessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testagreeablenessquestions)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in extroversionquestionpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testextroversionquestionpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdtestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarspredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarstestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogisticpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogistictestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreepredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreetestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Extroversion Based on Agreeableness:\nTraining Accuracies")
    plt2.set_title("Predicting Extroversion Based on Agreeableness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(agreeablenessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(agreeablenessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testagreeablenessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    neuroticismclf_sgd.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(agreeablenessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testagreeablenessquestions)
    
    neuroticismclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    neuroticismclf_lars.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismlarspredictions = neuroticismclf_lars.predict(agreeablenessquestions)
    neuroticismlarstestpredictions = neuroticismclf_lars.predict(testagreeablenessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(agreeablenessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testagreeablenessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    neuroticismclf_decisiontree.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(agreeablenessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testagreeablenessquestions)

    accuracies2 = []

    mistakes = 0
    i = 0
    for prediction in neuroticismquestionpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testneuroticismquestionpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdtestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarspredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarstestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogisticpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogistictestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreepredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreetestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies2)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies2[i])
        else:
            testingaccuracies.append(accuracies2[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Neuroticism Based on Agreeableness:\nTraining Accuracies")
    plt2.set_title("Predicting Neuroticism Based on Agreeableness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(agreeablenessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(agreeablenessquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testagreeablenessquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    conscientiousnessclf_sgd.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(agreeablenessquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testagreeablenessquestions)
    
    conscientiousnessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    conscientiousnessclf_lars.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslarspredictions = conscientiousnessclf_lars.predict(agreeablenessquestions)
    conscientiousnesslarstestpredictions = conscientiousnessclf_lars.predict(testagreeablenessquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(agreeablenessquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testagreeablenessquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    conscientiousnessclf_decisiontree.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(agreeablenessquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testagreeablenessquestions)

    accuracies3 = []

    mistakes = 0
    i = 0
    for prediction in conscientiousnessquestionpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testconscientiousnessquestionpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdtestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarspredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarstestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogisticpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogistictestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreepredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreetestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies3)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies3[i])
        else:
            testingaccuracies.append(accuracies3[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Conscientiousness Based on Agreeableness:\nTraining Accuracies")
    plt2.set_title("Predicting Conscientiousness Based on Agreeableness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    opennessclf_percept3.fit(agreeablenessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(agreeablenessquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testagreeablenessquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    opennessclf_sgd.fit(agreeablenessquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(agreeablenessquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testagreeablenessquestions)
    
    opennessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    opennessclf_lars.fit(agreeablenessquestions, opennesscorrectlabels)
    opennesslarspredictions = opennessclf_lars.predict(agreeablenessquestions)
    opennesslarstestpredictions = opennessclf_lars.predict(testagreeablenessquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(agreeablenessquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(agreeablenessquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testagreeablenessquestions)
    
    opennessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    opennessclf_decisiontree.fit(agreeablenessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(agreeablenessquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testagreeablenessquestions)

    accuracies4 = []

    mistakes = 0
    i = 0
    for prediction in opennessquestionpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testopennessquestionpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdtestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarspredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarstestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogisticpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogistictestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreepredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreetestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on agreeableness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies4)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies4[i])
        else:
            testingaccuracies.append(accuracies4[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Openness Based on Agreeableness:\nTraining Accuracies")
    plt2.set_title("Predicting Openness Based on Agreeableness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictBasedOnConscientiousness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):
    
    '''
    Parses a matrix with all data for question columns of the trait being used for prediction.
    Records the preferences of the trait being predicted as the correct labels (this will happen for the four other traits)
    Runs multiple regression algorithms to predict each other trait
    '''

    conscientiousnessquestions = allsampledataset[:,30:40] # Select conscientiousness question columns
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testconscientiousnessquestions = alltestdataset[:,30:40] # Select conscientiousness question columns
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    extroversionclf_percept3.fit(conscientiousnessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(conscientiousnessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testconscientiousnessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    extroversionclf_sgd.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(conscientiousnessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testconscientiousnessquestions)
    
    extroversionclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    extroversionclf_lars.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversionlarspredictions = extroversionclf_lars.predict(conscientiousnessquestions)
    extroversionlarstestpredictions = extroversionclf_lars.predict(testconscientiousnessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(conscientiousnessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testconscientiousnessquestions)
    
    extroversionclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    extroversionclf_decisiontree.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(conscientiousnessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testconscientiousnessquestions)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in extroversionquestionpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testextroversionquestionpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdtestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarspredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarstestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogisticpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogistictestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreepredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreetestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Extroversion Based on Conscientiousness:\nTraining Accuracies")
    plt2.set_title("Predicting Extroversion Based on Conscientiousness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(conscientiousnessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(conscientiousnessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testconscientiousnessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    neuroticismclf_sgd.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(conscientiousnessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testconscientiousnessquestions)
    
    neuroticismclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    neuroticismclf_lars.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismlarspredictions = neuroticismclf_lars.predict(conscientiousnessquestions)
    neuroticismlarstestpredictions = neuroticismclf_lars.predict(testconscientiousnessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(conscientiousnessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testconscientiousnessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    neuroticismclf_decisiontree.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(conscientiousnessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testconscientiousnessquestions)

    accuracies2 = []

    mistakes = 0
    i = 0
    for prediction in neuroticismquestionpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testneuroticismquestionpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdtestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarspredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarstestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogisticpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogistictestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreepredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreetestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies2)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies2[i])
        else:
            testingaccuracies.append(accuracies2[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Neuroticism Based on Conscientiousness:\nTraining Accuracies")
    plt2.set_title("Predicting Neuroticism Based on Conscientiousness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(conscientiousnessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(conscientiousnessquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testconscientiousnessquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    agreeablenessclf_sgd.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(conscientiousnessquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testconscientiousnessquestions)
    
    agreeablenessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    agreeablenessclf_lars.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesslarspredictions = agreeablenessclf_lars.predict(conscientiousnessquestions)
    agreeablenesslarstestpredictions = agreeablenessclf_lars.predict(testconscientiousnessquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(conscientiousnessquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testconscientiousnessquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    agreeablenessclf_decisiontree.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(conscientiousnessquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testconscientiousnessquestions)

    accuracies3 = []

    mistakes = 0
    i = 0
    for prediction in agreeablenessquestionpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testagreeablenessquestionpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdtestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarspredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarstestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogisticpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogistictestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreepredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreetestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies3)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies3[i])
        else:
            testingaccuracies.append(accuracies3[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Agreeableness Based on Conscientiousness:\nTraining Accuracies")
    plt2.set_title("Predicting Agreeableness Based on Conscientiousness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    opennessclf_percept3.fit(conscientiousnessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(conscientiousnessquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testconscientiousnessquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    opennessclf_sgd.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(conscientiousnessquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testconscientiousnessquestions)
    
    opennessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    opennessclf_lars.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennesslarspredictions = opennessclf_lars.predict(conscientiousnessquestions)
    opennesslarstestpredictions = opennessclf_lars.predict(testconscientiousnessquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(conscientiousnessquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testconscientiousnessquestions)
    
    opennessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    opennessclf_decisiontree.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(conscientiousnessquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testconscientiousnessquestions)

    accuracies4 = []

    mistakes = 0
    i = 0
    for prediction in opennessquestionpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testopennessquestionpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesssgdtestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarspredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslarstestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogisticpredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennesslogistictestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreepredictions:
        if opennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in opennessdecisiontreetestpredictions:
        if testopennesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict openness based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict openness based on conscientiousness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies4)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies4[i])
        else:
            testingaccuracies.append(accuracies4[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Openness Based on Conscientiousness:\nTraining Accuracies")
    plt2.set_title("Predicting Openness Based on Conscientiousness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def PredictBasedOnOpenness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):
    
    '''
    Parses a matrix with all data for question columns of the trait being used for prediction.
    Records the preferences of the trait being predicted as the correct labels (this will happen for the four other traits)
    Runs multiple regression algorithms to predict each other trait
    '''

    opennessquestions = allsampledataset[:,40:50] # Select openness question columns
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    
    testopennessquestions = alltestdataset[:,40:50] # Select openness question columns
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    extroversionclf_percept3.fit(opennessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(opennessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testopennessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    extroversionclf_sgd.fit(opennessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(opennessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testopennessquestions)
    
    extroversionclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    extroversionclf_lars.fit(opennessquestions, extroversioncorrectlabels)
    extroversionlarspredictions = extroversionclf_lars.predict(opennessquestions)
    extroversionlarstestpredictions = extroversionclf_lars.predict(testopennessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(opennessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(opennessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testopennessquestions)
    
    extroversionclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    extroversionclf_decisiontree.fit(opennessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(opennessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testopennessquestions)

    accuracies = []

    mistakes = 0
    i = 0
    for prediction in extroversionquestionpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testextroversionquestionpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionsgdtestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarspredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlarstestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogisticpredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversionlogistictestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreepredictions:
        if extroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in extroversiondecisiontreetestpredictions:
        if testextroversioncorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict extroversion based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict extroversion based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies[i])
        else:
            testingaccuracies.append(accuracies[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Extroversion Based on Openness:\nTraining Accuracies")
    plt2.set_title("Predicting Extroversion Based on Openness:\nTesting Accuracies")

    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    neuroticismclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(opennessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(opennessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testopennessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    neuroticismclf_sgd.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(opennessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testopennessquestions)
    
    neuroticismclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    neuroticismclf_lars.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismlarspredictions = neuroticismclf_lars.predict(opennessquestions)
    neuroticismlarstestpredictions = neuroticismclf_lars.predict(testopennessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(opennessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testopennessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    neuroticismclf_decisiontree.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(opennessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testopennessquestions)

    accuracies2 = []

    mistakes = 0
    i = 0
    for prediction in neuroticismquestionpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testneuroticismquestionpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismsgdtestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarspredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlarstestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogisticpredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismlogistictestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreepredictions:
        if neuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in neuroticismdecisiontreetestpredictions:
        if testneuroticismcorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict neuroticism based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict neuroticism based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies2.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies2)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies2[i])
        else:
            testingaccuracies.append(accuracies2[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Neuroticism Based on Openness:\nTraining Accuracies")
    plt2.set_title("Predicting Neuroticism Based on Openness:\nTesting Accuracies")

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(opennessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(opennessquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testopennessquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    agreeablenessclf_sgd.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(opennessquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testopennessquestions)
    
    agreeablenessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    agreeablenessclf_lars.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenesslarspredictions = agreeablenessclf_lars.predict(opennessquestions)
    agreeablenesslarstestpredictions = agreeablenessclf_lars.predict(testopennessquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(opennessquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testopennessquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    agreeablenessclf_decisiontree.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(opennessquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testopennessquestions)

    accuracies3 = []

    mistakes = 0
    i = 0
    for prediction in agreeablenessquestionpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testagreeablenessquestionpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesssgdtestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarspredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslarstestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogisticpredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenesslogistictestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreepredictions:
        if agreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in agreeablenessdecisiontreetestpredictions:
        if testagreeablenesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict agreeableness based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict agreeableness based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies3.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies3)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies3[i])
        else:
            testingaccuracies.append(accuracies3[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Agreeableness Based on Openness:\nTraining Accuracies")
    plt2.set_title("Predicting Agreeableness Based on Openness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=100, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(opennessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(opennessquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testopennessquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    conscientiousnessclf_sgd.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(opennessquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testopennessquestions)
    
    conscientiousnessclf_lars = linear_model.LassoLars(alpha=1, max_iter=100)
    conscientiousnessclf_lars.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslarspredictions = conscientiousnessclf_lars.predict(opennessquestions)
    conscientiousnesslarstestpredictions = conscientiousnessclf_lars.predict(testopennessquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=100, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(opennessquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testopennessquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeRegressor(max_depth=5)
    conscientiousnessclf_decisiontree.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(opennessquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testopennessquestions)

    accuracies4 = []

    mistakes = 0
    i = 0
    for prediction in conscientiousnessquestionpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness training questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in testconscientiousnessquestionpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness testing questions using perceptron with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness training questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesssgdtestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarspredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness training questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslarstestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness testing questions using lars with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogisticpredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness training questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnesslogistictestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness testing questions using logistic with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreepredictions:
        if conscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness training questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))
    
    mistakes = 0
    i = 0
    for prediction in conscientiousnessdecisiontreetestpredictions:
        if testconscientiousnesscorrectlabels[i] * prediction <= 0:
            mistakes += 1
        i += 1

    print("Able to predict conscientiousness based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))))
    print("Able to predict conscientiousness based on openness testing questions using decision tree with %{} accuracy".format(1 - (mistakes/(i + 1))), file=file_out)

    accuracies4.append((1 - (mistakes/(i + 1))))

    labels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']
    testinglabels = ['Percept.', 'SGD', 'Lars', 'Logistic', 'D.T.']

    trainingaccuracies = []
    testingaccuracies = []

    for i in range(len(accuracies4)):
        throwaway, remain = divmod(i, 2)
        if remain == 0: # Place every other accuracy into corresponding accuracy list for plotting
            trainingaccuracies.append(accuracies4[i])
        else:
            testingaccuracies.append(accuracies4[i])

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2) # Set up graph for training and testing plots
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])

    plt1.set_title("Predicting Conscientiousness Based on Openness:\nTraining Accuracies")
    plt2.set_title("Predicting Conscientiousness Based on Openness:\nTesting Accuracies")
    plt1.bar(labels, trainingaccuracies)
    plt2.bar(testinglabels, testingaccuracies)

    plt1.set_ylim([(min(trainingaccuracies) - 0.05), (max(trainingaccuracies) + 0.05)])
    plt2.set_ylim([(min(testingaccuracies) - 0.05), (max(testingaccuracies) + 0.05)])

    plt.show()
    
    return

def ClusteringExtroversionNeuroticism(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    extroversion = dataset[:,0]
    neuroticism = dataset[:,1]

    datasettoscale = (np.array([extroversion, neuroticism])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringExtroversionAgreeableness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    extroversion = dataset[:,0]
    agreeableness = dataset[:,2]

    datasettoscale = (np.array([extroversion, agreeableness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringExtroversionConscientiousness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    extroversion = dataset[:,0]
    conscientiousness = dataset[:,3]

    datasettoscale = (np.array([extroversion, conscientiousness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringExtroversionOpenness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    extroversion = dataset[:,0]
    openness = dataset[:,4]

    datasettoscale = (np.array([extroversion, openness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringNeuroticismAgreeableness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    neuroticism = dataset[:,1]
    agreeableness = dataset[:,2]

    datasettoscale = (np.array([neuroticism, agreeableness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringNeuroticismConscientiousness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    neuroticism = dataset[:,1]
    conscientiousness = dataset[:,3]

    datasettoscale = (np.array([neuroticism, conscientiousness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringNeuroticismOpenness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    neuroticism = dataset[:,1]
    openness = dataset[:,4]

    datasettoscale = (np.array([neuroticism, openness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringAgreeablenessConscientiousness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    agreeableness = dataset[:,2]
    conscientiousness = dataset[:,3]

    datasettoscale = (np.array([agreeableness, conscientiousness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringAgreeablenessOpenness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    agreeableness = dataset[:,2]
    openness = dataset[:,4]

    datasettoscale = (np.array([agreeableness, openness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def ClusteringConscientiousnessOpenness(dataset):

    '''
    Makes clusters based on two given traits.
    '''

    # Set up six plots for displaying six clustering methods
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    plt1 = plt.subplot(G[0, 0])
    plt2 = plt.subplot(G[0, 1])
    plt3 = plt.subplot(G[0, 2])
    plt4 = plt.subplot(G[1, 0])
    plt5 = plt.subplot(G[1, 1])
    plt6 = plt.subplot(G[1, 2])

    conscientiousness = dataset[:,3]
    openness = dataset[:,4]

    datasettoscale = (np.array([conscientiousness, openness])).transpose()

    scaleddataset = StandardScaler().fit_transform(datasettoscale)

    # Compute DBSCAN
    db = DBSCAN(eps=25, min_samples=0.25).fit(scaleddataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    cmap = plt.cm.get_cmap("Spectral")

    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = scaleddataset[class_member_mask & core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = scaleddataset[class_member_mask & ~core_samples_mask]
        plt1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt1.set_title('Clustering at 25 epsilon cut\nNon-optics DBSCAN')

    stacks = np.vstack(datasettoscale)

    clusters = OPTICS(min_samples=0.25, xi=.05, min_cluster_size=0.25)

    clusters2 = OPTICS(min_samples = 0.25, xi=0.2, min_cluster_size = 0.25)

    clusters.fit(stacks)
    clusters2.fit(stacks)

    labels_050 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=25)
    labels_200 = cluster_optics_dbscan(reachability=clusters.reachability_, core_distances=clusters.core_distances_, ordering=clusters.ordering_, eps=50)

    labels = clusters.labels_[clusters.ordering_]

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters2.labels_ == klass]
        plt2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt2.plot(stacks[clusters2.labels_ == -1, 0], stacks[clusters2.labels_ == -1, 1], 'k+', alpha=0.1)
    plt2.set_title('Automatic Clustering\nOPTICS')


    # OPTICS2
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = stacks[clusters.labels_ == klass]
        plt3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt3.plot(stacks[clusters.labels_ == -1, 0], stacks[clusters.labels_ == -1, 1], 'k+', alpha=0.1)
    plt3.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 25
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = stacks[labels_050 == klass]
        plt4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    plt4.plot(stacks[labels_050 == -1, 0], stacks[labels_050 == -1, 1], 'k+', alpha=0.1)
    plt4.set_title('Clustering at 25 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = stacks[labels_200 == klass]
        plt5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt5.plot(stacks[labels_200 == -1, 0], stacks[labels_200 == -1, 1], 'k+', alpha=0.1)
    plt5.set_title('Clustering at 50 epsilon cut\nDBSCAN')

    # WARD
    print("Compute structured hierarchical clustering...")
    st = time()
    n_clusters = 4
    ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',)
    ward.fit(datasettoscale)
    print("Elapsed time: ", time() - st)

    #plt6.axis('equal')
    plt6.scatter(datasettoscale[:,0], datasettoscale[:,1], c=ward.labels_, cmap='rainbow')
    plt6.set_title("Ward clustering with four clusters")

    plt.tight_layout()
    plt.show()

    return

def main():

    Shrinker()
    dataset, sampledataset, testdataset = ReadInData()
    sampledatasettotals = CalculateIndividualTotals(sampledataset)
    testdatasettotals = CalculateIndividualTotals(testdataset)
    datasettotals = CalculateIndividualTotals(dataset)
    sampledatasetpreferences = CalculateIndividualPreferences(sampledatasettotals)
    testdatasetpreferences = CalculateIndividualPreferences(testdatasettotals)

    print("Done calculating totals and preferences...")

    ClusteringExtroversionNeuroticism(datasettotals)
    ClusteringExtroversionAgreeableness(datasettotals)
    ClusteringExtroversionConscientiousness(datasettotals)
    ClusteringExtroversionOpenness(datasettotals)
    ClusteringNeuroticismAgreeableness(datasettotals)
    ClusteringNeuroticismConscientiousness(datasettotals)
    ClusteringNeuroticismOpenness(datasettotals)
    ClusteringAgreeablenessConscientiousness(datasettotals)
    ClusteringAgreeablenessOpenness(datasettotals)
    ClusteringConscientiousnessOpenness(datasettotals)

    file_out = open('./output.txt', 'w')

    print("", file=file_out)

    PredictExtroversion(file_out, sampledatasettotals, sampledatasetpreferences, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictNeuroticism(file_out, sampledatasettotals, sampledatasetpreferences, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictAgreeableness(file_out, sampledatasettotals, sampledatasetpreferences, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictConscientiousness(file_out, sampledatasettotals, sampledatasetpreferences, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictOpenness(file_out, sampledatasettotals, sampledatasetpreferences, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)

    print("", file=file_out)
    print("", file=file_out)

    PredictBasedOnExtroversion(file_out, sampledataset, sampledatasettotals, sampledatasetpreferences, testdataset, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictBasedOnNeuroticism(file_out, sampledataset, sampledatasettotals, sampledatasetpreferences, testdataset, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictBasedOnAgreeableness(file_out, sampledataset, sampledatasettotals, sampledatasetpreferences, testdataset, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictBasedOnConscientiousness(file_out, sampledataset, sampledatasettotals, sampledatasetpreferences, testdataset, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)
    PredictBasedOnOpenness(file_out, sampledataset, sampledatasettotals, sampledatasetpreferences, testdataset, testdatasettotals, testdatasetpreferences)
    print("", file=file_out)

    print("", file=file_out)
    print("", file=file_out)
    
    file_out.close()

    return

main()
