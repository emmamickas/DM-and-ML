#Emma Mickas
#ID: 011651116
#CptS437 Project

import csv
from time import time

import numpy as np
from sklearn import linear_model, metrics
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA

allUsersQuestions = {}
allQuestions = {}
allQuestionNames = []

def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def CleanDataForReading():
    open_file = open('data-final.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Filtered.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 1

    for line in file_in:
        if i == 1:
            i += 1
        if 'NULL' not in line[:50]:
            file_out.writerow(line[:50])
    
    open_file.close()
    open_file2.close()

def TestingShrinker():
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Testing.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0

    for line in file_in:
        if i == 0:
            i += 1
            continue
        file_out.writerow(line[:100])
        i += 1
        if i > 10:
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

    trainingdataset = dataset[0:7]
    testdataset = dataset[7:10]

    return dataset, trainingdataset, testdataset

def Shrinker():
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = '\t')
    open_file2 = open('data-final - Shrunken.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file2, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    i = 0
    linecount = 0
    start = 0

    for line in file_in:
        if start == 0:
            start += 1
            continue
        file_out.writerow(line[:100])
        linecount += 1
        i += 1
        if linecount >= 10:
            break
    
    open_file.close()
    open_file2.close()
    return

def ReadInData():

    #open_file = open('data-final - Shrunken.csv', 'r', encoding='utf-8', newline='') # Shrunken version of the dataset for ease of testing
    open_file = open('data-final - Filtered.csv', 'r', encoding='utf-8', newline='') # Full dataset after first cleaning (filtering out unnecessary values)
    fulldataset = np.loadtxt(open_file, delimiter='\t', usecols=(range(50)), max_rows=1015000, skiprows=1) # Read in all the data, skip the header

    print(fulldataset.shape)

    fulllength, throwaway = fulldataset.shape
    #split = int(fulllength) * 0.67 # Manual split method, makes testing a bit easier
    #sampledataset = fulldataset[0:int(split)] # Manual split method, makes testing a bit easier
    #testdataset = fulldataset[int(split):int(fulllength)] # Manual split method, makes testing a bit easier

    open_file.close()

    dataset = resample(fulldataset, n_samples=10000, replace=False) # Get only 10000 samples
    sampledataset, testdataset = train_test_split(dataset, test_size=0.33, random_state=1) # Split into training and testing

    print("Done reading in data...")

    return dataset, sampledataset, testdataset

def CleanDataForConsistency(dataset):
    cleaneddataset = np.zeros(shape=(dataset.shape[0], 50))

    i = 0

    # Scale each value so that more positive means more fitting the personality trait

    for row in dataset:
        for j in range(50):
            if j >= 0 and j < 10: # Extroversion question
                if (j % 2) == 0: # Even numbered question, higher answer = more extroverted
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less extroverted
                    cleaneddataset[i][j] = 5 - row[j]
            elif j >= 10 and j < 20: # Neuroticism question
                if j == 10:
                    cleaneddataset[i][j] = row[j]
                elif j == 11:
                    cleaneddataset[i][j] = 5 - row[j]
                elif j == 12:
                    cleaneddataset[i][j] = row[j]
                elif j == 13:
                    cleaneddataset[i][j] = 5 - row[j]
                else:
                    cleaneddataset[i][j] = row[j]
            elif  j >= 20 and j < 30: # Agreeableness question
                if (j % 2) == 0: # Even numbered question, higher answer = less agreeable
                    cleaneddataset[i][j] = 5 - row[j]
                else: # Odd numbered question, higher answer = more agreeable
                    cleaneddataset[i][j] = row[j]
            elif j >= 30 and j < 40: # Conscientiousness question
                if (j % 2) == 0: # Even numbered question, higher answer = more conscientious
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less conscientious
                    cleaneddataset[i][j] = 5 - row[j]
            elif j >= 40 and j < 50: # Openness question
                if (j == 47):
                    cleaneddataset[i][j] = row[j]
                elif (j == 48):
                    cleaneddataset[i][j] = 5 - row[j]
                elif (j == 49):
                    cleaneddataset[i][j] = row[j]
                elif (j % 2) == 0: # Even numbered question, higher answer = more open
                    cleaneddataset[i][j] = row[j]
                else: # Odd numbered question, higher answer = less open
                    cleaneddataset[i][j] = 5 - row[j]

        i += 1

    return cleaneddataset

def NormalizeIndividualQuestionsData(dataset):
    normalized_array = np.column_stack((normalize(dataset[:, 0:10], norm="l1"), normalize(dataset[:, 10:20], norm="l1"), normalize(dataset[:, 20:30], norm="l1"), normalize(dataset[:, 30:40], norm="l1"), normalize(dataset[:, 40:50], norm="l1")))
    return normalized_array

def NormalizeData(dataset):
    normalized_array = normalize(dataset, norm="l1")
    return normalized_array

def CalculateExtroversion(row):
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
    total = 0
    total += row[10]
    total -= row[11]
    total += row[12]
    total -= row[13]
    total += row[14]
    total += row[15]
    total += row[16]
    total += row[17]
    total += row[18]
    total += row[19]
    return total

def CalculateAgreeableness(row):
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
    total = 0
    total += row[40]
    total -= row[41]
    total += row[42]
    total -= row[43]
    total += row[44]
    total -= row[45]
    total += row[46]
    total += row[47]
    total -= row[48]
    total += row[49]
    return total

def CalculateCleanedTotals(dataset):
    datasettotals = np.zeros(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        for j in range(0, 10):
            datasettotals[i][0] += row[j]
        for j in range(10, 20):
            datasettotals[i][1] += row[j]
        for j in range(20, 30):
            datasettotals[i][2] += row[j]
        for j in range(30, 40):
            datasettotals[i][3] += row[j]
        for j in range(40, 50):
            datasettotals[i][4] += row[j]
        i += 1

    return datasettotals

def CalculateIndividualTotals(dataset):
    open_file = open('data-final - IndividualTotals.csv', 'w', encoding='utf-8', newline='')
    file_out = csv.writer(open_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    datasettotals = np.zeros(shape=(dataset.shape[0], 5))

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
    datasetpreferences = np.zeros(shape=(dataset.shape[0], 5))

    i = 0

    for row in dataset:
        if row[0] >= 0:
            datasetpreferences[i][0] = 1
        elif row[0] < 0:
            datasetpreferences[i][0] = -1

        if row[1] >= 15:
            datasetpreferences[i][1] = 1
        elif row[1] < 15:
            datasetpreferences[i][1] = -1

        if row[2] >= 0:
            datasetpreferences[i][2] = 1
        elif row[2] < 0:
            datasetpreferences[i][2] = -1

        if row[3] >= 0:
            datasetpreferences[i][3] = 1
        elif row[3] < 0:
            datasetpreferences[i][3] = -1

        if row[4] >= 5:
            datasetpreferences[i][4] = 1
        elif row[4] < 5:
            datasetpreferences[i][4] = -1

        i += 1

    return datasetpreferences

def CalculateCleanedPreferences(datasettotals):
    datasetpreferences = np.zeros(shape=(datasettotals.shape[0], 5))

    i = 0

    for row in datasettotals:
        if row[0] >= 25:
            datasetpreferences[i][0] = 1
        elif row[0] < 25:
            datasetpreferences[i][0] = -1

        if row[1] >= 25:
            datasetpreferences[i][1] = 1
        elif row[1] < 25:
            datasetpreferences[i][1] = -1

        if row[2] >= 25:
            datasetpreferences[i][2] = 1
        elif row[2] < 25:
            datasetpreferences[i][2] = -1

        if row[3] >= 25:
            datasetpreferences[i][3] = 1
        elif row[3] < 25:
            datasetpreferences[i][3] = -1

        if row[4] >= 25:
            datasetpreferences[i][4] = 1
        elif row[4] < 25:
            datasetpreferences[i][4] = -1

        i += 1

    return datasetpreferences

def PerceptronForPruning(dataset, datasetpreferences, characteristicstoprune, characteristictopredict, iterations, learningrate, numtopruneto):
    weights = [0] * 10
    weightSums = [0] * 10

    count = 0

    datasetsamples = dataset[:,characteristicstoprune[0]:characteristicstoprune[1]]

    datasetlabels = datasetpreferences[:,characteristictopredict]

    file_out = open('perceptron_output.txt', 'a')

    for iteration in range(iterations):
        sampleNumber = 0 # Reset example number for the next iteration
        for sample in datasetsamples:
            if (datasetlabels[sampleNumber]) * (np.dot(weights, sample)) <= 0: # Mistake!
                weights = weights + np.multiply((learningrate * datasetlabels[sampleNumber]), sample) # Update weight
                weightSums = np.add(weightSums, weights) # Update weight total to calculate average weight later
                count += 1 # Keep track of total number of weights to calculate average weight later
            sampleNumber += 1

    averageWeights = np.divide(weightSums, count)

    file_out.close()

    maxWeightIndexes = [-1] * numtopruneto

    for i in range(numtopruneto):
        curMaxWeight = -999999
        curMaxWeightIndex = -1
        for j in range(len(weights)):
            #print("curMaxWeight:", curMaxWeight)
            #print("abs(weight[j]):", abs(weight[j]))
            if j in maxWeightIndexes:
                continue
            if (abs(weights[j]) > curMaxWeight):
                curMaxWeight = abs(weights[j])
                curMaxWeightIndex = j
        maxWeightIndexes[i] = curMaxWeightIndex

    print("maxWeightIndexes:", maxWeightIndexes)

    selectedweights = [0] * numtopruneto

    for i in range(numtopruneto):
        selectedweights[i] = weights[maxWeightIndexes[i]]

    print("Selectedweights:", selectedweights)

    return maxWeightIndexes, weights, averageWeights

def PerceptronForPruningMultiplePredictions(dataset, datasetpreferences, characteristicstoprune, characteristicstopredict, iterations, learningrate, numtopruneto):
    weightSums = {}
    averageWeights = {}

    count = 0

    datasetsamples = dataset[:,characteristicstoprune[0]:characteristicstoprune[1]]

    for indextopredict in characteristicstopredict:
        weightSums[indextopredict] = [0] * 10 # Keep track of the weights used for predicting this trait
        datasetlabels = datasetpreferences[:,indextopredict]
        weights = [0] * 10

        file_out = open('perceptron_output.txt', 'a')

        for iteration in range(iterations):
            sampleNumber = 0 # Reset example number for the next iteration
            for sample in datasetsamples:
                if (datasetlabels[sampleNumber]) * (np.dot(weights, sample)) <= 0: # Mistake!
                    weights = weights + np.multiply((learningrate * datasetlabels[sampleNumber]), sample) # Update weight
                    weightSums[indextopredict] = np.add(weightSums[indextopredict], weights) # Update weight total to calculate average weight later
                    count += 1 # Keep track of total number of weights to calculate average weight later

                sampleNumber += 1

        averageWeights[indextopredict] = np.divide(weightSums[indextopredict], count)

    file_out.close()

    allweights = [0] * 10
    for indexpredicted, predictionweights in averageWeights.items():
        for i in range(len(predictionweights)):
            allweights[i] += abs(predictionweights[i])
    
    maxweightindexes = [-1] * numtopruneto
    for i in range(numtopruneto):
        curMaxWeight = -999999
        curMaxWeightIndex = -1
        for j in range(len(allweights)):
            #print("curMaxWeight:", curMaxWeight)
            #print("abs(allweights[j]):", abs(allweights[j]))
            if j in maxweightindexes:
                continue
            if (abs(allweights[j]) > curMaxWeight):
                curMaxWeight = abs(allweights[j])
                curMaxWeightIndex = j
        maxweightindexes[i] = curMaxWeightIndex

    return maxweightindexes

def PCAQuestionsSeparated(trainingdataset, testingdataset, traitbeingpredicted):
    
    print("PCAQUESTIONSSEPARATED")
    print(trainingdataset.shape)
    print(testingdataset.shape)

    pca = PCA(0.8)
    if (traitbeingpredicted != "Extroversion"):
        pca.fit(trainingdataset[:, 0:10])
        pcatrainingdataset1 = pca.transform(trainingdataset[:, 0:10])
        pcatestingdataset1 = pca.transform(testingdataset[:, 0:10])
    if (traitbeingpredicted != "Neuroticism"):
        pca.fit(trainingdataset[:, 10:20])
        pcatrainingdataset2 = pca.transform(trainingdataset[:, 10:20])
        pcatestingdataset2 = pca.transform(testingdataset[:, 10:20])
    if (traitbeingpredicted != "Agreeableness"):
        pca.fit(trainingdataset[:, 20:30])
        pcatrainingdataset3 = pca.transform(trainingdataset[:, 20:30])
        pcatestingdataset3 = pca.transform(testingdataset[:, 20:30])
    if (traitbeingpredicted != "Conscientiousness"):
        pca.fit(trainingdataset[:, 30:40])
        pcatrainingdataset4 = pca.transform(trainingdataset[:, 30:40])
        pcatestingdataset4 = pca.transform(testingdataset[:, 30:40])
    if (traitbeingpredicted != "Openness"):
        pca.fit(trainingdataset[:, 40:50])
        pcatrainingdataset5 = pca.transform(trainingdataset[:, 40:50])
        pcatestingdataset5 = pca.transform(testingdataset[:, 40:50])
    if (traitbeingpredicted == "Extroversion"):
        pcatrainingdataset = np.column_stack((pcatrainingdataset2, pcatrainingdataset3, pcatrainingdataset4, pcatrainingdataset5))
        pcatestingdataset = np.column_stack((pcatestingdataset2, pcatestingdataset3, pcatestingdataset4, pcatestingdataset5))
    if (traitbeingpredicted == "Neuroticism"):
        pcatrainingdataset = np.column_stack((pcatrainingdataset1, pcatrainingdataset3, pcatrainingdataset4, pcatrainingdataset5))
        pcatestingdataset = np.column_stack((pcatestingdataset1, pcatestingdataset3, pcatestingdataset4, pcatestingdataset5))
    if (traitbeingpredicted == "Agreeableness"):
        pcatrainingdataset = np.column_stack((pcatrainingdataset1, pcatrainingdataset2, pcatrainingdataset4, pcatrainingdataset5))
        pcatestingdataset = np.column_stack((pcatestingdataset1, pcatestingdataset2, pcatestingdataset4, pcatestingdataset5))
    if (traitbeingpredicted == "Conscientiousness"):
        pcatrainingdataset = np.column_stack((pcatrainingdataset1, pcatrainingdataset2, pcatrainingdataset3, pcatrainingdataset5))
        pcatestingdataset = np.column_stack((pcatestingdataset1, pcatestingdataset2, pcatestingdataset3, pcatestingdataset5))
    if (traitbeingpredicted == "Openness"):
        pcatrainingdataset = np.column_stack((pcatrainingdataset1, pcatrainingdataset2, pcatrainingdataset3, pcatrainingdataset4))
        pcatestingdataset = np.column_stack((pcatestingdataset1, pcatestingdataset2, pcatestingdataset3, pcatestingdataset4))

    print(pcatrainingdataset.shape)
    print(pcatestingdataset.shape)

    return pcatrainingdataset, pcatestingdataset

def PCAQuestions(trainingdataset, testingdataset):
    
    print("PCAQUESTIONS")

    print(trainingdataset.shape)
    print(testingdataset.shape)

    pca = PCA(0.8)
    pca.fit(trainingdataset)
    trainingpcadataset = pca.transform(trainingdataset)
    testingpcadataset = pca.transform(testingdataset)

    print(trainingpcadataset.shape)
    print(testingpcadataset.shape)

    return trainingpcadataset, testingpcadataset

def PredictExtroversion(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    individual_ens_estimators = []
    total_ens_estimators = []
    preference_ens_estimators = []
    pca_ens_estimators = []
    pcaseparated_ens_estimators = []

    trainingdatasetquestions = alltrainingdatasetquestions[:,10:] # Select all question columns not pertaining to extroversion
    trainingdatasettotals = alltrainingdatasettotals[:,1:] # Select all total columns but extroversion
    trainingdatasetpreferences = alltrainingdatasetpreferences[:,1:] # Select all preference columns but extroversion
    trainingcorrectlabels = alltrainingdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    trainingcorrectlabels = np.transpose(trainingcorrectlabels)

    testingdatasetquestions = alltestingdatasetquestions[:,10:] # Select all question columns not pertaining to extroversion
    pca_trainingdatasetquestionsseparated, pca_testingdatasetquestionsseparated = PCAQuestionsSeparated(alltrainingdatasetquestions, alltestingdatasetquestions, "Extroversion") # PCA guarenteeing the use of all five question types separately
    pca_trainingdatasetquestions, pca_testingdatasetquestions = PCAQuestions(trainingdatasetquestions, testingdatasetquestions) # PCA where questions may be combined
    testingdatasettotals = alltestingdatasettotals[:,1:] # Select all columns but extroversion
    testingdatasetpreferences = alltestingdatasetpreferences[:,1:] # Select all preference columns but extroversion
    testingcorrectlabels = alltestingdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    individual_ens_estimators.append(('perceptron', clf_percept0))
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    total_ens_estimators.append(('perceptron', clf_percept1))
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    preference_ens_estimators.append(('perceptron', clf_percept2))
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)

    clf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pca_ens_estimators.append(('perceptron', clf_percept3))
    clf_percept3.fit(pca_trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    pcapredictions = clf_percept3.predict(pca_trainingdatasetquestions)
    testpcapredictions = clf_percept3.predict(pca_testingdatasetquestions)

    clf_percept4 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pcaseparated_ens_estimators.append(('perceptron', clf_percept4))
    clf_percept4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels, sample_weight=None)
    pcaseparatedpredictions = clf_percept4.predict(pca_trainingdatasetquestionsseparated)
    testpcaseparatedpredictions = clf_percept4.predict(pca_testingdatasetquestionsseparated)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    individual_ens_estimators.append(('sgd', clf_sgd0))
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    total_ens_estimators.append(('sgd', clf_sgd1))
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    preference_ens_estimators.append(('sgd', clf_sgd2))
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_sgd3 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pca_ens_estimators.append(('sgd', clf_sgd3))
    clf_sgd3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    sgdpcapredictions = clf_sgd3.predict(pca_trainingdatasetquestions)
    sgdtestpcapredictions = clf_sgd3.predict(pca_testingdatasetquestions)
    
    clf_sgd4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pcaseparated_ens_estimators.append(('sgd', clf_sgd4))
    clf_sgd4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    sgdpcaseparatedpredictions = clf_sgd4.predict(pca_trainingdatasetquestionsseparated)
    sgdtestpcaseparatedpredictions = clf_sgd4.predict(pca_testingdatasetquestionsseparated)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    individual_ens_estimators.append(('log_reg', clf_logistic0))
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    total_ens_estimators.append(('log_reg', clf_logistic1))
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    preference_ens_estimators.append(('log_reg', clf_logistic2))
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_logistic3 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pca_ens_estimators.append(('log_reg', clf_logistic3))
    clf_logistic3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    logisticpcapredictions = clf_logistic3.predict(pca_trainingdatasetquestions)
    logistictestpcapredictions = clf_logistic3.predict(pca_testingdatasetquestions)
    
    clf_logistic4 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pcaseparated_ens_estimators.append(('log_reg', clf_logistic4))
    clf_logistic4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    logisticpcaseparatedpredictions = clf_logistic4.predict(pca_trainingdatasetquestionsseparated)
    logistictestpcaseparatedpredictions = clf_logistic4.predict(pca_testingdatasetquestionsseparated)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    individual_ens_estimators.append(('dt', clf_decisiontree0))
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    total_ens_estimators.append(('dt', clf_decisiontree1))
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    preference_ens_estimators.append(('dt', clf_decisiontree2))
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)
    
    clf_decisiontree3 = DecisionTreeClassifier(max_depth=20)
    pca_ens_estimators.append(('dt', clf_decisiontree3))
    clf_decisiontree3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    decisiontreepcapredictions = clf_decisiontree3.predict(pca_trainingdatasetquestions)
    decisiontreetestpcapredictions = clf_decisiontree3.predict(pca_testingdatasetquestions)
    
    clf_decisiontree4 = DecisionTreeClassifier(max_depth=20)
    pcaseparated_ens_estimators.append(('dt', clf_decisiontree4))
    clf_decisiontree4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    decisiontreepcaseparatedpredictions = clf_decisiontree4.predict(pca_trainingdatasetquestionsseparated)
    decisiontreetestpcaseparatedpredictions = clf_decisiontree4.predict(pca_testingdatasetquestionsseparated)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict extroversion based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    pca_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcapredictions)

    print("Able to predict extroversion based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy), file=file_out)

    pca_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcapredictions)

    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy), file=file_out)

    pcaseparated_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcaseparatedpredictions)

    print("Able to predict extroversion based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy), file=file_out)

    pcaseparated_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcaseparatedpredictions)

    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy), file=file_out)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict extroversion based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)

    pca_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcapredictions)

    print("Able to predict extroversion based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy), file=file_out)

    pca_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcapredictions)

    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy), file=file_out)

    pcaseparated_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcaseparatedpredictions)

    print("Able to predict extroversion based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy), file=file_out)

    pcaseparated_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcaseparatedpredictions)

    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy), file=file_out)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict extroversion based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict extroversion based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict extroversion based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    pca_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcapredictions)

    print("Able to predict extroversion based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy), file=file_out)

    pca_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcapredictions)

    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy), file=file_out)

    pcaseparated_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcaseparatedpredictions)

    print("Able to predict extroversion based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy), file=file_out)

    pcaseparated_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcaseparatedpredictions)

    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy), file=file_out)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict extroversion based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict extroversion based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict extroversion based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict extroversion based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict extroversion based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict extroversion based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict extroversion based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    pca_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcapredictions)

    print("Able to predict extroversion based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy), file=file_out)

    pca_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcapredictions)

    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy), file=file_out)

    pcaseparated_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcaseparatedpredictions)

    print("Able to predict extroversion based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy))
    print("Able to predict extroversion based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy), file=file_out)

    pcaseparated_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcaseparatedpredictions)

    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy), file=file_out)

    individual_voting = VotingClassifier(estimators=individual_ens_estimators)
    individual_voting.fit(trainingdatasetquestions, trainingcorrectlabels)
    votingquestionpredictions = individual_voting.predict(trainingdatasetquestions)

    questions_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingquestionpredictions)

    print("Able to predict extroversion based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy))
    print("Able to predict extroversion based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy), file=file_out)

    votingtestquestionpredictions = individual_voting.predict(testingdatasetquestions)
    questions_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestquestionpredictions)

    print("Able to predict extroversion based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy))
    print("Able to predict extroversion based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy), file=file_out)

    totals_voting = VotingClassifier(estimators=total_ens_estimators)
    totals_voting.fit(trainingdatasettotals, trainingcorrectlabels)
    votingtotalpredictions = totals_voting.predict(trainingdatasettotals)

    totals_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingtotalpredictions)

    print("Able to predict extroversion based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy))
    print("Able to predict extroversion based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy), file=file_out)

    votingtesttotalspredictions = totals_voting.predict(testingdatasettotals)
    totals_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtesttotalspredictions)

    print("Able to predict extroversion based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy))
    print("Able to predict extroversion based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy), file=file_out)

    preferences_voting = VotingClassifier(estimators=preference_ens_estimators)
    preferences_voting.fit(trainingdatasetpreferences, trainingcorrectlabels)
    preferencestotalpredictions = preferences_voting.predict(trainingdatasetpreferences)

    preferences_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencestotalpredictions)

    print("Able to predict extroversion based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy))
    print("Able to predict extroversion based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy), file=file_out)

    votingtestpreferencespredictions = preferences_voting.predict(testingdatasetpreferences)
    preferences_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpreferencespredictions)

    print("Able to predict extroversion based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy))
    print("Able to predict extroversion based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy), file=file_out)

    pca_voting = VotingClassifier(estimators=pca_ens_estimators)
    pca_voting.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    votingpcapredictions = pca_voting.predict(pca_trainingdatasetquestions)

    pac_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcapredictions)

    print("Able to predict extroversion based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy))
    print("Able to predict extroversion based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy), file=file_out)

    votingtestpcapredictions = pca_voting.predict(pca_testingdatasetquestions)
    pca_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcapredictions)

    print("Able to predict extroversion based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy))
    print("Able to predict extroversion based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy), file=file_out)

    pcaseparated_voting = VotingClassifier(estimators=pcaseparated_ens_estimators)
    pcaseparated_voting.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    votingpcaseparatedpredictions = pcaseparated_voting.predict(pca_trainingdatasetquestionsseparated)

    pacseparated_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcaseparatedpredictions)

    print("Able to predict extroversion based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy))
    print("Able to predict extroversion based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy), file=file_out)

    votingtestpcaseparatedpredictions = pcaseparated_voting.predict(pca_testingdatasetquestionsseparated)
    pcaseparated_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcaseparatedpredictions)

    print("Able to predict extroversion based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy))
    print("Able to predict extroversion based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy), file=file_out)

    return

def PredictNeuroticism(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    individual_ens_estimators = []
    total_ens_estimators = []
    preference_ens_estimators = []
    pca_ens_estimators = []
    pcaseparated_ens_estimators = []

    columnstodelete = list(range(10, 20))

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 1, axis=1) # Select all columns but neuroticism
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    trainingcorrectlabels = alltrainingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    
    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    pca_trainingdatasetquestionsseparated, pca_testingdatasetquestionsseparated = PCAQuestionsSeparated(alltrainingdatasetquestions, alltestingdatasetquestions, "Neuroticism") # PCA guarenteeing the use of all five question types separately
    pca_trainingdatasetquestions, pca_testingdatasetquestions = PCAQuestions(trainingdatasetquestions, testingdatasetquestions) # PCA where questions may be combined
    testingdatasettotals = np.delete(alltestingdatasettotals, 1, axis=1) # Select all columns but neuroticism
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 1, axis=1) # Select all columns but neuroticism
    testingcorrectlabels = alltestingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    individual_ens_estimators.append(('perceptron', clf_percept0))
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    total_ens_estimators.append(('perceptron', clf_percept1))
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    preference_ens_estimators.append(('perceptron', clf_percept2))
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)

    clf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pca_ens_estimators.append(('perceptron', clf_percept3))
    clf_percept3.fit(pca_trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    pcapredictions = clf_percept3.predict(pca_trainingdatasetquestions)
    testpcapredictions = clf_percept3.predict(pca_testingdatasetquestions)

    clf_percept4 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pcaseparated_ens_estimators.append(('perceptron', clf_percept4))
    clf_percept4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels, sample_weight=None)
    pcaseparatedpredictions = clf_percept4.predict(pca_trainingdatasetquestionsseparated)
    testpcaseparatedpredictions = clf_percept4.predict(pca_testingdatasetquestionsseparated)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    individual_ens_estimators.append(('sgd', clf_sgd0))
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    total_ens_estimators.append(('sgd', clf_sgd1))
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    preference_ens_estimators.append(('sgd', clf_sgd2))
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_sgd3 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pca_ens_estimators.append(('sgd', clf_sgd3))
    clf_sgd3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    sgdpcapredictions = clf_sgd3.predict(pca_trainingdatasetquestions)
    sgdtestpcapredictions = clf_sgd3.predict(pca_testingdatasetquestions)
    
    clf_sgd4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pcaseparated_ens_estimators.append(('sgd', clf_sgd4))
    clf_sgd4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    sgdpcaseparatedpredictions = clf_sgd4.predict(pca_trainingdatasetquestionsseparated)
    sgdtestpcaseparatedpredictions = clf_sgd4.predict(pca_testingdatasetquestionsseparated)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    individual_ens_estimators.append(('log_reg', clf_logistic0))
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    total_ens_estimators.append(('log_reg', clf_logistic1))
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    preference_ens_estimators.append(('log_reg', clf_logistic2))
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_logistic3 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pca_ens_estimators.append(('log_reg', clf_logistic3))
    clf_logistic3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    logisticpcapredictions = clf_logistic3.predict(pca_trainingdatasetquestions)
    logistictestpcapredictions = clf_logistic3.predict(pca_testingdatasetquestions)
    
    clf_logistic4 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pcaseparated_ens_estimators.append(('log_reg', clf_logistic4))
    clf_logistic4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    logisticpcaseparatedpredictions = clf_logistic4.predict(pca_trainingdatasetquestionsseparated)
    logistictestpcaseparatedpredictions = clf_logistic4.predict(pca_testingdatasetquestionsseparated)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    individual_ens_estimators.append(('dt', clf_decisiontree0))
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    total_ens_estimators.append(('dt', clf_decisiontree1))
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    preference_ens_estimators.append(('dt', clf_decisiontree2))
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)
    
    clf_decisiontree3 = DecisionTreeClassifier(max_depth=20)
    pca_ens_estimators.append(('dt', clf_decisiontree3))
    clf_decisiontree3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    decisiontreepcapredictions = clf_decisiontree3.predict(pca_trainingdatasetquestions)
    decisiontreetestpcapredictions = clf_decisiontree3.predict(pca_testingdatasetquestions)
    
    clf_decisiontree4 = DecisionTreeClassifier(max_depth=20)
    pcaseparated_ens_estimators.append(('dt', clf_decisiontree4))
    clf_decisiontree4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    decisiontreepcaseparatedpredictions = clf_decisiontree4.predict(pca_trainingdatasetquestionsseparated)
    decisiontreetestpcaseparatedpredictions = clf_decisiontree4.predict(pca_testingdatasetquestionsseparated)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict neuroticism based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    pca_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcapredictions)

    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy), file=file_out)

    pca_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcapredictions)

    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy), file=file_out)

    pcaseparated_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcaseparatedpredictions)

    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy), file=file_out)

    pcaseparated_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy), file=file_out)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict neuroticism based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)

    pca_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcapredictions)

    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy), file=file_out)

    pca_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcapredictions)

    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy), file=file_out)

    pcaseparated_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy), file=file_out)

    pcaseparated_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy), file=file_out)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict neuroticism based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict neuroticism based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict neuroticism based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    pca_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcapredictions)

    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy), file=file_out)

    pca_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcapredictions)

    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy), file=file_out)

    pcaseparated_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy), file=file_out)

    pcaseparated_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy), file=file_out)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict neuroticism based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    pca_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcapredictions)

    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy), file=file_out)

    pca_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcapredictions)

    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy), file=file_out)

    pcaseparated_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcaseparatedpredictions)

    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy))
    print("Able to predict neuroticism based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy), file=file_out)

    pcaseparated_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy), file=file_out)

    individual_voting = VotingClassifier(estimators=individual_ens_estimators)
    individual_voting.fit(trainingdatasetquestions, trainingcorrectlabels)
    votingquestionpredictions = individual_voting.predict(trainingdatasetquestions)

    questions_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingquestionpredictions)

    print("Able to predict neuroticism based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy))
    print("Able to predict neuroticism based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy), file=file_out)

    votingtestquestionpredictions = individual_voting.predict(testingdatasetquestions)
    questions_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestquestionpredictions)

    print("Able to predict neuroticism based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy))
    print("Able to predict neuroticism based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy), file=file_out)

    totals_voting = VotingClassifier(estimators=total_ens_estimators)
    totals_voting.fit(trainingdatasettotals, trainingcorrectlabels)
    votingtotalpredictions = totals_voting.predict(trainingdatasettotals)

    totals_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingtotalpredictions)

    print("Able to predict neuroticism based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy))
    print("Able to predict neuroticism based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy), file=file_out)

    votingtesttotalspredictions = totals_voting.predict(testingdatasettotals)
    totals_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtesttotalspredictions)

    print("Able to predict neuroticism based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy))
    print("Able to predict neuroticism based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy), file=file_out)

    preferences_voting = VotingClassifier(estimators=preference_ens_estimators)
    preferences_voting.fit(trainingdatasetpreferences, trainingcorrectlabels)
    preferencestotalpredictions = preferences_voting.predict(trainingdatasetpreferences)

    preferences_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencestotalpredictions)

    print("Able to predict neuroticism based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy))
    print("Able to predict neuroticism based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy), file=file_out)

    votingtestpreferencespredictions = preferences_voting.predict(testingdatasetpreferences)
    preferences_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpreferencespredictions)

    print("Able to predict neuroticism based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy))
    print("Able to predict neuroticism based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy), file=file_out)

    pca_voting = VotingClassifier(estimators=pca_ens_estimators)
    pca_voting.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    votingpcapredictions = pca_voting.predict(pca_trainingdatasetquestions)

    pac_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcapredictions)

    print("Able to predict neuroticism based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy))
    print("Able to predict neuroticism based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy), file=file_out)

    votingtestpcapredictions = pca_voting.predict(pca_testingdatasetquestions)
    pca_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcapredictions)

    print("Able to predict neuroticism based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy))
    print("Able to predict neuroticism based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy), file=file_out)

    pcaseparated_voting = VotingClassifier(estimators=pcaseparated_ens_estimators)
    pcaseparated_voting.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    votingpcaseparatedpredictions = pcaseparated_voting.predict(pca_trainingdatasetquestionsseparated)

    pacseparated_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy))
    print("Able to predict neuroticism based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy), file=file_out)

    votingtestpcaseparatedpredictions = pcaseparated_voting.predict(pca_testingdatasetquestionsseparated)
    pcaseparated_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcaseparatedpredictions)

    print("Able to predict neuroticism based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy))
    print("Able to predict neuroticism based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy), file=file_out)

    return

def PredictAgreeableness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    individual_ens_estimators = []
    total_ens_estimators = []
    preference_ens_estimators = []
    pca_ens_estimators = []
    pcaseparated_ens_estimators = []

    columnstodelete = list(range(20, 30))

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 2, axis=1) # Select all columns but agreeableness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    
    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    pca_trainingdatasetquestionsseparated, pca_testingdatasetquestionsseparated = PCAQuestionsSeparated(alltrainingdatasetquestions, alltestingdatasetquestions, "Agreeableness") # PCA guarenteeing the use of all five question types separately
    pca_trainingdatasetquestions, pca_testingdatasetquestions = PCAQuestions(trainingdatasetquestions, testingdatasetquestions) # PCA where questions may be combined
    testingdatasettotals = np.delete(alltestingdatasettotals, 2, axis=1) # Select all columns but agreeableness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 2, axis=1) # Select all columns but agreeableness
    testingcorrectlabels = alltestingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    individual_ens_estimators.append(('perceptron', clf_percept0))
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    total_ens_estimators.append(('perceptron', clf_percept1))
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    preference_ens_estimators.append(('perceptron', clf_percept2))
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)

    clf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pca_ens_estimators.append(('perceptron', clf_percept3))
    clf_percept3.fit(pca_trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    pcapredictions = clf_percept3.predict(pca_trainingdatasetquestions)
    testpcapredictions = clf_percept3.predict(pca_testingdatasetquestions)

    clf_percept4 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pcaseparated_ens_estimators.append(('perceptron', clf_percept4))
    clf_percept4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels, sample_weight=None)
    pcaseparatedpredictions = clf_percept4.predict(pca_trainingdatasetquestionsseparated)
    testpcaseparatedpredictions = clf_percept4.predict(pca_testingdatasetquestionsseparated)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    individual_ens_estimators.append(('sgd', clf_sgd0))
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    total_ens_estimators.append(('sgd', clf_sgd1))
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    preference_ens_estimators.append(('sgd', clf_sgd2))
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_sgd3 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pca_ens_estimators.append(('sgd', clf_sgd3))
    clf_sgd3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    sgdpcapredictions = clf_sgd3.predict(pca_trainingdatasetquestions)
    sgdtestpcapredictions = clf_sgd3.predict(pca_testingdatasetquestions)
    
    clf_sgd4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pcaseparated_ens_estimators.append(('sgd', clf_sgd4))
    clf_sgd4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    sgdpcaseparatedpredictions = clf_sgd4.predict(pca_trainingdatasetquestionsseparated)
    sgdtestpcaseparatedpredictions = clf_sgd4.predict(pca_testingdatasetquestionsseparated)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    individual_ens_estimators.append(('log_reg', clf_logistic0))
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    total_ens_estimators.append(('log_reg', clf_logistic1))
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    preference_ens_estimators.append(('log_reg', clf_logistic2))
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_logistic3 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pca_ens_estimators.append(('log_reg', clf_logistic3))
    clf_logistic3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    logisticpcapredictions = clf_logistic3.predict(pca_trainingdatasetquestions)
    logistictestpcapredictions = clf_logistic3.predict(pca_testingdatasetquestions)
    
    clf_logistic4 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pcaseparated_ens_estimators.append(('log_reg', clf_logistic4))
    clf_logistic4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    logisticpcaseparatedpredictions = clf_logistic4.predict(pca_trainingdatasetquestionsseparated)
    logistictestpcaseparatedpredictions = clf_logistic4.predict(pca_testingdatasetquestionsseparated)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    individual_ens_estimators.append(('dt', clf_decisiontree0))
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    total_ens_estimators.append(('dt', clf_decisiontree1))
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    preference_ens_estimators.append(('dt', clf_decisiontree2))
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)
    
    clf_decisiontree3 = DecisionTreeClassifier(max_depth=20)
    pca_ens_estimators.append(('dt', clf_decisiontree3))
    clf_decisiontree3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    decisiontreepcapredictions = clf_decisiontree3.predict(pca_trainingdatasetquestions)
    decisiontreetestpcapredictions = clf_decisiontree3.predict(pca_testingdatasetquestions)
    
    clf_decisiontree4 = DecisionTreeClassifier(max_depth=20)
    pcaseparated_ens_estimators.append(('dt', clf_decisiontree4))
    clf_decisiontree4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    decisiontreepcaseparatedpredictions = clf_decisiontree4.predict(pca_trainingdatasetquestionsseparated)
    decisiontreetestpcaseparatedpredictions = clf_decisiontree4.predict(pca_testingdatasetquestionsseparated)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict agreeableness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    pca_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcapredictions)

    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy), file=file_out)

    pca_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcapredictions)

    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy), file=file_out)

    pcaseparated_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcaseparatedpredictions)

    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy), file=file_out)

    pcaseparated_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy), file=file_out)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict agreeableness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)

    pca_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcapredictions)

    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy), file=file_out)

    pca_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcapredictions)

    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy), file=file_out)

    pcaseparated_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy), file=file_out)

    pcaseparated_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy), file=file_out)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict agreeableness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict agreeableness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict agreeableness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    pca_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcapredictions)

    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy), file=file_out)

    pca_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcapredictions)

    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy), file=file_out)

    pcaseparated_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy), file=file_out)

    pcaseparated_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy), file=file_out)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict agreeableness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    pca_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcapredictions)

    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy), file=file_out)

    pca_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcapredictions)

    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy), file=file_out)

    pcaseparated_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcaseparatedpredictions)

    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy))
    print("Able to predict agreeableness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy), file=file_out)

    pcaseparated_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy), file=file_out)

    individual_voting = VotingClassifier(estimators=individual_ens_estimators)
    individual_voting.fit(trainingdatasetquestions, trainingcorrectlabels)
    votingquestionpredictions = individual_voting.predict(trainingdatasetquestions)

    questions_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingquestionpredictions)

    print("Able to predict agreeableness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy))
    print("Able to predict agreeableness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy), file=file_out)

    votingtestquestionpredictions = individual_voting.predict(testingdatasetquestions)
    questions_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestquestionpredictions)

    print("Able to predict agreeableness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy))
    print("Able to predict agreeableness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy), file=file_out)

    totals_voting = VotingClassifier(estimators=total_ens_estimators)
    totals_voting.fit(trainingdatasettotals, trainingcorrectlabels)
    votingtotalpredictions = totals_voting.predict(trainingdatasettotals)

    totals_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingtotalpredictions)

    print("Able to predict agreeableness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy))
    print("Able to predict agreeableness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy), file=file_out)

    votingtesttotalspredictions = totals_voting.predict(testingdatasettotals)
    totals_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtesttotalspredictions)

    print("Able to predict agreeableness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy))
    print("Able to predict agreeableness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy), file=file_out)

    preferences_voting = VotingClassifier(estimators=preference_ens_estimators)
    preferences_voting.fit(trainingdatasetpreferences, trainingcorrectlabels)
    preferencestotalpredictions = preferences_voting.predict(trainingdatasetpreferences)

    preferences_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencestotalpredictions)

    print("Able to predict agreeableness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy))
    print("Able to predict agreeableness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy), file=file_out)

    votingtestpreferencespredictions = preferences_voting.predict(testingdatasetpreferences)
    preferences_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpreferencespredictions)

    print("Able to predict agreeableness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy))
    print("Able to predict agreeableness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy), file=file_out)

    pca_voting = VotingClassifier(estimators=pca_ens_estimators)
    pca_voting.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    votingpcapredictions = pca_voting.predict(pca_trainingdatasetquestions)

    pac_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcapredictions)

    print("Able to predict agreeableness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy))
    print("Able to predict agreeableness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy), file=file_out)

    votingtestpcapredictions = pca_voting.predict(pca_testingdatasetquestions)
    pca_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcapredictions)

    print("Able to predict agreeableness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy))
    print("Able to predict agreeableness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy), file=file_out)

    pcaseparated_voting = VotingClassifier(estimators=pcaseparated_ens_estimators)
    pcaseparated_voting.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    votingpcaseparatedpredictions = pcaseparated_voting.predict(pca_trainingdatasetquestionsseparated)

    pacseparated_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy))
    print("Able to predict agreeableness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy), file=file_out)

    votingtestpcaseparatedpredictions = pcaseparated_voting.predict(pca_testingdatasetquestionsseparated)
    pcaseparated_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcaseparatedpredictions)

    print("Able to predict agreeableness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy))
    print("Able to predict agreeableness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy), file=file_out)

    return

def PredictConscientiousness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    individual_ens_estimators = []
    total_ens_estimators = []
    preference_ens_estimators = []
    pca_ens_estimators = []
    pcaseparated_ens_estimators = []

    columnstodelete = list(range(30, 40))

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 3, axis=1) # Select all columns but conscientiousness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    
    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    pca_trainingdatasetquestionsseparated, pca_testingdatasetquestionsseparated = PCAQuestionsSeparated(alltrainingdatasetquestions, alltestingdatasetquestions, "Conscientiousness") # PCA guarenteeing the use of all five question types separately
    pca_trainingdatasetquestions, pca_testingdatasetquestions = PCAQuestions(trainingdatasetquestions, testingdatasetquestions) # PCA where questions may be combined
    testingdatasettotals = np.delete(alltestingdatasettotals, 3, axis=1) # Select all columns but conscientiousness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 3, axis=1) # Select all columns but conscientiousness
    testingcorrectlabels = alltestingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    individual_ens_estimators.append(('perceptron', clf_percept0))
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    total_ens_estimators.append(('perceptron', clf_percept1))
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    preference_ens_estimators.append(('perceptron', clf_percept2))
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)

    clf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pca_ens_estimators.append(('perceptron', clf_percept3))
    clf_percept3.fit(pca_trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    pcapredictions = clf_percept3.predict(pca_trainingdatasetquestions)
    testpcapredictions = clf_percept3.predict(pca_testingdatasetquestions)

    clf_percept4 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pcaseparated_ens_estimators.append(('perceptron', clf_percept4))
    clf_percept4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels, sample_weight=None)
    pcaseparatedpredictions = clf_percept4.predict(pca_trainingdatasetquestionsseparated)
    testpcaseparatedpredictions = clf_percept4.predict(pca_testingdatasetquestionsseparated)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    individual_ens_estimators.append(('sgd', clf_sgd0))
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    total_ens_estimators.append(('sgd', clf_sgd1))
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    preference_ens_estimators.append(('sgd', clf_sgd2))
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_sgd3 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pca_ens_estimators.append(('sgd', clf_sgd3))
    clf_sgd3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    sgdpcapredictions = clf_sgd3.predict(pca_trainingdatasetquestions)
    sgdtestpcapredictions = clf_sgd3.predict(pca_testingdatasetquestions)
    
    clf_sgd4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pcaseparated_ens_estimators.append(('sgd', clf_sgd4))
    clf_sgd4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    sgdpcaseparatedpredictions = clf_sgd4.predict(pca_trainingdatasetquestionsseparated)
    sgdtestpcaseparatedpredictions = clf_sgd4.predict(pca_testingdatasetquestionsseparated)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    individual_ens_estimators.append(('log_reg', clf_logistic0))
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    total_ens_estimators.append(('log_reg', clf_logistic1))
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    preference_ens_estimators.append(('log_reg', clf_logistic2))
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_logistic3 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pca_ens_estimators.append(('log_reg', clf_logistic3))
    clf_logistic3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    logisticpcapredictions = clf_logistic3.predict(pca_trainingdatasetquestions)
    logistictestpcapredictions = clf_logistic3.predict(pca_testingdatasetquestions)
    
    clf_logistic4 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pcaseparated_ens_estimators.append(('log_reg', clf_logistic4))
    clf_logistic4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    logisticpcaseparatedpredictions = clf_logistic4.predict(pca_trainingdatasetquestionsseparated)
    logistictestpcaseparatedpredictions = clf_logistic4.predict(pca_testingdatasetquestionsseparated)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    individual_ens_estimators.append(('dt', clf_decisiontree0))
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    total_ens_estimators.append(('dt', clf_decisiontree1))
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    preference_ens_estimators.append(('dt', clf_decisiontree2))
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)
    
    clf_decisiontree3 = DecisionTreeClassifier(max_depth=20)
    pca_ens_estimators.append(('dt', clf_decisiontree3))
    clf_decisiontree3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    decisiontreepcapredictions = clf_decisiontree3.predict(pca_trainingdatasetquestions)
    decisiontreetestpcapredictions = clf_decisiontree3.predict(pca_testingdatasetquestions)
    
    clf_decisiontree4 = DecisionTreeClassifier(max_depth=20)
    pcaseparated_ens_estimators.append(('dt', clf_decisiontree4))
    clf_decisiontree4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    decisiontreepcaseparatedpredictions = clf_decisiontree4.predict(pca_trainingdatasetquestionsseparated)
    decisiontreetestpcaseparatedpredictions = clf_decisiontree4.predict(pca_testingdatasetquestionsseparated)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict conscientiousness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    pca_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcapredictions)

    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy), file=file_out)

    pca_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcapredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy), file=file_out)

    pcaseparated_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy), file=file_out)

    pcaseparated_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy), file=file_out)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict conscientiousness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)

    pca_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcapredictions)

    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy), file=file_out)

    pca_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcapredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy), file=file_out)

    pcaseparated_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy), file=file_out)

    pcaseparated_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy), file=file_out)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict conscientiousness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict conscientiousness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    pca_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcapredictions)

    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy), file=file_out)

    pca_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcapredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy), file=file_out)

    pcaseparated_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy), file=file_out)

    pcaseparated_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy), file=file_out)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict conscientiousness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    pca_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcapredictions)

    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy), file=file_out)

    pca_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcapredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy), file=file_out)

    pcaseparated_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy), file=file_out)

    pcaseparated_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy), file=file_out)

    individual_voting = VotingClassifier(estimators=individual_ens_estimators)
    individual_voting.fit(trainingdatasetquestions, trainingcorrectlabels)
    votingquestionpredictions = individual_voting.predict(trainingdatasetquestions)

    questions_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingquestionpredictions)

    print("Able to predict conscientiousness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy))
    print("Able to predict conscientiousness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy), file=file_out)

    votingtestquestionpredictions = individual_voting.predict(testingdatasetquestions)
    questions_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestquestionpredictions)

    print("Able to predict conscientiousness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy), file=file_out)

    totals_voting = VotingClassifier(estimators=total_ens_estimators)
    totals_voting.fit(trainingdatasettotals, trainingcorrectlabels)
    votingtotalpredictions = totals_voting.predict(trainingdatasettotals)

    totals_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingtotalpredictions)

    print("Able to predict conscientiousness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy))
    print("Able to predict conscientiousness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy), file=file_out)

    votingtesttotalspredictions = totals_voting.predict(testingdatasettotals)
    totals_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtesttotalspredictions)

    print("Able to predict conscientiousness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy), file=file_out)

    preferences_voting = VotingClassifier(estimators=preference_ens_estimators)
    preferences_voting.fit(trainingdatasetpreferences, trainingcorrectlabels)
    preferencestotalpredictions = preferences_voting.predict(trainingdatasetpreferences)

    preferences_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencestotalpredictions)

    print("Able to predict conscientiousness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy))
    print("Able to predict conscientiousness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy), file=file_out)

    votingtestpreferencespredictions = preferences_voting.predict(testingdatasetpreferences)
    preferences_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpreferencespredictions)

    print("Able to predict conscientiousness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy), file=file_out)

    pca_voting = VotingClassifier(estimators=pca_ens_estimators)
    pca_voting.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    votingpcapredictions = pca_voting.predict(pca_trainingdatasetquestions)

    pac_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcapredictions)

    print("Able to predict conscientiousness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy))
    print("Able to predict conscientiousness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy), file=file_out)

    votingtestpcapredictions = pca_voting.predict(pca_testingdatasetquestions)
    pca_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcapredictions)

    print("Able to predict conscientiousness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy), file=file_out)

    pcaseparated_voting = VotingClassifier(estimators=pcaseparated_ens_estimators)
    pcaseparated_voting.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    votingpcaseparatedpredictions = pcaseparated_voting.predict(pca_trainingdatasetquestionsseparated)

    pacseparated_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy))
    print("Able to predict conscientiousness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy), file=file_out)

    votingtestpcaseparatedpredictions = pcaseparated_voting.predict(pca_testingdatasetquestionsseparated)
    pcaseparated_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcaseparatedpredictions)

    print("Able to predict conscientiousness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy))
    print("Able to predict conscientiousness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy), file=file_out)

    return

def PredictOpenness(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    individual_ens_estimators = []
    total_ens_estimators = []
    preference_ens_estimators = []
    pca_ens_estimators = []
    pcaseparated_ens_estimators = []

    columnstodelete = list(range(40, 50))

    trainingdatasetquestions = np.delete(alltrainingdatasetquestions, columnstodelete, axis=1)
    trainingdatasettotals = np.delete(alltrainingdatasettotals, 4, axis=1) # Select all columns but openness
    trainingdatasetpreferences = np.delete(alltrainingdatasetpreferences, 4, axis=1) # Select all columns but openness
    trainingcorrectlabels = alltrainingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testingdatasetquestions = np.delete(alltestingdatasetquestions, columnstodelete, axis=1)
    pca_trainingdatasetquestionsseparated, pca_testingdatasetquestionsseparated = PCAQuestionsSeparated(alltrainingdatasetquestions, alltestingdatasetquestions, "Openness") # PCA guarenteeing the use of all five question types separately
    pca_trainingdatasetquestions, pca_testingdatasetquestions = PCAQuestions(trainingdatasetquestions, testingdatasetquestions) # PCA where questions may be combined
    testingdatasettotals = np.delete(alltestingdatasettotals, 4, axis=1) # Select all columns but openness
    testingdatasetpreferences = np.delete(alltestingdatasetpreferences, 4, axis=1) # Select all columns but openness
    testingcorrectlabels = alltestingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    clf_percept0 = Perceptron(max_iter=20, random_state=0, eta0=1)
    individual_ens_estimators.append(('perceptron', clf_percept0))
    clf_percept0.fit(trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    perceptquestionpredictions = clf_percept0.predict(trainingdatasetquestions)
    testquestionpredictions = clf_percept0.predict(testingdatasetquestions)

    clf_percept1 = Perceptron(max_iter=20, random_state=0, eta0=1)
    total_ens_estimators.append(('perceptron', clf_percept1))
    clf_percept1.fit(trainingdatasettotals, trainingcorrectlabels, sample_weight=None)
    perceptpredictions = clf_percept1.predict(trainingdatasettotals)
    testpredictions = clf_percept1.predict(testingdatasettotals)

    clf_percept2 = Perceptron(max_iter=20, random_state=0, eta0=1)
    preference_ens_estimators.append(('perceptron', clf_percept2))
    clf_percept2.fit(trainingdatasetpreferences, trainingcorrectlabels, sample_weight=None)
    preferencepredictions = clf_percept2.predict(trainingdatasetpreferences)
    testpreferencepredictions = clf_percept2.predict(testingdatasetpreferences)

    clf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pca_ens_estimators.append(('perceptron', clf_percept3))
    clf_percept3.fit(pca_trainingdatasetquestions, trainingcorrectlabels, sample_weight=None)
    pcapredictions = clf_percept3.predict(pca_trainingdatasetquestions)
    testpcapredictions = clf_percept3.predict(pca_testingdatasetquestions)

    clf_percept4 = Perceptron(max_iter=20, random_state=0, eta0=1)
    pcaseparated_ens_estimators.append(('perceptron', clf_percept4))
    clf_percept4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels, sample_weight=None)
    pcaseparatedpredictions = clf_percept4.predict(pca_trainingdatasetquestionsseparated)
    testpcaseparatedpredictions = clf_percept4.predict(pca_testingdatasetquestionsseparated)
    
    clf_sgd0 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    individual_ens_estimators.append(('sgd', clf_sgd0))
    clf_sgd0.fit(trainingdatasetquestions, trainingcorrectlabels)
    sgdquestionpredictions = clf_sgd0.predict(trainingdatasetquestions)
    sgdtestquestionpredictions = clf_sgd0.predict(testingdatasetquestions)
    
    clf_sgd1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    total_ens_estimators.append(('sgd', clf_sgd1))
    clf_sgd1.fit(trainingdatasettotals, trainingcorrectlabels)
    sgdpredictions = clf_sgd1.predict(trainingdatasettotals)
    sgdtestpredictions = clf_sgd1.predict(testingdatasettotals)
    
    clf_sgd2 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    preference_ens_estimators.append(('sgd', clf_sgd2))
    clf_sgd2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    sgdpreferencepredictions = clf_sgd2.predict(trainingdatasetpreferences)
    sgdtestpreferencepredictions = clf_sgd2.predict(testingdatasetpreferences)
    
    clf_sgd3 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pca_ens_estimators.append(('sgd', clf_sgd3))
    clf_sgd3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    sgdpcapredictions = clf_sgd3.predict(pca_trainingdatasetquestions)
    sgdtestpcapredictions = clf_sgd3.predict(pca_testingdatasetquestions)
    
    clf_sgd4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    pcaseparated_ens_estimators.append(('sgd', clf_sgd4))
    clf_sgd4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    sgdpcaseparatedpredictions = clf_sgd4.predict(pca_trainingdatasetquestionsseparated)
    sgdtestpcaseparatedpredictions = clf_sgd4.predict(pca_testingdatasetquestionsseparated)
    
    clf_logistic0 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    individual_ens_estimators.append(('log_reg', clf_logistic0))
    clf_logistic0.fit(trainingdatasetquestions, trainingcorrectlabels)
    logisticquestionpredictions = clf_logistic0.predict(trainingdatasetquestions)
    logistictestquestionpredictions = clf_logistic0.predict(testingdatasetquestions)
    
    clf_logistic1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    total_ens_estimators.append(('log_reg', clf_logistic1))
    clf_logistic1.fit(trainingdatasettotals, trainingcorrectlabels)
    logisticpredictions = clf_logistic1.predict(trainingdatasettotals)
    logistictestpredictions = clf_logistic1.predict(testingdatasettotals)
    
    clf_logistic2 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    preference_ens_estimators.append(('log_reg', clf_logistic2))
    clf_logistic2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    logisticpreferencepredictions = clf_logistic2.predict(trainingdatasetpreferences)
    logistictestpreferencepredictions = clf_logistic2.predict(testingdatasetpreferences)
    
    clf_logistic3 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pca_ens_estimators.append(('log_reg', clf_logistic3))
    clf_logistic3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    logisticpcapredictions = clf_logistic3.predict(pca_trainingdatasetquestions)
    logistictestpcapredictions = clf_logistic3.predict(pca_testingdatasetquestions)
    
    clf_logistic4 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    pcaseparated_ens_estimators.append(('log_reg', clf_logistic4))
    clf_logistic4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    logisticpcaseparatedpredictions = clf_logistic4.predict(pca_trainingdatasetquestionsseparated)
    logistictestpcaseparatedpredictions = clf_logistic4.predict(pca_testingdatasetquestionsseparated)
    
    clf_decisiontree0 = DecisionTreeClassifier(max_depth=20)
    individual_ens_estimators.append(('dt', clf_decisiontree0))
    clf_decisiontree0.fit(trainingdatasetquestions, trainingcorrectlabels)
    decisiontreequestionpredictions = clf_decisiontree0.predict(trainingdatasetquestions)
    decisiontreetestquestionpredictions = clf_decisiontree0.predict(testingdatasetquestions)
    
    clf_decisiontree1 = DecisionTreeClassifier(max_depth=20)
    total_ens_estimators.append(('dt', clf_decisiontree1))
    clf_decisiontree1.fit(trainingdatasettotals, trainingcorrectlabels)
    decisiontreepredictions = clf_decisiontree1.predict(trainingdatasettotals)
    decisiontreetestpredictions = clf_decisiontree1.predict(testingdatasettotals)
    
    clf_decisiontree2 = DecisionTreeClassifier(max_depth=20)
    preference_ens_estimators.append(('dt', clf_decisiontree2))
    clf_decisiontree2.fit(trainingdatasetpreferences, trainingcorrectlabels)
    decisiontreepreferencepredictions = clf_decisiontree2.predict(trainingdatasetpreferences)
    decisiontreetestpreferencepredictions = clf_decisiontree2.predict(testingdatasetpreferences)
    
    clf_decisiontree3 = DecisionTreeClassifier(max_depth=20)
    pca_ens_estimators.append(('dt', clf_decisiontree3))
    clf_decisiontree3.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    decisiontreepcapredictions = clf_decisiontree3.predict(pca_trainingdatasetquestions)
    decisiontreetestpcapredictions = clf_decisiontree3.predict(pca_testingdatasetquestions)
    
    clf_decisiontree4 = DecisionTreeClassifier(max_depth=20)
    pcaseparated_ens_estimators.append(('dt', clf_decisiontree4))
    clf_decisiontree4.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    decisiontreepcaseparatedpredictions = clf_decisiontree4.predict(pca_trainingdatasetquestionsseparated)
    decisiontreetestpcaseparatedpredictions = clf_decisiontree4.predict(pca_testingdatasetquestionsseparated)

    questions_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptquestionpredictions)

    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy))
    print("Able to predict openness based on individual training questions using perceptron with %{} accuracy".format(questions_perceptron_training_accuracy), file=file_out)

    questions_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testquestionpredictions)

    print("Able to predict openness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing questions using perceptron with %{} accuracy".format(questions_perceptron_testing_accuracy), file=file_out)

    totals_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, perceptpredictions)

    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy))
    print("Able to predict openness based on individual training totals using perceptron with %{} accuracy".format(totals_perceptron_training_accuracy), file=file_out)

    totals_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpredictions)

    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing totals using perceptron with %{} accuracy".format(totals_perceptron_testing_accuracy), file=file_out)

    preferences_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencepredictions)

    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy))
    print("Able to predict openness based on individual training preferences using perceptron with %{} accuracy".format(preferences_perceptron_training_accuracy), file=file_out)

    preferences_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using perceptron with %{} accuracy".format(preferences_perceptron_testing_accuracy), file=file_out)

    pca_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcapredictions)

    print("Able to predict openness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy))
    print("Able to predict openness based on individual training questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_training_accuracy), file=file_out)

    pca_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcapredictions)

    print("Able to predict openness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca without separation of questions using perceptron with %{} accuracy".format(pca_perceptron_testing_accuracy), file=file_out)

    pcaseparated_perceptron_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, pcaseparatedpredictions)

    print("Able to predict openness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy))
    print("Able to predict openness based on individual training questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_training_accuracy), file=file_out)

    pcaseparated_perceptron_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, testpcaseparatedpredictions)

    print("Able to predict openness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca with separation of questions using perceptron with %{} accuracy".format(pcaseparated_perceptron_testing_accuracy), file=file_out)

    questions_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdquestionpredictions)

    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy))
    print("Able to predict openness based on individual training questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_training_accuracy), file=file_out)

    questions_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestquestionpredictions)

    print("Able to predict openness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing questions using stochastic gradient descent with %{} accuracy".format(questions_sgd_testing_accuracy), file=file_out)

    totals_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpredictions)

    print("Able to predict openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy))
    print("Able to predict openness based on individual training totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_training_accuracy), file=file_out)

    totals_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpredictions)

    print("Able to predict openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing totals using stochastic gradient descent with %{} accuracy".format(totals_sgd_testing_accuracy), file=file_out)

    preferences_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpreferencepredictions)

    print("Able to predict openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy))
    print("Able to predict openness based on individual training preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_training_accuracy), file=file_out)

    preferences_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using stochastic gradient descent with %{} accuracy".format(preferences_sgd_testing_accuracy), file=file_out)

    pca_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcapredictions)

    print("Able to predict openness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy))
    print("Able to predict openness based on individual training questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_training_accuracy), file=file_out)

    pca_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcapredictions)

    print("Able to predict openness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca without separation of questions using stochastic gradient descent with %{} accuracy".format(pca_sgd_testing_accuracy), file=file_out)

    pcaseparated_sgd_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, sgdpcaseparatedpredictions)

    print("Able to predict openness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy))
    print("Able to predict openness based on individual training questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_training_accuracy), file=file_out)

    pcaseparated_sgd_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, sgdtestpcaseparatedpredictions)

    print("Able to predict openness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca with separation of questions using stochastic gradient descent with %{} accuracy".format(pcaseparated_sgd_testing_accuracy), file=file_out)

    questions_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticquestionpredictions)

    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy))
    print("Able to predict openness based on individual training questions using logistic regression with %{} accuracy".format(questions_logistic_training_accuracy), file=file_out)

    questions_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestquestionpredictions)

    print("Able to predict openness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing questions using logistic regression with %{} accuracy".format(questions_logistic_testing_accuracy), file=file_out)

    totals_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpredictions)

    print("Able to predict openness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy))
    print("Able to predict openness based on individual training totals using logistic regression with %{} accuracy".format(totals_logistic_training_accuracy), file=file_out)

    totals_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpredictions)

    print("Able to predict openness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing totals using logistic regression with %{} accuracy".format(totals_logistic_testing_accuracy), file=file_out)

    preference_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpreferencepredictions)

    print("Able to predict openness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy))
    print("Able to predict openness based on individual training preferences using logistic regression with %{} accuracy".format(preference_logistic_training_accuracy), file=file_out)

    preference_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using logistic regression with %{} accuracy".format(preference_logistic_testing_accuracy), file=file_out)

    pca_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcapredictions)

    print("Able to predict openness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy))
    print("Able to predict openness based on individual training questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_training_accuracy), file=file_out)

    pca_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcapredictions)

    print("Able to predict openness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca without separation of questions using logistic regression with %{} accuracy".format(pca_logistic_testing_accuracy), file=file_out)

    pcaseparated_logistic_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, logisticpcaseparatedpredictions)

    print("Able to predict openness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy))
    print("Able to predict openness based on individual training questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_training_accuracy), file=file_out)

    pcaseparated_logistic_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, logistictestpcaseparatedpredictions)

    print("Able to predict openness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca with separation of questions using logistic regression with %{} accuracy".format(pcaseparated_logistic_testing_accuracy), file=file_out)

    questions_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreequestionpredictions)

    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy))
    print("Able to predict openness based on individual training questions using decision tree with %{} accuracy".format(questions_dt_training_accuracy), file=file_out)

    questions_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestquestionpredictions)

    print("Able to predict openness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy))
    print("Able to predict openness based on individual testing questions using decision tree with %{} accuracy".format(questions_dt_testing_accuracy), file=file_out)

    totals_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepredictions)

    print("Able to predict openness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy))
    print("Able to predict openness based on individual training totals using decision tree with %{} accuracy".format(totals_dt_training_accuracy), file=file_out)

    totals_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpredictions)

    print("Able to predict openness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy))
    print("Able to predict openness based on individual testing totals using decision tree with %{} accuracy".format(totals_dt_testing_accuracy), file=file_out)

    preferences_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepreferencepredictions)

    print("Able to predict openness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy))
    print("Able to predict openness based on individual training preferences using decision tree with %{} accuracy".format(preferences_dt_training_accuracy), file=file_out)

    preferences_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpreferencepredictions)

    print("Able to predict openness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using decision tree with %{} accuracy".format(preferences_dt_testing_accuracy), file=file_out)

    pca_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcapredictions)

    print("Able to predict openness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy))
    print("Able to predict openness based on individual training questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_training_accuracy), file=file_out)

    pca_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcapredictions)

    print("Able to predict openness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca without separation of questions using decision tree with %{} accuracy".format(pca_dt_testing_accuracy), file=file_out)

    pcaseparated_dt_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, decisiontreepcaseparatedpredictions)

    print("Able to predict openness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy))
    print("Able to predict openness based on individual training questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_training_accuracy), file=file_out)

    pcaseparated_dt_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, decisiontreetestpcaseparatedpredictions)

    print("Able to predict openness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy))
    print("Able to predict openness based on individual testing questions after pca with separation of questions using decision tree with %{} accuracy".format(pcaseparated_dt_testing_accuracy), file=file_out)

    individual_voting = VotingClassifier(estimators=individual_ens_estimators)
    individual_voting.fit(trainingdatasetquestions, trainingcorrectlabels)
    votingquestionpredictions = individual_voting.predict(trainingdatasetquestions)

    questions_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingquestionpredictions)

    print("Able to predict openness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy))
    print("Able to predict openness based on individual training questions using ensemble of the above with %{} accuracy".format(questions_voting_training_accuracy), file=file_out)

    votingtestquestionpredictions = individual_voting.predict(testingdatasetquestions)
    questions_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestquestionpredictions)

    print("Able to predict openness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy))
    print("Able to predict openness based on individual testing questions using ensemble of the above with %{} accuracy".format(questions_voting_testing_accuracy), file=file_out)

    totals_voting = VotingClassifier(estimators=total_ens_estimators)
    totals_voting.fit(trainingdatasettotals, trainingcorrectlabels)
    votingtotalpredictions = totals_voting.predict(trainingdatasettotals)

    totals_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingtotalpredictions)

    print("Able to predict openness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy))
    print("Able to predict openness based on individual training totals using ensemble of the above with %{} accuracy".format(totals_voting_training_accuracy), file=file_out)

    votingtesttotalspredictions = totals_voting.predict(testingdatasettotals)
    totals_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtesttotalspredictions)

    print("Able to predict openness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy))
    print("Able to predict openness based on individual testing totals using ensemble of the above with %{} accuracy".format(totals_voting_testing_accuracy), file=file_out)

    preferences_voting = VotingClassifier(estimators=preference_ens_estimators)
    preferences_voting.fit(trainingdatasetpreferences, trainingcorrectlabels)
    preferencestotalpredictions = preferences_voting.predict(trainingdatasetpreferences)

    preferences_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, preferencestotalpredictions)

    print("Able to predict openness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy))
    print("Able to predict openness based on individual training preferences using ensemble of the above with %{} accuracy".format(preferences_voting_training_accuracy), file=file_out)

    votingtestpreferencespredictions = preferences_voting.predict(testingdatasetpreferences)
    preferences_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpreferencespredictions)

    print("Able to predict openness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy))
    print("Able to predict openness based on individual testing preferences using ensemble of the above with %{} accuracy".format(preferences_voting_testing_accuracy), file=file_out)

    pca_voting = VotingClassifier(estimators=pca_ens_estimators)
    pca_voting.fit(pca_trainingdatasetquestions, trainingcorrectlabels)
    votingpcapredictions = pca_voting.predict(pca_trainingdatasetquestions)

    pac_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcapredictions)

    print("Able to predict openness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy))
    print("Able to predict openness based on individual training pca questions using ensemble of the above with %{} accuracy".format(pac_voting_training_accuracy), file=file_out)

    votingtestpcapredictions = pca_voting.predict(pca_testingdatasetquestions)
    pca_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcapredictions)

    print("Able to predict openness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy))
    print("Able to predict openness based on individual testing pca questions using ensemble of the above with %{} accuracy".format(pca_voting_testing_accuracy), file=file_out)

    pcaseparated_voting = VotingClassifier(estimators=pcaseparated_ens_estimators)
    pcaseparated_voting.fit(pca_trainingdatasetquestionsseparated, trainingcorrectlabels)
    votingpcaseparatedpredictions = pcaseparated_voting.predict(pca_trainingdatasetquestionsseparated)

    pacseparated_voting_training_accuracy = metrics.accuracy_score(trainingcorrectlabels, votingpcaseparatedpredictions)

    print("Able to predict openness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy))
    print("Able to predict openness based on individual training pca separated questions using ensemble of the above with %{} accuracy".format(pacseparated_voting_training_accuracy), file=file_out)

    votingtestpcaseparatedpredictions = pcaseparated_voting.predict(pca_testingdatasetquestionsseparated)
    pcaseparated_voting_testing_accuracy = metrics.accuracy_score(testingcorrectlabels, votingtestpcaseparatedpredictions)

    print("Able to predict openness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy))
    print("Able to predict openness based on individual testing pca separated questions using ensemble of the above with %{} accuracy".format(pcaseparated_voting_testing_accuracy), file=file_out)

    return

def PredictBasedOnExtroversion(file_out, alltrainingdatasetquestions, alltrainingdatasettotals, alltrainingdatasetpreferences, alltestingdatasetquestions, alltestingdatasettotals, alltestingdatasetpreferences):

    extroversionquestions = alltrainingdatasetquestions[:,0:10] # Select extroversion question columns
    reducedextroversionquestionindeces = PerceptronForPruningMultiplePredictions(alltrainingdatasetquestions, alltrainingdatasetpreferences, (0, 10), [1, 2, 3, 4], 5, 1, 3)
    reducedextroversionquestions = extroversionquestions[:,reducedextroversionquestionindeces]

    neuroticismcorrectlabels = alltrainingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = alltrainingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = alltrainingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = alltrainingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testextroversionquestions = alltestingdatasetquestions[:,0:10] # Select extroversion question columns
    testreducedextroversionquestions = testextroversionquestions[:,reducedextroversionquestionindeces]
    testneuroticismcorrectlabels = alltestingdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestingdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestingdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestingdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    # NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(extroversionquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(extroversionquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testextroversionquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(extroversionquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testextroversionquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(extroversionquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testextroversionquestions)
    
    neuroticismclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree.fit(extroversionquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(extroversionquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testextroversionquestions)

    neuroticismclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3_reducedquestions.fit(reducedextroversionquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(reducedextroversionquestions)
    testneuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(testreducedextroversionquestions)
    
    neuroticismclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd_reducedquestions.fit(reducedextroversionquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(reducedextroversionquestions)
    neuroticismsgdtestpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(testreducedextroversionquestions)
    
    neuroticismclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic_reducedquestions.fit(reducedextroversionquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(reducedextroversionquestions)
    neuroticismlogistictestpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(testreducedextroversionquestions)
    
    neuroticismclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree_reducedquestions.fit(reducedextroversionquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(reducedextroversionquestions)
    neuroticismdecisiontreetestpredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(testreducedextroversionquestions)

    # Evaluation
    neuroticism_perceptron_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions)
    neuroticicm_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions), file=file_out)

    neuroticism_perceptron_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions)
    neuroticicm_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions)
    neuroticicm_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions)
    neuroticism_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions)
    neuroticicm_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions)
    neuroticism_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions)
    neuroticicm_dt_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy))
    print("Able to predict neuroticism based on extroversion training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions)
    neuroticism_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions_reducedquestions)

    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy))
    print("Able to predict neuroticism based on extroversion testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on extroversion testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on extroversion testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(extroversionquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(extroversionquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testextroversionquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(extroversionquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testextroversionquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(extroversionquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testextroversionquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(extroversionquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(extroversionquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testextroversionquestions)
    
    agreeablenessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3_reducedquestions.fit(reducedextroversionquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(reducedextroversionquestions)
    testagreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(testreducedextroversionquestions)
    
    agreeablenessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd_reducedquestions.fit(reducedextroversionquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(reducedextroversionquestions)
    agreeablenesssgdtestpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(testreducedextroversionquestions)
    
    agreeablenessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic_reducedquestions.fit(reducedextroversionquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(reducedextroversionquestions)
    agreeablenesslogistictestpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(testreducedextroversionquestions)
    
    agreeablenessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree_reducedquestions.fit(reducedextroversionquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(reducedextroversionquestions)
    agreeablenessdecisiontreetestpredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(testreducedextroversionquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)
    agreeableness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)
    agreeableness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)
    agreeableness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)
    agreeableness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)
    agreeableness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)
    agreeableness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)
    agreeableness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on extroversion training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)
    agreeableness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on extroversion testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions), file=file_out)

    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(extroversionquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(extroversionquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testextroversionquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(extroversionquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testextroversionquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(extroversionquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testextroversionquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(extroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(extroversionquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testextroversionquestions)

    conscientiousnessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3_reducedquestions.fit(reducedextroversionquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(reducedextroversionquestions)
    testconscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(testreducedextroversionquestions)
    
    conscientiousnessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd_reducedquestions.fit(reducedextroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(reducedextroversionquestions)
    conscientiousnesssgdtestpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(testreducedextroversionquestions)
    
    conscientiousnessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic_reducedquestions.fit(reducedextroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(reducedextroversionquestions)
    conscientiousnesslogistictestpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(testreducedextroversionquestions)
    
    conscientiousnessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree_reducedquestions.fit(reducedextroversionquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(reducedextroversionquestions)
    conscientiousnessdecisiontreetestpredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(testreducedextroversionquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)
    conscientiousness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)
    conscientiousness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)
    conscientiousness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)
    conscientiousness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)
    conscientiousness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)
    conscientiousness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)
    conscientiousness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on extroversion training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)
    conscientiousness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on extroversion testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(extroversionquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(extroversionquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testextroversionquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(extroversionquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(extroversionquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testextroversionquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(extroversionquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(extroversionquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testextroversionquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(extroversionquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(extroversionquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testextroversionquestions)

    opennessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3_reducedquestions.fit(reducedextroversionquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(reducedextroversionquestions)
    testopennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(testreducedextroversionquestions)
    
    opennessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd_reducedquestions.fit(reducedextroversionquestions, opennesscorrectlabels)
    opennesssgdpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(reducedextroversionquestions)
    opennesssgdtestpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(testreducedextroversionquestions)
    
    opennessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic_reducedquestions.fit(reducedextroversionquestions, opennesscorrectlabels)
    opennesslogisticpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(reducedextroversionquestions)
    opennesslogistictestpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(testreducedextroversionquestions)
    
    opennessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree_reducedquestions.fit(reducedextroversionquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(reducedextroversionquestions)
    opennessdecisiontreetestpredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(testreducedextroversionquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)
    openness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on extroversion training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)
    print("Able to predict openness based on extroversion training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)
    openness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict openness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)
    openness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions_reducedquestions)

    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on extroversion training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)
    print("Able to predict openness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)
    openness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions_reducedquestions)

    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)
    print("Able to predict openness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)
    openness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions_reducedquestions)

    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on extroversion training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)
    print("Able to predict openness based on extroversion training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)
    openness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions_reducedquestions)

    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)
    print("Able to predict openness based on extroversion testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)
    openness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions_reducedquestions)

    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on extroversion training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)
    print("Able to predict openness based on extroversion training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)
    openness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on extroversion testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)
    print("Able to predict openness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions))
    print("Able to predict openness based on extroversion testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions), file=file_out)

    return

def PredictBasedOnNeuroticism(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    neuroticismquestions = allsampledataset[:,10:20] # Select neuroticism question columns
    reducedneuroticismquestionindeces = PerceptronForPruningMultiplePredictions(allsampledataset, allsampledatasetpreferences, (10, 20), [0, 2, 3, 4], 5, 1, 3)
    reducedneuroticismquestions = neuroticismquestions[:,reducedneuroticismquestionindeces]

    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testneuroticismquestions = alltestdataset[:,10:20] # Select neuroticism question columns
    testreducedneuroticismquestions = testneuroticismquestions[:,reducedneuroticismquestionindeces]
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3.fit(neuroticismquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(neuroticismquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testneuroticismquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(neuroticismquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testneuroticismquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(neuroticismquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testneuroticismquestions)
    
    extroversionclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree.fit(neuroticismquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(neuroticismquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testneuroticismquestions)

    extroversionclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3_reducedquestions.fit(reducedneuroticismquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(reducedneuroticismquestions)
    testextroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(testreducedneuroticismquestions)
    
    extroversionclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd_reducedquestions.fit(reducedneuroticismquestions, extroversioncorrectlabels)
    extroversionsgdpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(reducedneuroticismquestions)
    extroversionsgdtestpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(testreducedneuroticismquestions)
    
    extroversionclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic_reducedquestions.fit(reducedneuroticismquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(reducedneuroticismquestions)
    extroversionlogistictestpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(testreducedneuroticismquestions)
    
    extroversionclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree_reducedquestions.fit(reducedneuroticismquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(reducedneuroticismquestions)
    extroversiondecisiontreetestpredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(testreducedneuroticismquestions)

    # Evaluation
    extroversion_perceptron_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions)
    extroversion_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions), file=file_out)

    extroversion_perceptron_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions)
    extroversion_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions)
    extroversion_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions)
    extroversion_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions)
    extroversion_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions)
    extroversion_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions), file=file_out)

    extroversion_dt_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions)
    extroversion_dt_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy))
    print("Able to predict extroversion based on neuroticism training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions), file=file_out)

    extroversion_dt_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions)
    extroversion_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions_reducedquestions)

    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy))
    print("Able to predict extroversion based on neuroticism testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(neuroticismquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(neuroticismquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testneuroticismquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(neuroticismquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testneuroticismquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(neuroticismquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testneuroticismquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(neuroticismquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(neuroticismquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testneuroticismquestions)
    
    agreeablenessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3_reducedquestions.fit(reducedneuroticismquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(reducedneuroticismquestions)
    testagreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(testreducedneuroticismquestions)
    
    agreeablenessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd_reducedquestions.fit(reducedneuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(reducedneuroticismquestions)
    agreeablenesssgdtestpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(testreducedneuroticismquestions)
    
    agreeablenessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic_reducedquestions.fit(reducedneuroticismquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(reducedneuroticismquestions)
    agreeablenesslogistictestpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(testreducedneuroticismquestions)
    
    agreeablenessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree_reducedquestions.fit(reducedneuroticismquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(reducedneuroticismquestions)
    agreeablenessdecisiontreetestpredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(testreducedneuroticismquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)
    agreeableness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on neuroticism training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)
    agreeableness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on neuroticism testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)
    agreeableness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)
    agreeableness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)
    agreeableness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on neuroticism training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)
    agreeableness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on neuroticism testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)
    agreeableness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on neuroticism training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)
    agreeableness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict agreeableness based on neuroticism testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on neuroticism testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions), file=file_out)

    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(neuroticismquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(neuroticismquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testneuroticismquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(neuroticismquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testneuroticismquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(neuroticismquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testneuroticismquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(neuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(neuroticismquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testneuroticismquestions)

    conscientiousnessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3_reducedquestions.fit(reducedneuroticismquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(reducedneuroticismquestions)
    testconscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(testreducedneuroticismquestions)
    
    conscientiousnessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd_reducedquestions.fit(reducedneuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(reducedneuroticismquestions)
    conscientiousnesssgdtestpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(testreducedneuroticismquestions)
    
    conscientiousnessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic_reducedquestions.fit(reducedneuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(reducedneuroticismquestions)
    conscientiousnesslogistictestpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(testreducedneuroticismquestions)
    
    conscientiousnessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree_reducedquestions.fit(reducedneuroticismquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(reducedneuroticismquestions)
    conscientiousnessdecisiontreetestpredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(testreducedneuroticismquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)
    conscientiousness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on neuroticism training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)
    conscientiousness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on neuroticism testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)
    conscientiousness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)
    conscientiousness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)
    conscientiousness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on neuroticism training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)
    conscientiousness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on neuroticism testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)
    conscientiousness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on neuroticism training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)
    conscientiousness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on neuroticism testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on neuroticism testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(neuroticismquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(neuroticismquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testneuroticismquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(neuroticismquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(neuroticismquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testneuroticismquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(neuroticismquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(neuroticismquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testneuroticismquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(neuroticismquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(neuroticismquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testneuroticismquestions)

    opennessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3_reducedquestions.fit(reducedneuroticismquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(reducedneuroticismquestions)
    testopennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(testreducedneuroticismquestions)
    
    opennessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd_reducedquestions.fit(reducedneuroticismquestions, opennesscorrectlabels)
    opennesssgdpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(reducedneuroticismquestions)
    opennesssgdtestpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(testreducedneuroticismquestions)
    
    opennessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic_reducedquestions.fit(reducedneuroticismquestions, opennesscorrectlabels)
    opennesslogisticpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(reducedneuroticismquestions)
    opennesslogistictestpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(testreducedneuroticismquestions)
    
    opennessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree_reducedquestions.fit(reducedneuroticismquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(reducedneuroticismquestions)
    opennessdecisiontreetestpredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(testreducedneuroticismquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)
    openness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on neuroticism training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)
    openness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on neuroticism testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)
    openness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on neuroticism training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)
    openness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on neuroticism testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)
    openness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on neuroticism training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)
    openness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on neuroticism testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)
    openness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions_reducedquestions)

    print("Able to predict openness based on neuroticism training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on neuroticism training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)
    openness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict openness based on neuroticism testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on neuroticism testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)
    print("Able to predict openness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions))
    print("Able to predict openness based on neuroticism testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions), file=file_out)

    return

def PredictBasedOnAgreeableness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    agreeablenessquestions = allsampledataset[:,20:30] # Select agreeableness question columns
    reducedagreeablenessquestionindeces = PerceptronForPruningMultiplePredictions(allsampledataset, allsampledatasetpreferences, (20, 30), [0, 1, 3, 4], 5, 1, 3)
    reducedagreeablenessquestions = agreeablenessquestions[:,reducedagreeablenessquestionindeces]

    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testagreeablenessquestions = alltestdataset[:,20:30] # Select agreeableness question columns
    testreducedagreeablenessquestions = testagreeablenessquestions[:,reducedagreeablenessquestionindeces]
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for conscientiousness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3.fit(agreeablenessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(agreeablenessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testagreeablenessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(agreeablenessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testagreeablenessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(agreeablenessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testagreeablenessquestions)
    
    extroversionclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree.fit(agreeablenessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(agreeablenessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testagreeablenessquestions)

    extroversionclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3_reducedquestions.fit(reducedagreeablenessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(reducedagreeablenessquestions)
    testextroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(testreducedagreeablenessquestions)
    
    extroversionclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd_reducedquestions.fit(reducedagreeablenessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(reducedagreeablenessquestions)
    extroversionsgdtestpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(testreducedagreeablenessquestions)
    
    extroversionclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic_reducedquestions.fit(reducedagreeablenessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(reducedagreeablenessquestions)
    extroversionlogistictestpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(testreducedagreeablenessquestions)
    
    extroversionclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree_reducedquestions.fit(reducedagreeablenessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(reducedagreeablenessquestions)
    extroversiondecisiontreetestpredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(testreducedagreeablenessquestions)

    # Evaluation
    extroversion_perceptron_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions)
    extroversion_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy))
    print("Able to predict extroversion based on agreeableness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions), file=file_out)

    extroversion_perceptron_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions)
    extroversion_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy))
    print("Able to predict extroversion based on agreeableness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions)
    extroversion_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy))
    print("Able to predict extroversion based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions)
    extroversion_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy))
    print("Able to predict extroversion based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions)
    extroversion_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy))
    print("Able to predict extroversion based on agreeableness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions)
    extroversion_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy))
    print("Able to predict extroversion based on agreeableness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions), file=file_out)

    extroversion_dt_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions)
    extroversion_dt_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy))
    print("Able to predict extroversion based on agreeableness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions), file=file_out)

    extroversion_dt_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions)
    extroversion_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions_reducedquestions)

    print("Able to predict extroversion based on agreeableness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy))
    print("Able to predict extroversion based on agreeableness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions), file=file_out)

    # NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(agreeablenessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(agreeablenessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testagreeablenessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(agreeablenessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testagreeablenessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(agreeablenessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testagreeablenessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree.fit(agreeablenessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(agreeablenessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testagreeablenessquestions)

    neuroticismclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3_reducedquestions.fit(reducedagreeablenessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(reducedagreeablenessquestions)
    testneuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(testreducedagreeablenessquestions)
    
    neuroticismclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd_reducedquestions.fit(reducedagreeablenessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(reducedagreeablenessquestions)
    neuroticismsgdtestpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(testreducedagreeablenessquestions)
    
    neuroticismclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic_reducedquestions.fit(reducedagreeablenessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(reducedagreeablenessquestions)
    neuroticismlogistictestpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(testreducedagreeablenessquestions)
    
    neuroticismclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree_reducedquestions.fit(reducedagreeablenessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(reducedagreeablenessquestions)
    neuroticismdecisiontreetestpredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(testreducedagreeablenessquestions)

    # Evaluation
    neuroticism_perceptron_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions)
    neuroticicm_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy))
    print("Able to predict neuroticism based on agreeableness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions), file=file_out)

    neuroticism_perceptron_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions)
    neuroticicm_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on agreeableness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions)
    neuroticicm_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy))
    print("Able to predict neuroticism based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions)
    neuroticism_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy))
    print("Able to predict neuroticism based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions)
    neuroticicm_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy))
    print("Able to predict neuroticism based on agreeableness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions)
    neuroticism_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy))
    print("Able to predict neuroticism based on agreeableness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions)
    neuroticicm_dt_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy))
    print("Able to predict neuroticism based on agreeableness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions)
    neuroticism_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions_reducedquestions)

    print("Able to predict neuroticism based on agreeableness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy))
    print("Able to predict neuroticism based on agreeableness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions), file=file_out)

    #CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(agreeablenessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(agreeablenessquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testagreeablenessquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(agreeablenessquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testagreeablenessquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(agreeablenessquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testagreeablenessquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(agreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(agreeablenessquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testagreeablenessquestions)

    conscientiousnessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3_reducedquestions.fit(reducedagreeablenessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(reducedagreeablenessquestions)
    testconscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(testreducedagreeablenessquestions)
    
    conscientiousnessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd_reducedquestions.fit(reducedagreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(reducedagreeablenessquestions)
    conscientiousnesssgdtestpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(testreducedagreeablenessquestions)
    
    conscientiousnessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic_reducedquestions.fit(reducedagreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(reducedagreeablenessquestions)
    conscientiousnesslogistictestpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(testreducedagreeablenessquestions)
    
    conscientiousnessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree_reducedquestions.fit(reducedagreeablenessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(reducedagreeablenessquestions)
    conscientiousnessdecisiontreetestpredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(testreducedagreeablenessquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)
    conscientiousness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on agreeableness training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)
    conscientiousness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on agreeableness testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)
    conscientiousness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)
    conscientiousness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)
    conscientiousness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on agreeableness training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)
    conscientiousness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on agreeableness testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)
    conscientiousness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on agreeableness training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)
    conscientiousness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on agreeableness testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on agreeableness testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(agreeablenessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(agreeablenessquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testagreeablenessquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(agreeablenessquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(agreeablenessquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testagreeablenessquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(agreeablenessquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(agreeablenessquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testagreeablenessquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(agreeablenessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(agreeablenessquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testagreeablenessquestions)

    opennessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3_reducedquestions.fit(reducedagreeablenessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(reducedagreeablenessquestions)
    testopennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(testreducedagreeablenessquestions)
    
    opennessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd_reducedquestions.fit(reducedagreeablenessquestions, opennesscorrectlabels)
    opennesssgdpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(reducedagreeablenessquestions)
    opennesssgdtestpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(testreducedagreeablenessquestions)
    
    opennessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic_reducedquestions.fit(reducedagreeablenessquestions, opennesscorrectlabels)
    opennesslogisticpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(reducedagreeablenessquestions)
    opennesslogistictestpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(testreducedagreeablenessquestions)
    
    opennessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree_reducedquestions.fit(reducedagreeablenessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(reducedagreeablenessquestions)
    opennessdecisiontreetestpredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(testreducedagreeablenessquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)
    openness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on agreeableness training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)
    openness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on agreeableness testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)
    openness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on agreeableness training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)
    openness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on agreeableness testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)
    openness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on agreeableness training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)
    openness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on agreeableness testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)
    openness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions_reducedquestions)

    print("Able to predict openness based on agreeableness training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on agreeableness training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)
    openness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict openness based on agreeableness testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on agreeableness testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)
    print("Able to predict openness based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions))
    print("Able to predict openness based on agreeableness testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions), file=file_out)

    return

def PredictBasedOnConscientiousness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    conscientiousnessquestions = allsampledataset[:,30:40] # Select conscientiousness question columns
    reducedconscientiousnessquestionindeces = PerceptronForPruningMultiplePredictions(allsampledataset, allsampledatasetpreferences, (30, 40), [0, 1, 2, 4], 5, 1, 3)
    reducedconscientiousnessquestions = conscientiousnessquestions[:,reducedconscientiousnessquestionindeces]
    
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    opennesscorrectlabels = allsampledatasetpreferences[:,4] # Select preferences for openness corresponding to columns
    
    testconscientiousnessquestions = alltestdataset[:,30:40] # Select conscientiousness question columns
    testreducedconscientiousnessquestions = testconscientiousnessquestions[:,reducedconscientiousnessquestionindeces]
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testopennesscorrectlabels = alltestdatasetpreferences[:,4] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3.fit(conscientiousnessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(conscientiousnessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testconscientiousnessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(conscientiousnessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testconscientiousnessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(conscientiousnessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testconscientiousnessquestions)
    
    extroversionclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree.fit(conscientiousnessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(conscientiousnessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testconscientiousnessquestions)

    extroversionclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3_reducedquestions.fit(reducedconscientiousnessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(reducedconscientiousnessquestions)
    testextroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    extroversionclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd_reducedquestions.fit(reducedconscientiousnessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(reducedconscientiousnessquestions)
    extroversionsgdtestpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    extroversionclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic_reducedquestions.fit(reducedconscientiousnessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(reducedconscientiousnessquestions)
    extroversionlogistictestpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    extroversionclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree_reducedquestions.fit(reducedconscientiousnessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(reducedconscientiousnessquestions)
    extroversiondecisiontreetestpredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(testreducedconscientiousnessquestions)

    # Evaluation
    extroversion_perceptron_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions)
    extroversion_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy))
    print("Able to predict extroversion based on conscientiousness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions), file=file_out)

    extroversion_perceptron_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions)
    extroversion_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy))
    print("Able to predict extroversion based on conscientiousness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions)
    extroversion_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy))
    print("Able to predict extroversion based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions)
    extroversion_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy))
    print("Able to predict extroversion based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions)
    extroversion_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy))
    print("Able to predict extroversion based on conscientiousness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions)
    extroversion_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy))
    print("Able to predict extroversion based on conscientiousness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions), file=file_out)

    extroversion_dt_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions)
    extroversion_dt_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy))
    print("Able to predict extroversion based on conscientiousness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions), file=file_out)

    extroversion_dt_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions)
    extroversion_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions_reducedquestions)

    print("Able to predict extroversion based on conscientiousness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy))
    print("Able to predict extroversion based on conscientiousness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions), file=file_out)

    # NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(conscientiousnessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(conscientiousnessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testconscientiousnessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(conscientiousnessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testconscientiousnessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(conscientiousnessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testconscientiousnessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree.fit(conscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(conscientiousnessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testconscientiousnessquestions)

    neuroticismclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3_reducedquestions.fit(reducedconscientiousnessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(reducedconscientiousnessquestions)
    testneuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    neuroticismclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd_reducedquestions.fit(reducedconscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(reducedconscientiousnessquestions)
    neuroticismsgdtestpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    neuroticismclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic_reducedquestions.fit(reducedconscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(reducedconscientiousnessquestions)
    neuroticismlogistictestpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    neuroticismclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree_reducedquestions.fit(reducedconscientiousnessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(reducedconscientiousnessquestions)
    neuroticismdecisiontreetestpredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(testreducedconscientiousnessquestions)

    # Evaluation
    neuroticism_perceptron_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions)
    neuroticicm_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy))
    print("Able to predict neuroticism based on conscientiousness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions), file=file_out)

    neuroticism_perceptron_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions)
    neuroticicm_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on conscientiousness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions)
    neuroticicm_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy))
    print("Able to predict neuroticism based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions)
    neuroticism_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy))
    print("Able to predict neuroticism based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions)
    neuroticicm_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy))
    print("Able to predict neuroticism based on conscientiousness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions)
    neuroticism_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy))
    print("Able to predict neuroticism based on conscientiousness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions)
    neuroticicm_dt_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy))
    print("Able to predict neuroticism based on conscientiousness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions)
    neuroticism_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions_reducedquestions)

    print("Able to predict neuroticism based on conscientiousness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy))
    print("Able to predict neuroticism based on conscientiousness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(conscientiousnessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(conscientiousnessquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testconscientiousnessquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(conscientiousnessquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testconscientiousnessquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(conscientiousnessquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testconscientiousnessquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(conscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(conscientiousnessquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testconscientiousnessquestions)
    
    agreeablenessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3_reducedquestions.fit(reducedconscientiousnessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(reducedconscientiousnessquestions)
    testagreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    agreeablenessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd_reducedquestions.fit(reducedconscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(reducedconscientiousnessquestions)
    agreeablenesssgdtestpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    agreeablenessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic_reducedquestions.fit(reducedconscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(reducedconscientiousnessquestions)
    agreeablenesslogistictestpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    agreeablenessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree_reducedquestions.fit(reducedconscientiousnessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(reducedconscientiousnessquestions)
    agreeablenessdecisiontreetestpredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(testreducedconscientiousnessquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)
    agreeableness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on conscientiousness training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)
    agreeableness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on conscientiousness testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)
    agreeableness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)
    agreeableness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)
    agreeableness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on conscientiousness training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)
    agreeableness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on conscientiousness testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)
    agreeableness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on conscientiousness training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)
    agreeableness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict agreeableness based on conscientiousness testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on conscientiousness testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions), file=file_out)

    #OPENNESS
    opennessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3.fit(conscientiousnessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions = opennessclf_percept3.predict(conscientiousnessquestions)
    testopennessquestionpredictions = opennessclf_percept3.predict(testconscientiousnessquestions)
    
    opennessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennesssgdpredictions = opennessclf_sgd.predict(conscientiousnessquestions)
    opennesssgdtestpredictions = opennessclf_sgd.predict(testconscientiousnessquestions)
    
    opennessclf_logistic = linear_model.LogisticRegression(penalty='l2', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennesslogisticpredictions = opennessclf_logistic.predict(conscientiousnessquestions)
    opennesslogistictestpredictions = opennessclf_logistic.predict(testconscientiousnessquestions)
    
    opennessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree.fit(conscientiousnessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions = opennessclf_decisiontree.predict(conscientiousnessquestions)
    opennessdecisiontreetestpredictions = opennessclf_decisiontree.predict(testconscientiousnessquestions)

    opennessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    opennessclf_percept3_reducedquestions.fit(reducedconscientiousnessquestions, opennesscorrectlabels, sample_weight=None)
    opennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(reducedconscientiousnessquestions)
    testopennessquestionpredictions_reducedquestions = opennessclf_percept3_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    opennessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    opennessclf_sgd_reducedquestions.fit(reducedconscientiousnessquestions, opennesscorrectlabels)
    opennesssgdpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(reducedconscientiousnessquestions)
    opennesssgdtestpredictions_reducedquestions = opennessclf_sgd_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    opennessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    opennessclf_logistic_reducedquestions.fit(reducedconscientiousnessquestions, opennesscorrectlabels)
    opennesslogisticpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(reducedconscientiousnessquestions)
    opennesslogistictestpredictions_reducedquestions = opennessclf_logistic_reducedquestions.predict(testreducedconscientiousnessquestions)
    
    opennessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    opennessclf_decisiontree_reducedquestions.fit(reducedconscientiousnessquestions, opennesscorrectlabels)
    opennessdecisiontreepredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(reducedconscientiousnessquestions)
    opennessdecisiontreetestpredictions_reducedquestions = opennessclf_decisiontree_reducedquestions.predict(testreducedconscientiousnessquestions)

    # Evaluation
    openness_perceptron_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions)
    openness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy))
    print("Able to predict openness based on conscientiousness training questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness training reduced questions using perceptron with %{} accuracy".format(openness_perceptron_training_accuracy_reducedquestions), file=file_out)

    openness_perceptron_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions)
    openness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, testopennessquestionpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy))
    print("Able to predict openness based on conscientiousness testing questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness testing reduced questions using perceptron with %{} accuracy".format(openness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    openness_sgd_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions)
    openness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesssgdpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy))
    print("Able to predict openness based on conscientiousness training questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness training reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_training_accuracy_reducedquestions), file=file_out)

    openness_sgd_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions)
    openness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesssgdtestpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy))
    print("Able to predict openness based on conscientiousness testing questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness testing reduced questions using stochastic gradient descent with %{} accuracy".format(openness_sgd_testing_accuracy_reducedquestions), file=file_out)

    openness_logistic_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions)
    openness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennesslogisticpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy))
    print("Able to predict openness based on conscientiousness training questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness training reduced questions using logistic with %{} accuracy".format(openness_logistic_training_accuracy_reducedquestions), file=file_out)

    openness_logistic_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions)
    openness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennesslogistictestpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy))
    print("Able to predict openness based on conscientiousness testing questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness testing reduced questions using logistic with %{} accuracy".format(openness_logistic_testing_accuracy_reducedquestions), file=file_out)

    openness_dt_training_accuracy = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions)
    openness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(opennesscorrectlabels, opennessdecisiontreepredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy))
    print("Able to predict openness based on conscientiousness training questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness training reduced questions using decision tree with %{} accuracy".format(openness_dt_training_accuracy_reducedquestions), file=file_out)

    openness_dt_testing_accuracy = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions)
    openness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testopennesscorrectlabels, opennessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict openness based on conscientiousness testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy))
    print("Able to predict openness based on conscientiousness testing questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy), file=file_out)
    print("Able to predict openness based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions))
    print("Able to predict openness based on conscientiousness testing reduced questions using decision tree with %{} accuracy".format(openness_dt_testing_accuracy_reducedquestions), file=file_out)

    return

def PredictBasedOnOpenness(file_out, allsampledataset, allsampledatasettotals, allsampledatasetpreferences, alltestdataset, alltestdatasettotals, alltestdatasetpreferences):

    opennessquestions = allsampledataset[:,30:40] # Select openness question columns
    reducedopennessquestionindeces = PerceptronForPruningMultiplePredictions(allsampledataset, allsampledatasetpreferences, (40, 50), [0, 1, 2, 3], 5, 1, 3)
    reducedopennessquestions = opennessquestions[:,reducedopennessquestionindeces]
    
    extroversioncorrectlabels = allsampledatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    neuroticismcorrectlabels = allsampledatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    agreeablenesscorrectlabels = allsampledatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    conscientiousnesscorrectlabels = allsampledatasetpreferences[:,3] # Select preferences for openness corresponding to columns
    
    testopennessquestions = alltestdataset[:,40:50] # Select openness question columns
    testreducedopennessquestions = testopennessquestions[:,reducedopennessquestionindeces]
    testextroversioncorrectlabels = alltestdatasetpreferences[:,0] # Select preferences for extroversion corresponding to columns
    testneuroticismcorrectlabels = alltestdatasetpreferences[:,1] # Select preferences for neuroticism corresponding to columns
    testagreeablenesscorrectlabels = alltestdatasetpreferences[:,2] # Select preferences for agreeableness corresponding to columns
    testconscientiousnesscorrectlabels = alltestdatasetpreferences[:,3] # Select preferences for openness corresponding to columns

    #EXTROVERSION
    extroversionclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3.fit(opennessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions = extroversionclf_percept3.predict(opennessquestions)
    testextroversionquestionpredictions = extroversionclf_percept3.predict(testopennessquestions)
    
    extroversionclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd.fit(opennessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions = extroversionclf_sgd.predict(opennessquestions)
    extroversionsgdtestpredictions = extroversionclf_sgd.predict(testopennessquestions)
    
    extroversionclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic.fit(opennessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions = extroversionclf_logistic.predict(opennessquestions)
    extroversionlogistictestpredictions = extroversionclf_logistic.predict(testopennessquestions)
    
    extroversionclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree.fit(opennessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions = extroversionclf_decisiontree.predict(opennessquestions)
    extroversiondecisiontreetestpredictions = extroversionclf_decisiontree.predict(testopennessquestions)

    extroversionclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    extroversionclf_percept3_reducedquestions.fit(reducedopennessquestions, extroversioncorrectlabels, sample_weight=None)
    extroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(reducedopennessquestions)
    testextroversionquestionpredictions_reducedquestions = extroversionclf_percept3_reducedquestions.predict(testreducedopennessquestions)
    
    extroversionclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    extroversionclf_sgd_reducedquestions.fit(reducedopennessquestions, extroversioncorrectlabels)
    extroversionsgdpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(reducedopennessquestions)
    extroversionsgdtestpredictions_reducedquestions = extroversionclf_sgd_reducedquestions.predict(testreducedopennessquestions)
    
    extroversionclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    extroversionclf_logistic_reducedquestions.fit(reducedopennessquestions, extroversioncorrectlabels)
    extroversionlogisticpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(reducedopennessquestions)
    extroversionlogistictestpredictions_reducedquestions = extroversionclf_logistic_reducedquestions.predict(testreducedopennessquestions)
    
    extroversionclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    extroversionclf_decisiontree_reducedquestions.fit(reducedopennessquestions, extroversioncorrectlabels)
    extroversiondecisiontreepredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(reducedopennessquestions)
    extroversiondecisiontreetestpredictions_reducedquestions = extroversionclf_decisiontree_reducedquestions.predict(testreducedopennessquestions)

    # Evaluation
    extroversion_perceptron_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions)
    extroversion_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on openness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy))
    print("Able to predict extroversion based on openness training questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy), file=file_out)
    print("Able to predict extroversion based on openness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness training reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_training_accuracy_reducedquestions), file=file_out)

    extroversion_perceptron_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions)
    extroversion_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, testextroversionquestionpredictions_reducedquestions)

    print("Able to predict extroversion based on openness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy))
    print("Able to predict extroversion based on openness testing questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on openness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness testing reduced questions using perceptron with %{} accuracy".format(extroversion_perceptron_testing_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions)
    extroversion_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionsgdpredictions_reducedquestions)

    print("Able to predict extroversion based on openness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy))
    print("Able to predict extroversion based on openness training questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy), file=file_out)
    print("Able to predict extroversion based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_training_accuracy_reducedquestions), file=file_out)

    extroversion_sgd_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions)
    extroversion_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionsgdtestpredictions_reducedquestions)

    print("Able to predict extroversion based on openness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy))
    print("Able to predict extroversion based on openness testing questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(extroversion_sgd_testing_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions)
    extroversion_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversionlogisticpredictions_reducedquestions)

    print("Able to predict extroversion based on openness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy))
    print("Able to predict extroversion based on openness training questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy), file=file_out)
    print("Able to predict extroversion based on openness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness training reduced questions using logistic with %{} accuracy".format(extroversion_logistic_training_accuracy_reducedquestions), file=file_out)

    extroversion_logistic_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions)
    extroversion_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversionlogistictestpredictions_reducedquestions)

    print("Able to predict extroversion based on openness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy))
    print("Able to predict extroversion based on openness testing questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on openness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness testing reduced questions using logistic with %{} accuracy".format(extroversion_logistic_testing_accuracy_reducedquestions), file=file_out)

    extroversion_dt_training_accuracy = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions)
    extroversion_dt_training_accuracy_reducedquestions = metrics.accuracy_score(extroversioncorrectlabels, extroversiondecisiontreepredictions_reducedquestions)

    print("Able to predict extroversion based on openness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy))
    print("Able to predict extroversion based on openness training questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy), file=file_out)
    print("Able to predict extroversion based on openness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness training reduced questions using decision tree with %{} accuracy".format(extroversion_dt_training_accuracy_reducedquestions), file=file_out)

    extroversion_dt_testing_accuracy = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions)
    extroversion_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testextroversioncorrectlabels, extroversiondecisiontreetestpredictions_reducedquestions)

    print("Able to predict extroversion based on openness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy))
    print("Able to predict extroversion based on openness testing questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy), file=file_out)
    print("Able to predict extroversion based on openness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions))
    print("Able to predict extroversion based on openness testing reduced questions using decision tree with %{} accuracy".format(extroversion_dt_testing_accuracy_reducedquestions), file=file_out)

    # NEUROTICISM
    neuroticismclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3.fit(opennessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions = neuroticismclf_percept3.predict(opennessquestions)
    testneuroticismquestionpredictions = neuroticismclf_percept3.predict(testopennessquestions)
    
    neuroticismclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions = neuroticismclf_sgd.predict(opennessquestions)
    neuroticismsgdtestpredictions = neuroticismclf_sgd.predict(testopennessquestions)
    
    neuroticismclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions = neuroticismclf_logistic.predict(opennessquestions)
    neuroticismlogistictestpredictions = neuroticismclf_logistic.predict(testopennessquestions)
    
    neuroticismclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree.fit(opennessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions = neuroticismclf_decisiontree.predict(opennessquestions)
    neuroticismdecisiontreetestpredictions = neuroticismclf_decisiontree.predict(testopennessquestions)

    neuroticismclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    neuroticismclf_percept3_reducedquestions.fit(reducedopennessquestions, neuroticismcorrectlabels, sample_weight=None)
    neuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(reducedopennessquestions)
    testneuroticismquestionpredictions_reducedquestions = neuroticismclf_percept3_reducedquestions.predict(testreducedopennessquestions)
    
    neuroticismclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    neuroticismclf_sgd_reducedquestions.fit(reducedopennessquestions, neuroticismcorrectlabels)
    neuroticismsgdpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(reducedopennessquestions)
    neuroticismsgdtestpredictions_reducedquestions = neuroticismclf_sgd_reducedquestions.predict(testreducedopennessquestions)
    
    neuroticismclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    neuroticismclf_logistic_reducedquestions.fit(reducedopennessquestions, neuroticismcorrectlabels)
    neuroticismlogisticpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(reducedopennessquestions)
    neuroticismlogistictestpredictions_reducedquestions = neuroticismclf_logistic_reducedquestions.predict(testreducedopennessquestions)
    
    neuroticismclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    neuroticismclf_decisiontree_reducedquestions.fit(reducedopennessquestions, neuroticismcorrectlabels)
    neuroticismdecisiontreepredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(reducedopennessquestions)
    neuroticismdecisiontreetestpredictions_reducedquestions = neuroticismclf_decisiontree_reducedquestions.predict(testreducedopennessquestions)

    # Evaluation
    neuroticism_perceptron_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions)
    neuroticicm_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy))
    print("Able to predict neuroticism based on openness training questions using perceptron with %{} accuracy".format(neuroticism_perceptron_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness training reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_training_accuracy_reducedquestions), file=file_out)

    neuroticism_perceptron_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions)
    neuroticicm_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, testneuroticismquestionpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy))
    print("Able to predict neuroticism based on openness testing questions using perceptron with %{} accuracy".format(neuroticism_perceptron_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness testing reduced questions using perceptron with %{} accuracy".format(neuroticicm_perceptron_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions)
    neuroticicm_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismsgdpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy))
    print("Able to predict neuroticism based on openness training questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticicm_sgd_training_accuracy_reducedquestions), file=file_out)

    neuroticism_sgd_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions)
    neuroticism_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismsgdtestpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy))
    print("Able to predict neuroticism based on openness testing questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(neuroticism_sgd_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions)
    neuroticicm_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismlogisticpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy))
    print("Able to predict neuroticism based on openness training questions using logistic with %{} accuracy".format(neuroticism_logistic_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness training reduced questions using logistic with %{} accuracy".format(neuroticicm_logistic_training_accuracy_reducedquestions), file=file_out)

    neuroticism_logistic_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions)
    neuroticism_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismlogistictestpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy))
    print("Able to predict neuroticism based on openness testing questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness testing reduced questions using logistic with %{} accuracy".format(neuroticism_logistic_testing_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_training_accuracy = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions)
    neuroticicm_dt_training_accuracy_reducedquestions = metrics.accuracy_score(neuroticismcorrectlabels, neuroticismdecisiontreepredictions_reducedquestions)

    print("Able to predict neuroticism based on openness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy))
    print("Able to predict neuroticism based on openness training questions using decision tree with %{} accuracy".format(neuroticism_dt_training_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness training reduced questions using decision tree with %{} accuracy".format(neuroticicm_dt_training_accuracy_reducedquestions), file=file_out)

    neuroticism_dt_testing_accuracy = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions)
    neuroticism_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testneuroticismcorrectlabels, neuroticismdecisiontreetestpredictions_reducedquestions)

    print("Able to predict neuroticism based on openness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy))
    print("Able to predict neuroticism based on openness testing questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy), file=file_out)
    print("Able to predict neuroticism based on openness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions))
    print("Able to predict neuroticism based on openness testing reduced questions using decision tree with %{} accuracy".format(neuroticism_dt_testing_accuracy_reducedquestions), file=file_out)

    #AGREEABLENESS
    agreeablenessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3.fit(opennessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions = agreeablenessclf_percept3.predict(opennessquestions)
    testagreeablenessquestionpredictions = agreeablenessclf_percept3.predict(testopennessquestions)
    
    agreeablenessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions = agreeablenessclf_sgd.predict(opennessquestions)
    agreeablenesssgdtestpredictions = agreeablenessclf_sgd.predict(testopennessquestions)
    
    agreeablenessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions = agreeablenessclf_logistic.predict(opennessquestions)
    agreeablenesslogistictestpredictions = agreeablenessclf_logistic.predict(testopennessquestions)
    
    agreeablenessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree.fit(opennessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions = agreeablenessclf_decisiontree.predict(opennessquestions)
    agreeablenessdecisiontreetestpredictions = agreeablenessclf_decisiontree.predict(testopennessquestions)
    
    agreeablenessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    agreeablenessclf_percept3_reducedquestions.fit(reducedopennessquestions, agreeablenesscorrectlabels, sample_weight=None)
    agreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(reducedopennessquestions)
    testagreeablenessquestionpredictions_reducedquestions = agreeablenessclf_percept3_reducedquestions.predict(testreducedopennessquestions)
    
    agreeablenessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    agreeablenessclf_sgd_reducedquestions.fit(reducedopennessquestions, agreeablenesscorrectlabels)
    agreeablenesssgdpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(reducedopennessquestions)
    agreeablenesssgdtestpredictions_reducedquestions = agreeablenessclf_sgd_reducedquestions.predict(testreducedopennessquestions)
    
    agreeablenessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    agreeablenessclf_logistic_reducedquestions.fit(reducedopennessquestions, agreeablenesscorrectlabels)
    agreeablenesslogisticpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(reducedopennessquestions)
    agreeablenesslogistictestpredictions_reducedquestions = agreeablenessclf_logistic_reducedquestions.predict(testreducedopennessquestions)
    
    agreeablenessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    agreeablenessclf_decisiontree_reducedquestions.fit(reducedopennessquestions, agreeablenesscorrectlabels)
    agreeablenessdecisiontreepredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(reducedopennessquestions)
    agreeablenessdecisiontreetestpredictions_reducedquestions = agreeablenessclf_decisiontree_reducedquestions.predict(testreducedopennessquestions)

    # Evaluation
    agreeableness_perceptron_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions)
    agreeableness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy))
    print("Able to predict agreeableness based on openness training questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness training reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_training_accuracy_reducedquestions), file=file_out)

    agreeableness_perceptron_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions)
    agreeableness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, testagreeablenessquestionpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy))
    print("Able to predict agreeableness based on openness testing questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness testing reduced questions using perceptron with %{} accuracy".format(agreeableness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions)
    agreeableness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesssgdpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy))
    print("Able to predict agreeableness based on openness training questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_training_accuracy_reducedquestions), file=file_out)

    agreeableness_sgd_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions)
    agreeableness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesssgdtestpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy))
    print("Able to predict agreeableness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(agreeableness_sgd_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions)
    agreeableness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenesslogisticpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy))
    print("Able to predict agreeableness based on openness training questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness training reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_training_accuracy_reducedquestions), file=file_out)

    agreeableness_logistic_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions)
    agreeableness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenesslogistictestpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy))
    print("Able to predict agreeableness based on openness testing questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness testing reduced questions using logistic with %{} accuracy".format(agreeableness_logistic_testing_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_training_accuracy = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions)
    agreeableness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(agreeablenesscorrectlabels, agreeablenessdecisiontreepredictions_reducedquestions)

    print("Able to predict agreeableness based on openness training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy))
    print("Able to predict agreeableness based on openness training questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness training reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_training_accuracy_reducedquestions), file=file_out)

    agreeableness_dt_testing_accuracy = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions)
    agreeableness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testagreeablenesscorrectlabels, agreeablenessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict agreeableness based on openness testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy))
    print("Able to predict agreeableness based on openness testing questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy), file=file_out)
    print("Able to predict agreeableness based on openness testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions))
    print("Able to predict agreeableness based on openness testing reduced questions using decision tree with %{} accuracy".format(agreeableness_dt_testing_accuracy_reducedquestions), file=file_out)

    # CONSCIENTIOUSNESS
    conscientiousnessclf_percept3 = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3.fit(opennessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(opennessquestions)
    testconscientiousnessquestionpredictions = conscientiousnessclf_percept3.predict(testopennessquestions)
    
    conscientiousnessclf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions = conscientiousnessclf_sgd.predict(opennessquestions)
    conscientiousnesssgdtestpredictions = conscientiousnessclf_sgd.predict(testopennessquestions)
    
    conscientiousnessclf_logistic = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions = conscientiousnessclf_logistic.predict(opennessquestions)
    conscientiousnesslogistictestpredictions = conscientiousnessclf_logistic.predict(testopennessquestions)
    
    conscientiousnessclf_decisiontree = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree.fit(opennessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions = conscientiousnessclf_decisiontree.predict(opennessquestions)
    conscientiousnessdecisiontreetestpredictions = conscientiousnessclf_decisiontree.predict(testopennessquestions)

    conscientiousnessclf_percept3_reducedquestions = Perceptron(max_iter=20, random_state=0, eta0=1)
    conscientiousnessclf_percept3_reducedquestions.fit(reducedopennessquestions, conscientiousnesscorrectlabels, sample_weight=None)
    conscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(reducedopennessquestions)
    testconscientiousnessquestionpredictions_reducedquestions = conscientiousnessclf_percept3_reducedquestions.predict(testreducedopennessquestions)
    
    conscientiousnessclf_sgd_reducedquestions = SGDClassifier(loss="hinge", penalty="l2", max_iter=20)
    conscientiousnessclf_sgd_reducedquestions.fit(reducedopennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesssgdpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(reducedopennessquestions)
    conscientiousnesssgdtestpredictions_reducedquestions = conscientiousnessclf_sgd_reducedquestions.predict(testreducedopennessquestions)
    
    conscientiousnessclf_logistic_reducedquestions = linear_model.LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=20, warm_start=True, intercept_scaling=10000.)
    conscientiousnessclf_logistic_reducedquestions.fit(reducedopennessquestions, conscientiousnesscorrectlabels)
    conscientiousnesslogisticpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(reducedopennessquestions)
    conscientiousnesslogistictestpredictions_reducedquestions = conscientiousnessclf_logistic_reducedquestions.predict(testreducedopennessquestions)
    
    conscientiousnessclf_decisiontree_reducedquestions = DecisionTreeClassifier(max_depth=20)
    conscientiousnessclf_decisiontree_reducedquestions.fit(reducedopennessquestions, conscientiousnesscorrectlabels)
    conscientiousnessdecisiontreepredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(reducedopennessquestions)
    conscientiousnessdecisiontreetestpredictions_reducedquestions = conscientiousnessclf_decisiontree_reducedquestions.predict(testreducedopennessquestions)

    # Evaluation
    conscientiousness_perceptron_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions)
    conscientiousness_perceptron_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy))
    print("Able to predict conscientiousness based on openness training questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness training reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_perceptron_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions)
    conscientiousness_perceptron_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, testconscientiousnessquestionpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy))
    print("Able to predict conscientiousness based on openness testing questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness testing reduced questions using perceptron with %{} accuracy".format(conscientiousness_perceptron_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions)
    conscientiousness_sgd_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesssgdpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy))
    print("Able to predict conscientiousness based on openness training questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness training reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_sgd_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions)
    conscientiousness_sgd_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesssgdtestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy))
    print("Able to predict conscientiousness based on openness testing questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness testing reduced questions using stochastic gradient descent with %{} accuracy".format(conscientiousness_sgd_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions)
    conscientiousness_logistic_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnesslogisticpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy))
    print("Able to predict conscientiousness based on openness training questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness training reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_logistic_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions)
    conscientiousness_logistic_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnesslogistictestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy))
    print("Able to predict conscientiousness based on openness testing questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness testing reduced questions using logistic with %{} accuracy".format(conscientiousness_logistic_testing_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_training_accuracy = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions)
    conscientiousness_dt_training_accuracy_reducedquestions = metrics.accuracy_score(conscientiousnesscorrectlabels, conscientiousnessdecisiontreepredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy))
    print("Able to predict conscientiousness based on openness training questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness training reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_training_accuracy_reducedquestions), file=file_out)

    conscientiousness_dt_testing_accuracy = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions)
    conscientiousness_dt_testing_accuracy_reducedquestions = metrics.accuracy_score(testconscientiousnesscorrectlabels, conscientiousnessdecisiontreetestpredictions_reducedquestions)

    print("Able to predict conscientiousness based on openness testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy))
    print("Able to predict conscientiousness based on openness testing questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy), file=file_out)
    print("Able to predict conscientiousness based on openness testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions))
    print("Able to predict conscientiousness based on openness testing reduced questions using decision tree with %{} accuracy".format(conscientiousness_dt_testing_accuracy_reducedquestions), file=file_out)

    return

def main():

    #Shrinker() Used for shrinking to predetermined values used while testing
    dataset, trainingdataset, testdataset = ReadInData()
    #print("Dataset:\n", dataset)
    #print("\n\n\n")
    #print("Trainingdataset:\n", trainingdataset)
    #print("\n\n\n")
    #print("Testingdataset:\n", testdataset)
    #print("\n\n\n")
    datasettotals_training = CalculateIndividualTotals(trainingdataset) # Totals without transforming/normalizing
    datasettotals_testing = CalculateIndividualTotals(testdataset) # Totals without transforming/normalizing
    #print("datasettotals_training:\n", datasettotals_training)
    #print("datasettotals_testing:\n", datasettotals_testing)
    #print("\n\n\n")
    datasetpreferences_training = CalculateIndividualPreferences(datasettotals_training) # Preferences (labels) for each trait calculated
    datasetpreferences_testing = CalculateIndividualPreferences(datasettotals_testing) # Preferences (labels) for each trait calculated
    #print("datasetpreferences_training:\n", datasetpreferences_training)
    #print("datasetpreferences_testing:\n", datasetpreferences_testing)
    #print("\n\n\n")
    cleandataset_training = CleanDataForConsistency(trainingdataset)
    cleandataset_testing = CleanDataForConsistency(testdataset)
    #print("cleandataset_training:\n", cleandataset_training)
    #print("cleandataset_testing:\n", cleandataset_testing)
    #print("\n\n\n")
    cleandatasettotals_training = CalculateCleanedTotals(cleandataset_training) # Totals with transforming/normalizing
    cleandatasettotals_testing = CalculateCleanedTotals(cleandataset_testing) # Totals with transforming/normalizing
    #print("cleandatasettotals_training:\n", cleandatasettotals_training)
    #print("cleandatasettotals_testing:\n", cleandatasettotals_testing)
    #print("\n\n\n")
    cleaneddatasetpreferences_training = CalculateCleanedPreferences(cleandatasettotals_training)
    cleaneddatasetpreferences_testing = CalculateCleanedPreferences(cleandatasettotals_testing)
    #print("cleaneddatasetpreferences_training:\n", cleaneddatasetpreferences_training)
    #print("cleaneddatasetpreferences_testing:\n", cleaneddatasetpreferences_testing)
    #print("\n\n\n")
    normalizeddataset_training = NormalizeData(cleandataset_training)
    normalizeddataset_testing = NormalizeData(cleandataset_testing)
    #print("normalizeddataset_training:\n", normalizeddataset_training)
    #print("normalizeddataset_testing:\n", normalizeddataset_testing)
    #print("\n\n\n")
    normalizeddatasettotals_training = NormalizeData(cleandatasettotals_training)
    normalizeddatasettotals_testing = NormalizeData(cleandatasettotals_testing)
    #print("normalizeddatasettotals_training:\n", normalizeddatasettotals_training)
    #print("normalizeddatasettotals_testing:\n", normalizeddatasettotals_testing)
    #print("\n\n\n")

    print("Done cleaning: transforming for consistency, normalizing, calculating totals, and calculating preferences...")

    datasettotals = CalculateIndividualTotals(dataset)
    cleandataset = CleanDataForConsistency(dataset)
    cleandatasettotals = CalculateCleanedTotals(cleandataset)


    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    file_out = open("PredictExtroversiondataoutput.txt", "w")
    PredictExtroversion(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictNeuroticismdataoutput.txt", "w")
    PredictNeuroticism(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictAgreeablenessdataoutput.txt", "w")
    PredictAgreeableness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictConscientiousnessdataoutput.txt", "w")
    PredictConscientiousness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictOpennessdataoutput.txt", "w")
    PredictOpenness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()

    file_out = open("PredictBasedOnExtroversiondataoutput.txt", "w")
    PredictBasedOnExtroversion(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnNeuroticismdataoutput.txt", "w")
    PredictBasedOnNeuroticism(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnAgreeablenessdataoutput.txt", "w")
    PredictBasedOnAgreeableness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnConscientiousnessdataoutput.txt", "w")
    PredictBasedOnConscientiousness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnOpennessdataoutput.txt", "w")
    PredictBasedOnOpenness(file_out, trainingdataset, datasettotals_training, datasetpreferences_training, testdataset, datasettotals_testing, datasetpreferences_testing)
    file_out.close()

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    file_out = open("PredictExtroversioncleandataoutput.txt", "w")
    PredictExtroversion(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictNeuroticismcleandataoutput.txt", "w")
    PredictNeuroticism(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictAgreeablenesscleandataoutput.txt", "w")
    PredictAgreeableness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictConscientiousnesscleandataoutput.txt", "w")
    PredictConscientiousness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictOpennesscleandataoutput.txt", "w")
    PredictOpenness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()

    file_out = open("PredictBasedOnExtroversioncleandataoutput.txt", "w")
    PredictBasedOnExtroversion(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnNeuroticismcleandataoutput.txt", "w")
    PredictBasedOnNeuroticism(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnAgreeablenesscleandataoutput.txt", "w")
    PredictBasedOnAgreeableness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnConscientiousnesscleandataoutput.txt", "w")
    PredictBasedOnConscientiousness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnOpennesscleandataoutput.txt", "w")
    PredictBasedOnOpenness(file_out, cleandataset_training, cleandatasettotals_training, cleaneddatasetpreferences_training, cleandataset_testing, cleandatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    file_out = open("PredictExtroversionnormalizeddataoutput.txt", "w")
    PredictExtroversion(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictNeuroticismnormalizeddataoutput.txt", "w")
    PredictNeuroticism(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictAgreeablenessnormalizeddataoutput.txt", "w")
    PredictAgreeableness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictConscientiousnessnormalizeddataoutput.txt", "w")
    PredictConscientiousness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictOpennessnormalizeddataoutput.txt", "w")
    PredictOpenness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()

    file_out = open("PredictBasedOnExtroversionnormalizeddataoutput.txt", "w")
    PredictBasedOnExtroversion(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnNeuroticismnormalizeddataoutput.txt", "w")
    PredictBasedOnNeuroticism(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnAgreeablenessnormalizeddataoutput.txt", "w")
    PredictBasedOnAgreeableness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnConscientiousnessnormalizeddataoutput.txt", "w")
    PredictBasedOnConscientiousness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()
    file_out = open("PredictBasedOnOpennessnormalizeddataoutput.txt", "w")
    PredictBasedOnOpenness(file_out, normalizeddataset_training, normalizeddatasettotals_training, cleaneddatasetpreferences_training, normalizeddataset_testing, normalizeddatasettotals_testing, cleaneddatasetpreferences_testing)
    file_out.close()

    return

main()
