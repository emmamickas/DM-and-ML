import csv
import math
import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D

allCookies = {}
allTestCookies = {}
numCookies = 0
numTestCookies = 0
vocabulary = {}
vocabularyList = []
stopwords = []

allCharacters = {}
allTestCharacters = {}
vowels = ['a', 'e', 'i', 'o', 'u']
numCharacters = 0
numTestCharacters = 0

def ReadInItems():
    stopwords_file_in = open('stoplist.txt', 'r')

    for line in stopwords_file_in:
        words = line.split()
        for word in words:
            stopwords.append(word)
    
    stopwords_file_in.close()

    data_file_in = open('traindata.txt', 'r')
    label_file_in = open('trainlabels.txt', 'r')

    global numCookies
    numCookies = 0

    for line in data_file_in:
        CreateCookieProfile(line, numCookies)
        numCookies += 1

    numCookies = 0

    for line in label_file_in:
        CreateCookieLabels(line, numCookies)
        numCookies += 1


    data_file_in.close()
    label_file_in.close()


    data_file_in = open('testdata.txt', 'r')
    label_file_in = open('testlabels.txt', 'r')

    global numTestCookies
    numTestCookies = 0

    for line in data_file_in:
        CreateTestCookieProfile(line, numTestCookies)
        numTestCookies += 1

    numTestCookies = 0

    for line in label_file_in:
        CreateTestCookieLabels(line, numTestCookies)
        numTestCookies += 1


    data_file_in.close()
    label_file_in.close()

    print("Done reading in cookies...")

    vocabularyList.sort()

    numWords = 0

    for word in vocabularyList:
        vocabulary[word] = numWords
        numWords += 1

    print("Done setting up vocabulary...")

    

    data_file_in = open('ocr_train.txt', 'r')

    global numCharacters
    numCharacters = 0

    for line in data_file_in:
        if (line.isspace()):
            continue
        CreateCharacterProfile(line, numCharacters)
        numCharacters += 1

    numCharacters = 0

    data_file_in.close()

    data_file_in = open('ocr_test.txt', 'r')

    global numTestCharacters
    numTestCharacters = 0

    for line in data_file_in:
        if (line.isspace()):
            continue
        CreateTestCharacterProfile(line, numTestCharacters)
        numTestCharacters += 1

    data_file_in.close()

    print("Done reading in characters...")

    return

def CreateCookieProfile(line, numCookies):
    #Initialize all words for a cookie being read in
    allCookies[numCookies] = {}
    allCookies[numCookies]["CookieWords"] = []
    for word in line.split():
        if (word not in stopwords) and (word not in vocabularyList):
            vocabularyList.append(word)
        if (word not in stopwords):
            allCookies[numCookies]["CookieWords"].append(word)
    return

def CreateCharacterProfile(line, numCharacters):
    #Initialize all features for a character being read in
    factors = line.split()

    featureString = []

    featureString = factors[1]
    featureString = featureString[2:]

    allCharacters[numCharacters] = {}
    allCharacters[numCharacters]["CharacterFeatures"] = []

    for feature in featureString:
        allCharacters[numCharacters]["CharacterFeatures"].append(int(feature))

    if (factors[2] in vowels):
        allCharacters[numCharacters]["CorrectLabel"] = 1
    elif (factors[2] not in vowels):
        allCharacters[numCharacters]["CorrectLabel"] = -1

    allCharacters[numCharacters]["Character"] = factors[2]

    return

def CreateCookieLabels(line, numCookies):
    #Initialize correct label about a training cookie being read in
    if (int(line) == 1):
        allCookies[numCookies]["CorrectLabel"] = 1
    elif (int(line) == 0):
        allCookies[numCookies]["CorrectLabel"] = -1
    return

def CreateTestCookieProfile(line, numTestCookies):
    #Initialize all words for a cookie being read in
    allTestCookies[numTestCookies] = {}
    allTestCookies[numTestCookies]["CookieWords"] = []
    for word in line.split():
        if (word not in stopwords) and (word not in vocabularyList):
            vocabularyList.append(word)
        if (word not in stopwords):
            allTestCookies[numTestCookies]["CookieWords"].append(word)
    return

def CreateTestCharacterProfile(line, numTestCharacters):
    #Initialize all features for a character being read in
    factors = line.split()

    featureString = []

    featureString = factors[1]
    featureString = featureString[2:]

    allTestCharacters[numTestCharacters] = {}
    allTestCharacters[numTestCharacters]["CharacterFeatures"] = []

    for feature in featureString:
        allTestCharacters[numTestCharacters]["CharacterFeatures"].append(int(feature))

    if (factors[2] in vowels):
        allTestCharacters[numTestCharacters]["CorrectLabel"] = 1
    elif (factors[2] not in vowels):
        allTestCharacters[numTestCharacters]["CorrectLabel"] = -1

    allTestCharacters[numTestCharacters]["Character"] = factors[2]

    return

def CreateTestCookieLabels(line, numTestCookies):
    #Initialize correct label about a testing cookie being read in
    if (int(line) == 1):
        allTestCookies[numTestCookies]["CorrectLabel"] = 1
    elif (int(line) == 0):
        allTestCookies[numTestCookies]["CorrectLabel"] = -1
    return

def CookiePreprocessing():
    numCookies = len(allCookies)
    cookieFeatureMatrix = np.zeros((numCookies, len(vocabulary))) # Initialize matrix for holding cookie features
    cookieFeatureMask = np.ones((numCookies, len(vocabulary))) # Initilize a mask for cookie features

    numTestCookies = len(allTestCookies)
    testCookieFeatureMatrix = np.zeros((numTestCookies, len(vocabulary)))
    testCookieFeatureMask = np.ones((numTestCookies, len(vocabulary)))

    for cookie in allCookies:
        for word in allCookies[cookie]["CookieWords"]: # Set each word found in the cookie feature list to 1 in the matrix
            cookieFeatureMatrix[cookie][(vocabulary[word])] = 1
            cookieFeatureMask[cookie][(vocabulary[word])] = 0
    
    for cookie in allTestCookies:
        for word in allTestCookies[cookie]["CookieWords"]: # Set each word found in the cookie feature list to 1 in the matrix
            testCookieFeatureMatrix[cookie][vocabulary[word]] = 1
            testCookieFeatureMask[cookie][vocabulary[word]] = 0

    print("Done preprocessing cookies...")

    return cookieFeatureMatrix, testCookieFeatureMatrix

def CharacterPreprocessing(numFeatures):
    numCharacters = len(allCharacters)
    characterFeatureMatrix = np.zeros((numCharacters, numFeatures)) # Initialize matrix for holding character features
    characterFeatureMask = np.ones((numCharacters, numFeatures)) # Initilize a mask for character features

    numTestCharacters = len(allTestCharacters)
    testCharacterFeatureMatrix = np.zeros((numTestCharacters, numFeatures))
    testCharacterFeatureMask = np.ones((numTestCharacters, numFeatures))

    for character in allCharacters:
        for feature in range(len(allCharacters[character]["CharacterFeatures"])): # Set each feature found in the character feature vector
            characterFeatureMatrix[character][feature] = allCharacters[character]["CharacterFeatures"][feature]
            characterFeatureMask[character][feature] = 0

    for character in allTestCharacters:
        for feature in range(len(allTestCharacters[character]["CharacterFeatures"])): # Set each feature found in the character feature vector
            testCharacterFeatureMatrix[character][feature] = allTestCharacters[character]["CharacterFeatures"][feature]
            testCharacterFeatureMask[character][feature] = 0

    print("Done preprocessing characters...")

    return characterFeatureMatrix, testCharacterFeatureMatrix

def BinaryClassifier(LearningRate, TrainingExamples, TestingExamples, TrainingIterations, numFeatures, examplesDictionary, testingDictionary):

    weight = [0] * numFeatures
    weightSums = [0] * numFeatures

    mistakes = 0
    iterationMistakeTotals = [0] * TrainingIterations
    iterationAccuracies = []
    count = 0

    file_out = open('output.txt', 'a')

    for iteration in range(TrainingIterations):

        iterationAccuracies.append([0,0.0]) # Add a new element to keep track of iteration mistake count and iteration accuracy percent
        exampleNumber = 0 # Reset example number for the next iteration

        for example in TrainingExamples:

            if (examplesDictionary[exampleNumber]["CorrectLabel"]) * (np.dot(weight, example)) <= 0: # Mistake!
                mistakes += 1 # Increment total mistakes
                iterationMistakeTotals[iteration] += 1 # Increment mistakes for the current iteration
                weight = weight + np.multiply((LearningRate * examplesDictionary[exampleNumber]["CorrectLabel"]), example) # Update weight
                weightSums = np.add(weightSums, weight) # Update weight total to calculate average weight later
                #weightSums = weightSums + weight
                count += 1 # Keep track of total number of weights to calculate average weight later

            exampleNumber += 1

        iterationAccuracies[iteration][0] = PerceptronTesting(weight, TrainingExamples, examplesDictionary, len(examplesDictionary)) # Save this iteration's accuracy on training data
        iterationAccuracies[iteration][1] = PerceptronTesting(weight, TestingExamples, testingDictionary, len(testingDictionary)) # Save this iterations's accuracy on testing data

    for iteration in range(len(iterationMistakeTotals)): # Write iteration mistake counts to file
        print("iteration-{} {}".format(iteration + 1, iterationMistakeTotals[iteration]), file=file_out)

    print("", file=file_out)

    for iteration in range(len(iterationAccuracies)): # Write iteration accuracies to file
        print("iteration-{} {} {}".format(iteration + 1, iterationAccuracies[iteration][0], iterationAccuracies[iteration][1]), file=file_out)

    averageWeights = np.divide(weightSums, count)

    file_out.close()

    return weight, averageWeights

def PerceptronTesting(Weight, Examples, exampleDictionary, numExamples):
    mistakes = 0
    accuracy = 0.0
    exampleNumber = 0

    numExamples = len(exampleDictionary)

    # Test the prediction of each example based on the current weight
    for example in Examples:
        if (np.dot(Weight, example)) * (exampleDictionary[exampleNumber]["CorrectLabel"]) <= 0: # Mistake!
            mistakes += 1
        exampleNumber += 1

    accuracy = 1 - (mistakes/numExamples)

    return accuracy


def TestCharOutput(sampleMatrix, testMatrix):
    '''
    This function uses the scikit learn perceptron to test my answers. If you'd like to see it working on my data, uncomment the print statements.
    '''
    X = sampleMatrix
    X_test = testMatrix
    y = []
    for character in allCharacters:
        y.append(allCharacters[character]["CorrectLabel"])
    clf_percept = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept.fit(X, y, sample_weight=None)
    y_pred = clf_percept.predict(X)

    y_pred_test = clf_percept.predict(X_test)

    mistakes = 0
    i = 0
    for pred in y_pred:
        if allCharacters[i]["CorrectLabel"] * pred <= 0:
            mistakes += 1
        i += 1

    numCharacters = len(allCharacters)

    #print("Training Accuracy of characters (according to perceptron): {}".format(1 - (mistakes/numCharacters)))

    testMistakes = 0
    i = 0
    for pred in y_pred_test:
        if allTestCharacters[i]["CorrectLabel"] * pred <= 0:
            testMistakes += 1
        i += 1

    numTestCharacters = len(allTestCharacters)

    #print("Test Accuracy of characters (according to perceptron): {}".format(1 - (testMistakes/numTestCharacters)))
    return

def TestCookieOutput(sampleMatrix, testMatrix):
    '''
    This function uses the scikit learn perceptron to test my answers. If you'd like to see it working on my data, uncomment the print statements.
    '''
    X = sampleMatrix
    X_test = testMatrix
    y = []
    for cookie in allCookies:
        y.append(allCookies[cookie]["CorrectLabel"])
    clf_percept = Perceptron(max_iter=20, random_state=0, eta0=1)
    clf_percept.fit(X, y, sample_weight=None)
    y_pred = clf_percept.predict(X)

    y_pred_test = clf_percept.predict(X_test)

    mistakes = 0
    i = 0
    for pred in y_pred:
        if allCookies[i]["CorrectLabel"] * pred <= 0:
            mistakes += 1
        i += 1

    numCookies = len(allCookies)

    #print("Training Accuracy of cookies (according to perceptron): {}".format(1 - (mistakes/numCookies)))

    testMistakes = 0
    i = 0
    for pred in y_pred_test:
        if allTestCookies[i]["CorrectLabel"] * pred <= 0:
            testMistakes += 1
        i += 1

    numTestCookies = len(allTestCookies)

    #print("Test Accuracy of cookies (according to perceptron): {}".format(1 - (testMistakes/numTestCookies)))
    return



def main():
    ReadInItems()

    file_out = open('output.txt', 'w')
    file_out.close()

    cookieFeatureMatrix, testCookieFeatureMatrix = CookiePreprocessing()
    weight, averageWeight = BinaryClassifier(1, cookieFeatureMatrix, testCookieFeatureMatrix, 20, len(vocabulary), allCookies, allTestCookies)

    standardCookieTrainingAccuracy = PerceptronTesting(weight, cookieFeatureMatrix, allCookies, len(allCookies))
    averageCookieTrainingAccuracy = PerceptronTesting(averageWeight, cookieFeatureMatrix, allCookies, len(allCookies))
    standardCookieTestingAccuracy = PerceptronTesting(weight, testCookieFeatureMatrix, allTestCookies, len(allTestCookies))
    averageCookieTestingAccuracy = PerceptronTesting(averageWeight, testCookieFeatureMatrix, allTestCookies, len(allTestCookies))
    
    file_out = open('output.txt', 'a')
    print("\nFinal standard training accuracy of cookies (according to me): {}".format(standardCookieTrainingAccuracy))
    #print("Final average training accuracy of cookies (according to me): {}".format(averageCookieTrainingAccuracy))
    print("\n{} {}\n".format(standardCookieTrainingAccuracy, averageCookieTrainingAccuracy), file=file_out)
    print("Final standard testing accuracy of cookies (according to me): {}\n".format(standardCookieTestingAccuracy))
    #print("Final average testing accuracy of cookies (according to me): {}".format(averageCookieTestingAccuracy))
    print("{} {}\n\n".format(standardCookieTestingAccuracy, averageCookieTestingAccuracy), file=file_out)
    file_out.close()

    characterFeatureMatrix, testCharacterFeatureMatrix = CharacterPreprocessing(len(allCharacters[0]["CharacterFeatures"]))
    weight, averageWeight = BinaryClassifier(1, characterFeatureMatrix, testCharacterFeatureMatrix, 20, len(allCharacters[0]["CharacterFeatures"]), allCharacters, allTestCharacters)

    standardCharacterTrainingAccuracy = PerceptronTesting(weight, characterFeatureMatrix, allCharacters, len(allCharacters))
    averageCharacterTrainingAccuracy = PerceptronTesting(averageWeight, characterFeatureMatrix, allCharacters, len(allCharacters))
    standardCharacterTestingAccuracy = PerceptronTesting(weight, testCharacterFeatureMatrix, allTestCharacters, len(allTestCharacters))
    averageCharacterTestingAccuracy = PerceptronTesting(averageWeight, testCharacterFeatureMatrix, allTestCharacters, len(allTestCharacters))

    TestCookieOutput(cookieFeatureMatrix, testCookieFeatureMatrix)

    file_out = open('output.txt', 'a')
    print("\nFinal standard training accuracy of ocr (according to me): {}".format(standardCharacterTrainingAccuracy))
    #print("Final average training accuracy ocr (according to me): {}".format(averageCharacterTrainingAccuracy))
    print("\n{} {}\n".format(standardCharacterTrainingAccuracy, averageCharacterTrainingAccuracy), file=file_out)
    print("Final standard testing accuracy ocr (according to me): {}\n".format(standardCharacterTestingAccuracy))
    #print("Final average testing accuracy ocr (according to me): {}".format(averageCharacterTestingAccuracy))
    print("{} {}\n\n".format(standardCharacterTestingAccuracy, averageCharacterTestingAccuracy), file=file_out)
    file_out.close()

    TestCharOutput(characterFeatureMatrix, testCharacterFeatureMatrix)

    return

main()
