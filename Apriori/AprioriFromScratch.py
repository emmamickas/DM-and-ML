allItems = {}
numEntries = 0

def APrioriPass1():
    file_in = open('browsing-data.txt', 'r')
    numEntries = 0
    fileLines = []
    for line in file_in:
        CalcInitialSupport(line)
        numEntries += 1
        items = line.split()
        fileLines.append(items)
    file_in.close()
    print("Done making items...")
    return numEntries, fileLines

def CalcInitialSupport(line):
    items = line.split() # Split line into separate items
    items.sort()
    for item in items:
        if item in allItems:
            allItems[item] += 1 # If item exists in allItems dictionary, increment by 1
        else:
            allItems[item] = 1 # If item does not yet exist in allItems dictionary, initialize at 1
    return

def APrioriFilter(items):
    frequentItems = {}
    for item in items.keys():
        if items[item] >= 100:
            frequentItems[item] = items[item] # If key (item) has count higher than 100, add to frequent item list
    return frequentItems

def APrioriCreatePairs(items, fileLines):
    candidatePairs = {}
    itemKeys = list(items)
    itemKeys.sort()
    # The following nested for loop initializes a dictionary containing all the possible pairs of frequent items, with the counts of those pairs being set to 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if itemKeys[i] != itemKeys[j] and (itemKeys[i], itemKeys[j]) not in candidatePairs:
                candidatePairs[(itemKeys[i], itemKeys[j])] = 0 # If this pair of items is not already in candidate pairs, initialize count of that pair to 0
    # The following nested for loop loops through every basket and counts the instances of all pairs from those baskets
    print("Done making pairs...")
    i = 0
    for line in fileLines:
        i += 1
        #if i % 500 == 0:
            #print("Finding pairs on line {}".format(i))
        basketItems = line # Read in from file, basket by basket
        #for (item1, item2) in candidatePairs:
        #    if item1 in basketItems and item2 in basketItems:
        #        candidatePairs[(item1, item2)] += 1 # If this pair is found in a basket and exists in the candidate pairs dictionary, increment its count by 1
        for j in range(len(basketItems)):
            for k in range(j + 1, len(basketItems)):
                if (basketItems[j], basketItems[k]) in candidatePairs:
                    candidatePairs[(basketItems[j], basketItems[k])]+= 1
                elif (basketItems[k], basketItems[j]) in candidatePairs:
                    candidatePairs[(basketItems[k], basketItems[j])]+= 1

    # The total count of every pair has been counted in the pairs dictionary
    return candidatePairs

def APrioriCreateTriples(pairs, fileLines):
    candidateTriples = {}
    pairKeys = list(pairs)
    pairKeys.sort()
    for i in range(len(pairKeys)):
        item1, item2 = pairKeys[i] # Grab two items that are in a frequent pair
        for j in range(i + 1, len(pairKeys)):
            if item1 == pairKeys[j][0]:
                item3 = pairKeys[j][1] # Find a third item that exists in a frequent pair with the first item
                if item3 != item2: # Check to make sure that the third item is not the second item
                    for j in range(i + 1, len(pairKeys)):
                        if item2 == pairKeys[j][0] and item3 == pairKeys[j][1]: # Check if the third item found exists in a frequent pair with the second item
                            if item3 != item1: # Check to make sure that the third item is not the first item
                                candidateTriples[(item1, item2, item3)] = 0 # All three items exist in frequent pairs with one another, found a candidate triple
                            else:
                                continue
                else:
                    continue

    lineCount = 0
    for line in fileLines:
        lineCount += 1
        #if lineCount % 500 == 0:
            #print("Finding triples on line {}".format(lineCount))
        basketItems = line # Read in from file, basket by basket
        #for (item1, item2, item3) in candidateTriples:
        #    if item1 in basketItems and item2 in basketItems and item3 in basketItems:
        #        candidateTriples[(item1, item2, item3)] += 1
        for i in range(len(basketItems)):
            for j in range(i + 1, len(basketItems)):
                for k in range(j + 1, len(basketItems)):
                    if (basketItems[i], basketItems[j], basketItems[k]) in candidateTriples:
                        candidateTriples[(basketItems[i], basketItems[j], basketItems[k])]+= 1
                    elif (basketItems[i], basketItems[k], basketItems[j]) in candidateTriples:
                        candidateTriples[(basketItems[i], basketItems[k], basketItems[j])]+= 1
                    elif (basketItems[j], basketItems[i], basketItems[k]) in candidateTriples:
                        candidateTriples[(basketItems[j], basketItems[i], basketItems[k])]+= 1
                    elif (basketItems[j], basketItems[k], basketItems[i]) in candidateTriples:
                        candidateTriples[(basketItems[j], basketItems[k], basketItems[i])]+= 1
                    elif (basketItems[k], basketItems[i], basketItems[j]) in candidateTriples:
                        candidateTriples[(basketItems[k], basketItems[i], basketItems[j])]+= 1
                    elif (basketItems[k], basketItems[j], basketItems[i]) in candidateTriples:
                        candidateTriples[(basketItems[k], basketItems[j], basketItems[i])]+= 1
                
    return candidateTriples


def displayCounts(items):
    i = 0
    for item in items.keys():
        print("{} ({})".format(item, items[item]))
        i+= 1
        if (i >= 5):
            break
    print("\n")
    return

def sortFrequencies(items):
    sortedItemList = []
    for item in items.keys():
        #print("Inserting: {} into {}".format(item, sortedItemList))
        for i in range(len(items)):
            if (i == len(sortedItemList)): # Reached the end of the list, no more comparisons, insert item
                sortedItemList.append(item)
                break
            elif (items[item] > items[(sortedItemList[i])]): # The next item's confidence is greater than the last, insert at space i
                sortedItemList.insert(i, item)
                break
            elif (items[item] == items[(sortedItemList[i])]): # The next item is the same as the last
                if item < (sortedItemList[i]): # The next item's element comes first lexicographically, insert it at space i
                    sortedItemList.insert(i, item)
                    break
    sortedItemDict = {}
    for item in sortedItemList:
        sortedItemDict[item] = items[item] # Insert items from sorted list into dictionary with frequencies

    return sortedItemDict

def breakTiesInPairs(pairs):
    sortedPairs = {}
    prevConfidence = 0.0
    topFivePairs = {}
    for pair in pairs:
        if (prevConfidence != pairs[pair] and len(topFivePairs) >= 5):
            break
        topFivePairs[(pair[0], pair[1])]= pairs[pair]
        prevConfidence = pairs[pair]
    pairKeys = list(topFivePairs)
    (curItem1, curItem2) = pairKeys[0]
    while True:
        for i in range(1, len(pairKeys)):
            if pairs[(curItem1, curItem2)] > pairs[(pairKeys[i])]: #If the current pair has confidence greater than the next, no other comparison needed
                continue
            elif pairs[(curItem1, curItem2)] < pairs[(pairKeys[i])]: #If the current pair has confidence less than the next, switch to the higher confidence pair, no other comparison needed
                (curItem1, curItem2) = pairKeys[i]
            else: #If the two pairs have the same confidence
                if (curItem1 < pairKeys[i][0]): #Compare the first elements lexicographically
                    continue
                elif (curItem1 > pairKeys[i][0]):
                    (curItem1, curItem2) = pairKeys[i]
                else: #If the first two elements are the same, compare the second elements lexicographically
                    if (curItem2 < pairKeys[i][1]):
                        continue
                    elif (curItem2 > pairKeys[i][1]):
                        (curItem1, curItem2) = pairKeys[i]
        sortedPairs[(curItem1, curItem2)] = pairs[(curItem1, curItem2)] #Record the pair with the highest confidence or smallest lexicographic ordering as the next pair
        pairKeys.remove((curItem1, curItem2))
        if (len(pairKeys) <= 0):
            break
        else:
            (curItem1, curItem2) = pairKeys[0]
    return sortedPairs

def breakTiesInTriples(triples):
    sortedTriples = {}
    prevConfidence = 0.0
    topFiveTriples = {}
    for triple in triples:
        if (prevConfidence != triples[triple] and len(topFiveTriples) >= 5):
            break
        topFiveTriples[(triple[0], triple[1], triple[2])]= triples[triple]
        prevConfidence = triples[triple]
    tripleKeys = list(topFiveTriples)
    (curItem1, curItem2, curItem3) = tripleKeys[0]
    while True:
        for i in range(1, len(tripleKeys)):
            if triples[(curItem1, curItem2, curItem3)] > triples[(tripleKeys[i])]: #If the current triple has confidence greater than the next, no other comparison needed
                continue
            elif triples[(curItem1, curItem2, curItem3)] < triples[(tripleKeys[i])]: #If the current triple has confidence less than the next, switch to the higher confidence triple, no other comparison needed
                (curItem1, curItem2, curItem3) = tripleKeys[i]
            else: #If the two triples have the same confidence
                if (curItem1 < tripleKeys[i][0]): #Compare the first elements lexicographically
                    continue
                elif (curItem1 > tripleKeys[i][0]):
                    (curItem1, curItem2, curItem3) = tripleKeys[i]
                else: #If the first two elements are the same, compare the second elements lexicographically
                    if (curItem2 < tripleKeys[i][1]):
                        continue
                    elif (curItem2 > tripleKeys[i][1]):
                        (curItem1, curItem2, curItem3) = tripleKeys[i]
                    else: #If the second two elements are the same, compare the third elements lexicographically
                        if (curItem3 < tripleKeys[i][2]):
                            continue
                        elif (curItem3 > tripleKeys[i][2]):
                            (curItem1, curItem2, curItem3) = tripleKeys[i]
        sortedTriples[(curItem1, curItem2, curItem3)] = triples[(curItem1, curItem2, curItem3)] #Record the triple with the highest confidence or smallest lexicographic ordering as the next triple
        tripleKeys.remove((curItem1, curItem2, curItem3))
        if (len(tripleKeys) <= 0):
            break
        else:
            (curItem1, curItem2, curItem3) = tripleKeys[0]
    return sortedTriples

def computePairConfidence(items, pairs):
    pairConfidences = {}
    for (item1, item2) in pairs:
        pairConfidences[(item1, item2)] = pairs[(item1, item2)] / items[item1] # Calculate chance of item1 and item2 in basket with item2
    for (item1, item2) in pairs:
        pairConfidences[(item2, item1)] = pairs[(item1, item2)] / items[item2] # Calculate chance of item1 and item2 in basket with item2
    return pairConfidences

def computeTripleConfidence(pairs, triples):
    tripleConfidences = {}
    for (item1, item2, item3) in triples:
        tripleConfidences[(item1, item2, item3)] = triples[(item1, item2, item3)] / pairs[(item1, item2)] # Calculate chance of item1, item2, and item1 in basket with item1 and item2
    for (item1, item2, item3) in triples:
        tripleConfidences[(item1, item3, item2)] = triples[(item1, item2, item3)] / pairs[(item1, item3)] # Calculate chance of item1, item2, and item1 in basket with item1 and item3
    for (item1, item2, item3) in triples:
        tripleConfidences[(item2, item3, item1)] = triples[(item1, item2, item3)] / pairs[(item2, item3)]  # Calculate chance of item1, item2, and item1 in basket with item2 and item3
    return tripleConfidences

def removeInfrequentFromItems(fileLines, items):
    newFileLines = []
    for line in fileLines:
        for i in range(len(line)):
            if (i >= len(line)):
                i = 0
            if line[i] not in items:
                item1 = line[i]
                line.remove(item1)
        newFileLines.append(line)
    return newFileLines

def displayLines(fileLines):
    for line in fileLines:
        if 'DAI93865' in line and 'FRO40251' not in line:
            print("Looking for FRO40251 in {}, not found".format(line))
    for line in fileLines:
        if 'DAI23334' in line and 'ELE92920' in line and 'DAI62779' not in line:
            print("Looking for DAI62779 in {}, not found".format(line))
        if 'DAI31081' in line and 'GRO85051' in line and 'FRO40251' not in line:
            print("Looking for FRO40251 in {}, not found".format(line))
        if 'DAI55911' in line and 'GRO85051' in line and 'FRO40251' not in line:
            print("Looking for FRO40251 in {}, not found".format(line))
        if 'DAI62779' in line and 'DAI88079' in line and 'FRO40251' not in line:
            print("Looking for FRO40251 in {}, not found".format(line))
        if 'DAI75645' in line and 'GRO85051' in line and 'FRO40251' not in line:
            print("Looking for FRO40251 in {}, not found".format(line))
    return

def displayPairConfidence(pairs):
    file_out = open('./output.txt', 'w')
    file_out.write("OUTPUT A\n")
    i = 0
    for pair in pairs:
        file_out.write("{} {} {}\n".format(pair[0], pair[1], pairs[pair]))
        i += 1
        if i >= 5:
            break
    file_out.close()
    return

def displayTripleConfidence(triples):
    file_out = open('./output.txt', 'a')
    file_out.write("OUTPUT B\n")
    i = 0
    for triple in triples:
        file_out.write("{} {} {} {}\n".format(triple[0], triple[1], triple[2], triples[triple]))
        i += 1
        if i >= 5:
            break
    file_out.close()
    return

def main():
    numEntries, fileLines = APrioriPass1()
    print("Total entries = {}\n".format(numEntries))
    frequentItems = APrioriFilter(allItems)
    frequentItems = sortFrequencies(frequentItems)
    #displayCounts(frequentItems)
    print("Done counting items...")
    fileLines = removeInfrequentFromItems(fileLines, frequentItems)
    candidatePairs = APrioriCreatePairs(frequentItems, fileLines)
    frequentPairs = APrioriFilter(candidatePairs)
    frequentPairs = sortFrequencies(frequentPairs)
    #displayCounts(frequentPairs)
    print("Done counting pairs...")
    candidateTriples = APrioriCreateTriples(frequentPairs, fileLines)
    frequentTriples = APrioriFilter(candidateTriples)
    frequentTriples = sortFrequencies(frequentTriples)
    print("Done counting triples...")
    #displayCounts(frequentTriples)
    pairConfidences = computePairConfidence(frequentItems, frequentPairs)
    pairConfidences = sortFrequencies(pairConfidences)
    #displayCounts(pairConfidences)
    #displayLines(fileLines)
    tripleConfidences = computeTripleConfidence(frequentPairs, frequentTriples)
    tripleConfidences = sortFrequencies(tripleConfidences)
    #displayCounts(tripleConfidences)
    topFivePairs = breakTiesInPairs(pairConfidences)
    topFiveTriples = breakTiesInTriples(tripleConfidences)
    displayPairConfidence(topFivePairs)
    displayTripleConfidence(topFiveTriples)
    return

main()
