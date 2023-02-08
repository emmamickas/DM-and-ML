import csv
import math
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_matrix

allMovies = {}
movieSimilarities = [[]]
numUsers = 0
numMovies = 0
allUsers = []
userRatings = {}
userRecommendations = {}
toBeEstimated = {}

def TestingShrinker():
    open_file = open('ratings.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = ',')
    fileLines = []
    for line in file_in:
        fileLines.append(line)
    open_file.close()
    open_file = open('ratingstesting.csv', 'w', encoding='utf-8')
    file_out = csv.writer(open_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    i = 0
    for line in fileLines:
        if (i == 0):
            file_out.writerow(line)
        elif (int(line[1]) >= 1 and int(line[1]) <= 10):
            file_out.writerow(line)
        i += 1
    open_file.close()
    
    open_file = open('movies.csv', 'r', encoding='utf-8', newline='')
    file_in = csv.reader(open_file, delimiter = ',')
    fileLines = []
    for line in file_in:
        fileLines.append(line)
    open_file.close()
    open_file = open('moviestesting.csv', 'w', encoding='utf-8')
    file_out = csv.writer(open_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    i = 0
    for line in fileLines:
        if (i == 0):
            file_out.writerow(line)
        elif (int(line[0]) >= 1 and int(line[0]) <= 10):
            file_out.writerow(line)
        i += 1
    open_file.close()
    return


def ReadInItems():
    #open_file = open('moviestesting.csv', 'r', encoding='utf-8') # This line can be switched with the one below to use for testing, uncomment call to TestingShrinker in main to use file
    open_file = open('movies.csv', 'r', encoding='utf-8')
    file_in = csv.reader(open_file, delimiter = ',')

    global numMovies
    numMovies = 0
    numEntries = 0

    for line in file_in:
        if numEntries == 0:
            numEntries += 1
            continue
        CreateItemProfile(line)
        numEntries += 1
        numMovies += 1
    open_file.close()

    print("Done reading in movies...")

    #open_file = open('ratingstesting.csv', 'r', encoding='utf-8') # This line can be switched with the one below to use for testing, uncomment call to TestingShrinker in main to use file
    open_file = open('ratings.csv', 'r', encoding='utf-8')
    file_in = csv.reader(open_file, delimiter = ',')

    numEntries = 0

    for line in file_in:
        if numEntries == 0:
            numEntries += 1
            continue
        AddRatings(line) # Record the rating for the movie currently being read in
        numEntries += 1

    for movieId in allMovies.keys():
        if (len(allMovies[movieId]["Ratings"]) == 0): # Check if the movie has no ratings
            allMovies[movieId]["AverageRating"] = 0
        else:
            allMovies[movieId]["AverageRating"] = (allMovies[movieId]["RatingTotal"] / len(allMovies[movieId]["Ratings"])) # Calculate the average rating to be used in normalizing later
    open_file.close()

    print("Done making movie profiles...")

    global allUsers
    allUsers = sorted(allUsers, key = lambda x: int(x))

    userRatingMatrix = np.zeros((numMovies, (int(allUsers[-1]) + 1))) # Initialize matrix for holding user ratings
    normalizedUserRatingMatrix = np.zeros((numMovies, (int(allUsers[-1]) + 1))) # Initialize a matrix for holding normalized user ratings
    userRatingMatrix, normalizedUserRatingMatrix, userRatingMask = InsertRatings(userRatingMatrix, normalizedUserRatingMatrix) # Insert existing ratings into these matrices

    return userRatingMatrix, normalizedUserRatingMatrix, userRatingMask


def CreateItemProfile(line):
    #Initialize all information about a movie being read in
    global numMovies
    movieId = 0
    movieId = line[0]
    movieTitle = line[1]
    movieGenres = line[2].split('|')
    movie = {}
    movie["Id"] = movieId
    movie["EntryNumber"] = numMovies
    movie["Title"] = movieTitle
    movie["Genres"] = movieGenres
    movie["Ratings"] = {}
    movie["RatingTotal"] = 0
    movie["AverageRating"] = 0
    movie["Similarities"] = {}
    movie["DotProducts"] = {}
    allMovies[movieId] = movie
    return


def AddRatings(line):
    global allUsers
    movieProfile = {}
    userId = line[0]
    movieId = line[1]
    movieRating = float(line[2])
    movieRatingTimestamp = line[3]

    movieProfile = allMovies[movieId]
    movieRatings = movieProfile["Ratings"] # Get currently existing rating dictionary to add information to

    movieRatings[userId] = (movieRating, movieRatingTimestamp) # Add a new rating to the movie
    movieProfile["Ratings"] = movieRatings
    allMovies[movieId] = movieProfile
    movieProfile["RatingTotal"] += movieRating # Keep track of rating total for use of averaging later

    if userId not in allUsers: # Keep track of all users who have rated a movie
        allUsers.append(userId)
    if userId not in list(userRatings.keys()):
        userRatings[userId] = {}

    userRatings[userId][movieId] = movieRating # Keep track of rating based on user for scalability later
    allUsers = sorted(allUsers, key = lambda x: int(x))

    return


def InsertRatings(userRatingMatrix, normalizedUserRatingMatrix):

    userRatingMask = np.ones((numMovies, (int(allUsers[-1]) + 1))) # Initilize a mask for user ratings
    normalizedUserRatingMask = np.ones((numMovies, (int(allUsers[-1]) + 1))) # Intialize a mask for normalized user ratings

    for userId in userRatings: # Loop through users to find existing ratings
        for movieId in userRatings[userId]: # Loop through only existing ratings for scalability an insert those ratings into the matrices
            userRatingMatrix[(allMovies[movieId]["EntryNumber"]), (int(userId))] = float(userRatings[userId][movieId])
            userRatingMask[(allMovies[movieId]["EntryNumber"]), (int(userId))] = 0
            normalizedUserRatingMask[(allMovies[movieId]["EntryNumber"]), (int(userId))] = 0
    
    maskedUserRatings = ma.masked_array(userRatingMatrix, userRatingMask) # Mask the user rating matrix

    for movieId in allMovies:
        averageRating = np.ma.average(maskedUserRatings[(allMovies[movieId]["EntryNumber"])][:]) #Calculate the average rating for a movie
        normalizedUserRatingMatrix[(allMovies[movieId]["EntryNumber"])] = maskedUserRatings[(allMovies[movieId]["EntryNumber"])].__sub__(averageRating) # Subtract the average to normalize

    maskedNormalizedUserRatings = ma.masked_array(normalizedUserRatingMatrix, userRatingMask) # Mask the normalized ratings

    return maskedUserRatings, maskedNormalizedUserRatings, userRatingMask


def CalculateRatingSimilarityMetric(normalizedUserRatings):

    for movieId in allMovies:
        normalizedRatings = {}
        movieRatings = allMovies[movieId]["Ratings"]

        for userId in movieRatings.keys():
            normalizedRatings[userId] = (movieRatings[userId][0] - (allMovies[movieId]["AverageRating"]))
        
        allMovies[movieId]["NormalizedRatings"] = normalizedRatings # Keep track of normalized ratings in an easy to access place
        
    print("Done normalizing...")

    for movieId in allMovies:
        entryNumber = allMovies[movieId]["EntryNumber"]

        if (len(normalizedUserRatings[entryNumber, normalizedUserRatings[entryNumber].mask == False]) == 0):
            normalDenominator = 0
        else:
            normalDenominator = np.sum(np.square(normalizedUserRatings[entryNumber, normalizedUserRatings[entryNumber].mask == False]))

        allMovies[movieId]["NormalDenominator"] = math.sqrt(normalDenominator) # Keep track of normalized denominator in an easy to access place

    print("Done calculating normalized denominators...")

    for userId in userRatings:
        toBeEstimated[userId] = []

        for movieId1 in userRatings[userId]:

            for movieId2 in userRatings[userId]: #Loop through only items that have been rated by the same user for scalability

                if (movieId1 == movieId2):
                    allMovies[movieId1]["DotProducts"][movieId2] = 1
                    continue

                dotProduct = (allMovies[movieId1]["NormalizedRatings"][userId]) * (allMovies[movieId2]["NormalizedRatings"][userId]) # Calculate dot product and record for easy access later

                if (movieId2 in allMovies[movieId1]["DotProducts"]):
                    allMovies[movieId1]["DotProducts"][movieId2] += dotProduct
                else:
                    allMovies[movieId1]["DotProducts"][movieId2] = dotProduct

    print("Done calculating dot products...")

    movieSimilarities = ma.zeros((numMovies, numMovies)) # Initialize a matrix to store similarities
    movieSimilarityMask = np.ones((numMovies, numMovies)) # Intiialize a matrix to mask similarities

    for movieId1 in allMovies:
        entryNumber1 = allMovies[movieId1]["EntryNumber"]
        allMovies[movieId1]["Similarities"][movieId1] = 1

        for movieId2 in allMovies[movieId1]["DotProducts"]: # Loop through only movies that have a dot product value (share ratings by users) for scalability
            entryNumber2 = allMovies[movieId2]["EntryNumber"]

            if (movieId1 == movieId2): # Check for identical movie identities
                allMovies[movieId1]["Similarities"][movieId2] = 1
                movieSimilarities[entryNumber1, entryNumber2] = allMovies[movieId1]["Similarities"][movieId2]
                movieSimilarityMask[entryNumber1, entryNumber2] = 0
                continue

            if (allMovies[movieId1]["NormalDenominator"] != 0 and allMovies[movieId2]["NormalDenominator"] != 0):
                allMovies[movieId1]["Similarities"][movieId2] = allMovies[movieId1]["DotProducts"][movieId2] / ((allMovies[movieId1]["NormalDenominator"]) * (allMovies[movieId2]["NormalDenominator"])) # Calculate cosine similarity for the two movies
                movieSimilarityMask[entryNumber1, entryNumber2] = 0
                movieSimilarities[entryNumber1, entryNumber2] = allMovies[movieId1]["Similarities"][movieId2]


    maskedMovieSimilarities = ma.masked_array(movieSimilarities, movieSimilarityMask) # Mask similarity matrix

    print("Done calculating similarities...")

    return maskedMovieSimilarities


def CalculateNeighbors():

    movieNeighbors = np.zeros((numMovies, numMovies)) # Initialize a matrix to store similarities of neighbors of movies alone
    mask = np.ones((numMovies, numMovies)) # Intialize a mask for the neighbor matrix

    for movieId in allMovies:

        similarities = {}
        entryNumber = allMovies[movieId]["EntryNumber"]
        similarities = allMovies[movieId]["Similarities"]
        sortedNeighbors = {}

        if len(similarities) == 1: # If this movie only has one similarity, that similarity is itself, set similarity to '1' and continue on
            sortedNeighbors[movieId] = 1
            allMovies[movieId]["Neighbors"] = sortedNeighbors
            continue

        similarityValues = list(similarities.items())
        similarityValues = sorted(similarityValues, key = lambda movie: (-movie[1], movie[0])) # Sort all found similarity values
        i = 0

        for movieTuple in similarityValues: # Iterate to select only the top five similarities to include in neighborhood

            if (movieTuple[0] == movieId):
                continue

            sortedNeighbors[(movieTuple[0])] = movieTuple[1]
            neighborEntryNumber = allMovies[(movieTuple[0])]["EntryNumber"]
            movieNeighbors[entryNumber, neighborEntryNumber] = movieTuple[1]
            mask[entryNumber, neighborEntryNumber] = 0
            i += 1

            for userId in allMovies[(movieTuple[0])]["Ratings"]: # Find each user that has rated a neighbor
                if userId not in allMovies[movieId]["Ratings"].keys() and movieId not in toBeEstimated[userId]: # If a user hasn't rated this movie, but has rated that movies neighbor, store it to be estimated
                    toBeEstimated[userId].append(movieId)

            if (i >=5):
                break

        allMovies[movieId]["Neighbors"] = sortedNeighbors

    maskedMovieNeighbors = ma.masked_array(movieNeighbors, mask) # Mask the neighborhood array
    
    print("Done calculating neighbors...")

    return maskedMovieNeighbors


def EstimateRatings(movieNeighbors, userRatingMatrix, userRatingMask):

    allUserRatings = userRatingMatrix.copy() # Make a copy so as to not affect original user ratings being used in calculations
    allUserRatingsMask = userRatingMask # Make a copy of mask so as to not affect original user ratings being used in calculations

    for userId in toBeEstimated:

        for movieId in toBeEstimated[userId]:

            entryNumber = allMovies[movieId]["EntryNumber"]

            if (userRatingMatrix[entryNumber, int(userId)]) != np.nan: #User may have already rated this movie, set estimated rating to existing rating
                estimatedRating = userRatingMatrix[entryNumber, int(userId)]
            
            else:
                ratingDenominator = ma.masked_array(movieNeighbors[entryNumber], (userRatingMatrix[:,int(userId)]).mask) # Find rating denominator with movies that this user has rated
                #ratingDenominator = movieNeighbors[entryNumber].copy()

                if (np.sum(ratingDenominator) == 0 or ma.count(ratingDenominator) == 0): # If user has not rated any similar movies, estimated rating is 0
                    estimatedRating = 0

                else:
                    estimatedRating = np.dot(movieNeighbors[entryNumber], userRatingMatrix[:,int(userId)]) / np.sum(ratingDenominator) # User has rated movies in neighborhood, use dot product/similarity denominator to calculate rating

            allUserRatings[entryNumber, int(userId)] = estimatedRating # Record estimated rating
            allUserRatingsMask[entryNumber, int(userId)] = 0 # Update mask
            userRatings[userId][movieId] = estimatedRating # Record estimated rating
            
    #maskedAllUserRatings = ma.masked_array(allUserRatings, allUserRatingsMask) # Mask all user ratings
    
    print("Done estimating ratings...")

    return allUserRatings


def DisplayAllMovies():
    i = 0
    for movieId in allMovies:
        print("{}: {}\n".format(movieId, allMovies[movieId]["Neighbors"]))
        i += 1
    return


def OrderRecommended(allUserRatings):
    global allUsers
    allUsers = sorted(allUsers, key = lambda x: int(x)) # Make recommendations in order

    for userId in allUsers:
        rated = userRatings[userId]
        ratedTuples = list(rated.items())
        ratedTuples = sorted(ratedTuples, key = lambda movie: (-movie[1], movie[0])) # Sort first by rating, then by movie id number if a tie is found
        userRecommendations[userId] = ratedTuples
    return


def OutputRecommended():
    out_file = open("./output.txt", "w")
    allUsers.sort(key = lambda x: int(x))
    for userId in allUsers:
        print("User-{}".format(userId), end="", file=out_file)
        i = 0
        for recommended in userRecommendations[userId]:
            print(" movie-{}".format(recommended[0]), end="", file=out_file)
            i += 1
            if (i >= 5):
                break
        print("", file=out_file)
    out_file.close()
    print("Done outputing recommendations...")
    return


def main():
    #TestingShrinker()
    userRatingMatrix, normalizedUserRatingMatrix, userRatingMask = ReadInItems()
    movieSimilarities = CalculateRatingSimilarityMetric(normalizedUserRatingMatrix)
    movieNeighbors = CalculateNeighbors()
    allUserRatings = EstimateRatings(movieNeighbors, userRatingMatrix, userRatingMask)
    #DisplayAllMovies()
    #print(movieSimilarities)
    OrderRecommended(allUserRatings)
    OutputRecommended()
    return

main()
