
# coding: utf-8

# # Spark Machine Learning Tutorial
# 
# In order to get started with Spark and MLLib using iPython notebooks, it's necessary to use either the Databricks Spark MOOC Vagrant build or to install iPython Notebook at directed at the Hortonworks site.
# 
# 

# ### ** Part 1: Building an RDD and Running Statistics **

# #### 1(a) Load the data and parse it

# In[1]:

# Load Hubway bicycle ride sharing data
import numpy as np
np.set_printoptions(suppress=True)

import pyspark_csv as pycsv
sc.addPyFile('pyspark_csv.py')

def parseLine(x):
    return [int(y) for y in x.split(',')]

import os.path
baseDir = os.path.join('data')
fileName = os.path.join(baseDir, 'HubwayTrips.csv')

rawData = sc.textFile(fileName)

#Drop header row
#Duration,Morning,Afternoon,Evening,Weekday,Male,Age

header = sc.parallelize(rawData.take(1))
rawDataNoHeaders = rawData.subtract(header)

#Parse the lines
rideRDD = rawDataNoHeaders.map(parseLine).cache()
numRecords = rideRDD.count()
minDuration = rideRDD.map(lambda x : x[0]).min()
maxDuration = rideRDD.map(lambda x : x[0]).max()

print "Number of records : %d " % numRecords
print "Minimum duration : %d " % minDuration
print "Maximum duration : %d " % maxDuration



# #### 1(b) Use MLLib Statistics 

# In[132]:

from pyspark.mllib.stat import Statistics
summary = Statistics.colStats(rideRDD)
print "Duration\tMorning\tAfternoon\tEvening\tWeekday\tMale\tAge\n"
print("%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\n") % tuple(summary.mean())
print("%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\n") % tuple(summary.variance())
print("%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\n") % tuple(summary.numNonzeros())


# #### 1(c) Determine correlation of Age with Duration

# In[3]:

durationRDD = rideRDD.map(lambda x : x[0]) # Extract duration from the RDD
ageRDD = rideRDD.map(lambda x : x[6]) # Extract Age from the RDD
print(Statistics.corr(durationRDD, ageRDD, method="pearson")) # Print the Pearson correlation of Age vs. Duration


# ### ** Part 2: Linear Regression **

# #### ** (2a) Plotting **
# 
# 

# In[4]:

# Plot Age Vs. Duration
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.scatter(ageRDD.collect(), durationRDD.collect(),alpha=0.5)
plt.xlabel="Age"
plt.ylabel="Duration"
plt.tight_layout()
plt.show()


# In[5]:

# Plot Duration Histogram

plt.hist(durationRDD.collect(),bins=1000)
plt.xlabel="Duration"
plt.ylabel="Count"
plt.show()


# #### ** (2b) LabeledPoints **
# 
# Spark uses LabeledPoints for most of its machine learning methods.  It simply consists of an RDD of two-element vectors : the first is the label, the second is an array of features for that point. So, first we have to take our data and convert it to an RDD of LabeledPoints. Since we're predicting Duration, that will be our label.
# 
# A simple ParsePoint function that can be mapped over the rawDataRDD will suffice.

# In[6]:

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

def parsePoint(x):
    return LabeledPoint(x[0],x[1:]) #first field is label, all others are features

labeledRDD = rideRDD.map(parsePoint).cache() #use the parsepoint function to convert the RDD
print labeledRDD.take(5)


# #### ** (3b) Linear Regression**

# In[7]:


# Build the model
model = LinearRegressionWithSGD.train(labeledRDD)
print model
# Evaluate the model on training data
valuesAndPreds = labeledRDD.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))


# ### ** Part 3: Clustering **
# 
# #### Run the Hubway data through a k-means clustering algorithm to look for patterns

# In[8]:

from pyspark.mllib.clustering import KMeans, KMeansModel

# Build the model (cluster the data)
clusters = KMeans.train(rideRDD, 5, maxIterations=10,
        runs=10, initializationMode="random")

print "Duration\tMorning\tAfternoon\tEvening\tWeekday\tMale\tAge\n"
for center in clusters.clusterCenters:
    print ("%d8\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8d\n") % tuple(center)
#print clusters.clusterCenters


# ### ** Part 4: Regression Tree **
# Build a regression tree with the Hubway data to predict duration based on time of day, weekday, gender, and age.

# In[9]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = labeledRDD.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Categorical features have already been converted in this data set
model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print ('0:Morning,1:Afternoon,2:Evening,3:Weekday,4:Male,5:Age')
print('Learned regression tree model:')
print(model.toDebugString())


# ### ** Part 5: Classification  **
# Using PySpark_CSV, read in Titanic data and build classification models for it.
# 
# #### (5a) Logistic Regression

# In[67]:

from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.classification import LogisticRegressionWithSGD


    
sqlCtx = SQLContext(sc)
fileName = os.path.join(baseDir, 'titanic3.csv')
plaintext_rdd = sc.textFile(fileName)
titanicRawRDD = pycsv.csvToDataFrame(sqlCtx, plaintext_rdd).rdd

#remove blank rows
titanicRDD = titanicRawRDD.filter(lambda r : (r[2] != None) )

#pclass,survived,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home_dest

def parseRow(r):
    pclass = r[0]
    sex = 0 if r[3] == 'female' else 1
    age = r[4] if r[4] != None else -1 #flag invalid ages for filtering
    sibsp = r[5]
    parch = r[6]
    fare = r[8] if r[8] != None else -1 #flag missing fares for filtering
    try:
        lp =LabeledPoint(r[1], [pclass,sex,age,sibsp,parch,fare])
    except ValueError:
        lp = None
    return lp
    
#for row in titanicRDD.collect():
#     print row,parseRow(row),isnan(parseRow(row).features[2])
parsedTitanicRDD = titanicRDD.map(parseRow).filter(lambda lp : (lp.features[2] != -1) and (lp.features[5] != -1))
#print parsedTitanicRDD.collect()

pclasses = parsedTitanicRDD.map(lambda lp : lp.features[0]).distinct().collect()
sexes = parsedTitanicRDD.map(lambda lp : lp.features[1]).distinct().collect()
print pclasses
print sexes 

(trainingData, testData) = parsedTitanicRDD.randomSplit([0.7, 0.3])


# Train model
model = LogisticRegressionWithSGD.train(trainingData)

# evaluate the model on test data
results = testData.map(lambda p: (p.label, model.predict(p.features)))

# calculate the error
err = results.filter(lambda (v, p): v != p).count() / float(testData.count())

# Print results
print("Model Error = " + str(err))


# #### ** (5b) Classification Tree **

# In[69]:

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Categorical features have already been converted in this data set
#print trainingData.collect()
#print testData.collect()
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={0:len(pclasses)+1,1:len(sexes)},
                                    impurity='gini', maxDepth=5, maxBins=32, minInfoGain=0.001)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned classification tree model:')
print ('Class:Sex:Age:SibSp:ParCh:Fare')
print(model.toDebugString())


# ### ** Part 6: Recommenders  **
# Load the MovieLens data (1m records) and build a recommender with it.
# 
# Ratings format : UserID::MovieID::Rating::Timestamp
# 
# 
# Movies format : MovieID::Title::Genres
# 

# #### **(6a) Loading Data**

# In[113]:

def parseRatings(row):
    (userID,MovieID,Rating,Timestamp) = row.split("::")
    return (int(userID),int(MovieID),float(Rating))

def parseMovies(row):
    (MovieID,Title,Genres) = row.split("::")
    return (int(MovieID),Title,Genres)

ratingsRDD = sc.textFile( os.path.join(baseDir, 'ratings.dat')).map(parseRatings).cache()
print ratingsRDD.take(3)
moviesRDD = sc.textFile( os.path.join(baseDir, 'movies.dat')).map(parseMovies).cache()
print moviesRDD.take(3)
#moviesRDD = parseMovieFile('movies.dat').cache()


# #### ** (6b) Summary statistics and a more complex example **

# In[101]:

# Some summary stats on the movie data
numUsersRating = ratingsRDD.map(lambda r : r[0]).distinct().count()
numMoviesRated = ratingsRDD.map(lambda r : r[1]).distinct().count()
totalRatings = ratingsRDD.count()
distinctRatings = ratingsRDD.map(lambda r : r[2]).distinct().collect()
print "Total ratings %d, Total users %d, Total movies %d" % (totalRatings, numUsersRating, numMoviesRated)
print "Distinct Ratings : %s" % distinctRatings
mostRatedMovies = (ratingsRDD
                   .map(lambda r : (r[1],1))
                   .reduceByKey(lambda a,b : a+b)
                   .join(moviesRDD.map(lambda r : (r[0],r[1])))
                   .map(lambda (id,(numRatings, movieTitle)) : (id,movieTitle,numRatings))
                   .takeOrdered(25,key=lambda x : -x[2])                 
                  )
print mostRatedMovies


# #### ** (6c) Build the Recommender **

# In[126]:

import itertools
import math
from pyspark.mllib.recommendation import ALS

sc.setCheckpointDir('checkpoint/')
ALS.checkpointInterval = 2

#Create training and test sets (could also create a validation set if required)
(trainingData, testData) = ratingsRDD.randomSplit([0.7, 0.3])

model = ALS.train(trainingData, 8, 5, 0.1)
test_for_predict_RDD = testData.map(lambda x: (x[0], x[1]))
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = testData.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)
print rates_and_preds.take(10)


# #### ** (6d) Add personalized data **

# In[131]:

#Rate some movies to get personalized recommendations

myRatingsRDD = sc.parallelize ([
        [0,2858,3.0],
        [0,260,5.0],
        [0,1196,4.0],
        [0,480,4.0],
        [0,589,5.0],
        [0,1270,5.0],
        [0,1198,5.0],
        [0,1097,4.0],
        [0,858,2.0]
    ])

#Retrain the model with my preferences
#Exclude movies I've rated from the prediction set

newTrainingData = ratingsRDD.union(myRatingsRDD)
model = ALS.train(newTrainingData, 8, 5, 0.1)

#Generate (0,movieid) pairs for movies I haven't rated -- candidates for scoring
moviesIRated = myRatingsRDD.map(lambda row : row[1]).distinct().collect()
myUnratedMoviesRDD = (moviesRDD.filter(lambda x: x[0] not in moviesIRated).map(lambda x: (0, x[0])))

predictions = model.predictAll(myUnratedMoviesRDD).collect()
recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]

movies = moviesRDD.collect() #bring locally to print
print "Movies recommended for you:"
for i in xrange(len(recommendations)):
    print ("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')


# In[ ]:



