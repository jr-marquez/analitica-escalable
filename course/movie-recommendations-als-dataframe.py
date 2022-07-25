from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS
import sys

spark = SparkSession.builder.appName("ALSExample").master("spark://spark-master:7077").config("spark.driver.memory", "1g").getOrCreate()
names = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///course/datasets/ml-25m/movies.csv")
names.createOrReplaceTempView("names")
ratings = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///course/datasets/ml-25m/ratings.csv")    
print("Training recommendation model...")

als = ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId") \
    .setRatingCol("rating")
model = als.fit(ratings)
# Manually construct a dataframe of the user ID's we want recs for
userId = int(sys.argv[1])
userSchema = StructType([StructField("userId", IntegerType(), True)])
users = spark.createDataFrame([[userId,]], userSchema)

recommendations = model.recommendForUserSubset(users, 10).collect()
print("Top 10 recommendations for user ID " + str(userId))

for userRecs in recommendations:
    myRecs = userRecs[1]  #userRecs is (userID, [Row(movieId, rating), Row(movieID, rating)...])
    for rec in myRecs: #my Recs is just the column of recs for the user
       title = rec[0] #For each rec in the list, extract the movie ID and rating
       rating = rec[1]
       movieName = spark.sql("select title from names where movieId ='"+str(title)+"'")
       print(movieName.collect()[0][0]+" , "+str(rating))

spark.stop()        

