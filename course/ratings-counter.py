from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)

#RDD
lines = sc.textFile("file:///course/datasets/ml-100k/u.data")
#192 242 3 888918
#toma el 3
ratings = lines.map(lambda x: x.split()[2])
#cuenta la canitdad por rating , ejemplo aparecen 2 3 y pone (3,2)
result = ratings.countByValue()
sortedResults = collections.OrderedDict(sorted(
    result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
