#Con RDD clave valor se puede hacer muchas cosas, por ejemplo:
#groupByKey()
#sortByKey()
#keys() y values()
#se pueden hacer cosas tipo SQL
#Con k/v utilizar mapValues() y flatMapValues() si la transformación no
#va a afectar la clave...es mucho más eficiente. 

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("spark://spark-master:7077").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)

lines = sc.textFile("file:///course/datasets/fakefriends.csv")
#es una función que trae clave valor
# id,name,age, friends: 0,will,33,385
# age,friends
rdd = lines.map(parseLine)
#reduceByKey toma los valores para la misma clave y los suma
#(33,385)-->(33,(385,1))  , (33,2)-->(33,(2,1))
#(33,(387,2))
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
#(33,(387,2))-->(33,193.5)
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])


rows = averagesByAge.collect()
for row in rows:
    age,friends=row #es como decir age=row[0] y friends=row[1]
    print( "(age=%s,friends=%s)" % (
            age,friends 
        ))
