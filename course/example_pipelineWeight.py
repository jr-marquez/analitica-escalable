from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Imputer,VectorAssembler

from pyspark.sql import SparkSession

#spark = SparkSession.builder.appName("PipelineExample").master("spark://spark-master:7077").config("spark.driver.memory", "1g").getOrCreate()
spark = SparkSession.builder.appName("PipelineExample").getOrCreate()
file_type = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///course/datasets/heights_weights.csv")

# Configure an ML pipeline
# 2 Transformers
imputer = Imputer(inputCols=["Height","Weight"], outputCols=["out_Height", "out_Weight"])

#is important to mention that spark models recieve a vector (unique column with values)
assembler = VectorAssembler(inputCols=["out_Height", "out_Weight"],outputCol="features")

#1 Estimator
lr = LogisticRegression(featuresCol='features',labelCol="Male",maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[imputer, assembler, lr])

# Fit the pipeline to training .
model = pipeline.fit(file_type)
#model.save("/course/datasets/model")
# Prepare test values
test = spark.createDataFrame([
    (67,172),
    (80,200),
    (61,100)
], ["Height","Weight"])

# Make predictions on test h and w and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("Height", "Weight",  "prediction")
for row in selected.collect():
    h,w,p=row
    print( 
        "(Height=%d,Weight=%d) -->  prediction=%f" % (
            h, w, p   
        )
    )