from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PipelineExample").getOrCreate()
dataset = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///course/datasets/adult.csv")# Prepare training documents from a list of (id, text, label) tuples.

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
trainDF.cache().count() # Cache because accessing training data multiple times, el acceso es más rápido. 


categoricalCols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

# The following two lines are transformers. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]) 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 

# The label column ("income") is also a string value - it has two possible values, "<=50K" and ">50K". 
# Convert it to a numeric value using StringIndexer.
# es la variable a predecir
# another estimator :)
labelToIndex = StringIndexer(inputCol="income", outputCol="label")

# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
#is important to mention that spark models recieve a vector (unique column with values)
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)

# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)

selected = predDF.select("features", "label", "prediction")
for row in selected.collect():
    f,l,p=row
    print( 
        "(features=%s, label=%s) -->  prediction=%f" % (
            f, l, p
        )
    )