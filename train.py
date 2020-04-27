# Airline Tweet Sentiment Analysis
# Model Training

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Instantiate Spark-on-K8s Cluster
spark = SparkSession\
    .builder\
    .appName("Airline Tweet Sentiment Analysis Model Training")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","2")\
    .config("spark.driver.memory","2g")\
    .config("spark.executor.instances","2")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://cnow-cdp-bucket/warehouse/")\
.getOrCreate()

# Validate Connectivity
# spark.sql("SHOW databases").show()

# Prepare data
data = pd.read_csv("util/data/sentiment140.csv",encoding = "ISO-8859-1",header=None,names=["target","id","date","flag","user","body"])
data.head()

mySchema = StructType([ StructField("target", IntegerType(), True)\
                       ,StructField("id", StringType(), True)\
                       ,StructField("date", StringType(), True)\
                       ,StructField("flag", StringType(), True)\
                       ,StructField("user", StringType(), True)\
                       ,StructField("body", StringType(), True)])

df = spark.createDataFrame(data,schema=mySchema)
df.show(5)

# Create training, validation, and test sets
(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed = 2000)

# Prepare TF-IDF + Logistic Regression Model
tokenizer = Tokenizer(inputCol="body", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

# Train Model
lr = LogisticRegression(maxIter=20)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)

# Evaluate Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

spark.stop()