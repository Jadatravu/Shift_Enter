import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql.types import IntegerType,BooleanType,FloatType,StringType
from pyspark.sql.functions import udf,array,struct,col
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import re, os, sys
from datetime import datetime


def convert_to_float(s1):
    s2=re.sub(',','',s1)
    return float(s2)

def convert_to_str(i1):
    return str(i1)

def time_check(time_hr_val):
    if time_hr_val <= 9 and time_hr_val >= 7:
        return 1
    elif time_hr_val <= 20 and time_hr_val >= 17:
        return 1
    else:
        return 0

def c_time(s1):
    g1=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',s1)
    if g1 :
       hr = int(g1.group(4))
       return time_check(hr)
    else:
       g2=re.match(r'^(.*)\/(.*)\/(.*)\ (.*):(.*):(.*)\ (.*)',s1)
       if g2 :
           if g2.group(7) == 'PM':
                hr = int(g2.group(4)) + 12
           else:
                hr = int(g2.group(4))
           return time_check(hr)
    return -1

def check_time(time_str):
    try:
        gl=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',time_str)
        time_hr_val = int(gl.group(4))
        if time_hr_val <= 9 and time_hr_val >= 7:
           return 1
        elif time_hr_val <= 20 and time_hr_val >= 17:
           return 1
        else:
           return 0
    except Exception  as e:
        return 0

def c_weekend(do):
    if do.weekday() == 6 or do.weekday() == 5:
       return 1
    else:
       return 0

def check_weekend(date_str):
    s1=date_str
    g1=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',s1)
    if g1 :
       datetime_str = g1.group(1)+str('/')+ g1.group(2)+str('/')+g1.group(3)+str(' 00:00:00')
       datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
       return c_weekend(datetime_object)
    else:
       g2=re.match(r'^(.*)\/(.*)\/(.*)\ (.*):(.*):(.*)\ (.*)',s1)
       if g2 :
           datetime_str = g2.group(1)+str('/')+ g2.group(2)+str('/')+g2.group(3)+str(' 00:00:00')
           datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
           return c_weekend(datetime_object)
    return 0

conf = (SparkConf())
sc = pyspark.SparkContext(conf=conf)
sc.setLogLevel("ERROR")

cur_dir = os.getcwd()
train_path = os.path.join(cur_dir,'df_train_all.csv')
evaluate_path = os.path.join(cur_dir,'df_evaluate_all.csv')

spark = SQLContext(sc)
df1 = spark.read.csv('file://'+train_path, header = True, inferSchema = True,encoding='utf8')
df1_e = spark.read.csv('file://'+evaluate_path, header = True, inferSchema = True,encoding='utf8')

df1.printSchema()

c_udf = udf(lambda a: convert_to_float(a), FloatType())
i_udf = udf(lambda a: convert_to_str(a), StringType())
check_time_udf = udf(lambda a: c_time(a), IntegerType())
check_weekend_udf = udf(lambda a: check_weekend(a), IntegerType())


df01 = df1.withColumn('weekend',check_weekend_udf(df1['lpep_pickup_datetime']))
df02 = df01.withColumn('peak_time',check_time_udf(df01['lpep_pickup_datetime']))

df01_e = df1_e.withColumn('weekend',check_weekend_udf(df1_e['lpep_pickup_datetime']))
df02_e = df01_e.withColumn('peak_time',check_time_udf(df01_e['lpep_pickup_datetime']))
df_train = df02
df_train.printSchema()

df_evaluate = df02_e
df_evaluate.printSchema()

#categoricalColumns =['store_and_fwd_flag']
categoricalColumns =[]
stages = [] 
for categoricalCol in categoricalColumns:
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  stages += [stringIndexer, encoder]

#encColumns = ["VendorID","trip_type","payment_type","PULocationID","DOLocationID","RatecodeID","weekend","peak_time"]
encColumns = ["trip_type","payment_type","RatecodeID","weekend","peak_time"]
encColumns = []
for eCol in encColumns:
  encoder = OneHotEncoder(inputCol=eCol, outputCol=eCol+"classVec")
  stages += [encoder]

numericCols = ["improvement_surcharge","i_tolls_amount","tip_amount","extra","mta_tax","trip_distance", "passenger_count"]
assemblerInputs = map(lambda c: c + "classVec", categoricalColumns) + map(lambda c: c + "classVec", encColumns) + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df_train)
dataset = pipelineModel.transform(df_train)

pipeline_e = Pipeline(stages=stages)
pipelineModel_e = pipeline_e.fit(df_evaluate)
dataset_e = pipelineModel_e.transform(df_evaluate)


gbt = GBTRegressor(featuresCol = 'features', labelCol = 'i_total_amount')

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [2, 5])\
    .addGrid(gbt.maxIter, [10, 25])\
    .build()
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
cvModel = cv.fit(dataset)

cv_result = cvModel.transform(dataset)
cv_result.show()

rmse = evaluator.evaluate(cv_result)
print("RMSE on our train set: %g" % rmse)
dataset.show()
dataset_e.show()

cv_result_e = cvModel.transform(dataset_e)
cv_result_e.show()
result = cv_result_e.select(['i_total_amount','prediction'])
result.show()

rmse = evaluator.evaluate(cv_result_e)
print("RMSE on our evaluate set: %g" % rmse)

bModel = cvModel.bestModel
print(bModel._java_obj.getMaxIter())
print(bModel._java_obj.getMaxDepth())

sc.stop()
