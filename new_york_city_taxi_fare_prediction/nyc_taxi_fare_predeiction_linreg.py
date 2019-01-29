import sys
#sys.path.append('/home/ubuntu/packages/spark-2.2.0-bin-hadoop2.7/python/')
sys.path.append('/home/hadoop/spark/python/')
import pyspark
from pyspark import SparkConf
from pyspark.sql.types import IntegerType, StringType,FloatType
from pyspark.sql.functions import udf,array,struct,col
from pyspark.sql import SparkSession
import math
import re
import datetime
from numpy import array


def distance(origin, origin1, destination,destination1):
    lat1 = origin
    lon1 = origin1
    lat2 = destination
    lon2 = destination1
    #lat1 = float(lat1)
    #lat2 = float(lat2)
    #lon1 = float(lon1)
    #lon2 = float(lon2)
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

def func_distance(a,b,c,d):
     if (a is not None) and  (b is not None) and  (c is not None) and (d is not None):
        return distance(a,b,c,d)
     else:
        return 0
     #return distance((df_1.pickup_latitude, df_1.pickup_longitude),(df_1.dropoff_latitude,df_1.dropoff_longitude))



fun_dist_udf = udf(lambda a,b,c,d: func_distance(a,b,c,d), FloatType())

def check_time(time_str):
    try:
        gl=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',time_str)
        time_hr_val = int(gl.group(4))
        #print (time_hr_val)
        if time_hr_val <= 9 and time_hr_val >= 7:
           return 1
        elif time_hr_val <= 20 and time_hr_val >= 17:
           return 1
        else:
           return 0
    except Exception  as e:
        #print ("exceptipon occured")
        return 0

check_time_udf = udf(lambda a: check_time(a), IntegerType())

def check_weekend(date_str):
    #do=datetime.datetime.strptime(date_str,"%Y-%m-%d %H:%M:%S")
    do=date_str
    if do.weekday() == 6 or do.weekday() == 5:
       return 1
    else:
       return 0
check_weekend_udf = udf(lambda a: check_weekend(a), IntegerType())

slen = udf(lambda s: len(s), IntegerType())
def to_upper(s):
     if s is not None:
         return s.upper()

toupper_udf = udf(lambda s: to_upper(s), StringType())
def add_one(x):
     if x is not None:
         return x + 1
def add_hun(x):
     if x is not None:
         return x + 100
addone_udf = udf(lambda s: add_one(s), IntegerType())
addhun_udf = udf(lambda s: add_hun(s), FloatType())
conf = (SparkConf()
    .set("spark.driver.maxResultSize", "2g"))
conf.set('spark.executor.memory', '2100m')
conf.set('spark.executor.instances', '5')
conf.set('spark.worker.cleanup.enabled',True)
conf.set('spark.worker.cleanup.interval',30)
conf.set('spark.yarn.executor.memoryOverhead','400m')
conf.set('spark.executor.heartbeatInterval',10000000)
conf.set('spark.network.timeout', 100000000)
sc = pyspark.SparkContext(conf=conf)
from pyspark.sql import SQLContext

spark = SQLContext(sc)

df1 = spark.read.csv('hdfs://192.168.50.93:9000/user/hadoop/books1/train.csv', header = True, inferSchema = True)
#df1 = spark.read.csv('books1/train_temp.csv', header = True, inferSchema = True)
#df1 = spark.read.csv('hdfs://192.168.50.93:9000/books1/train.csv')
df1.printSchema()
print (dir(df1))

newDF1=df1.withColumn('Travel_Distance',fun_dist_udf(df1["pickup_latitude"],df1["pickup_longitude"],df1["dropoff_latitude"],df1["dropoff_longitude"]))
newDF1.printSchema()
newDF1.show(5)
newDF2 = newDF1.withColumn('Peak_Time',check_time_udf(df1['pickup_datetime']))
newDF2.printSchema()
newDF2.show(5)
#newDF3 = newDF2.withColumn('weekend',check_weekend_udf(df1['pickup_datetime']))
newDF3 = newDF2.withColumn('weekend',check_weekend_udf(df1['key']))
newDF3.printSchema()
newDF3.show(5)

trainX=newDF3.select('key','passenger_count','Travel_Distance','Peak_Time','weekend')
train_X=newDF3.select('pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','fare_amount','key','passenger_count','Travel_Distance','Peak_Time','weekend')
trainY=newDF3.select('fare_amount')

trainX.show(5)
trainY.show(5)


train_X4 = train_X.filter((col("passenger_count").isNotNull()) & (col("Travel_Distance").isNotNull()) & (col("Peak_Time").isNotNull()) & (col("weekend").isNotNull()))

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['passenger_count', 'Travel_Distance', 'Peak_Time', 'weekend'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(train_X4)
vhouse_df = vhouse_df.select(['features', 'fare_amount'])
vhouse_df.show(3)

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='fare_amount', maxIter=30, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(vhouse_df)
lr_model.write().overwrite().save("./spark-ml-30_derived-linearregression-model_6vm")
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
