import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql.types import IntegerType,BooleanType,FloatType,StringType
from pyspark.sql.functions import udf,array,struct,col
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

#change this value to the path of the datset
data_set_path = '/data1/home/ec2-user/csv_files/2017_Green_Taxi_Trip_Data.csv'

import re
def get_jan(s1):
    g1=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',s1)
    if g1 and g1.group(1) == "01":
       return True
    else:
       g2=re.match(r'^(.*)\/(.*)\/(.*)\ (.*):(.*):(.*)\ (.*)',s1)
       if g2 and g2.group(1) == "01":
           return True
    return False
    

def get_feb(s1):
    g1=re.match(r'^(.*)\-(.*)\-(.*)\ (.*):(.*):(.*)',s1)
    if g1 and g1.group(1) == "02":
       return True
    else:
       g2=re.match(r'^(.*)\/(.*)\/(.*)\ (.*):(.*):(.*)\ (.*)',s1)
       if g2 and g2.group(1) == "02":
           return True
    return False

def convert_to_float(s1):
    s2=re.sub(',','',str(s1))
    return float(s2)

def convert_to_str(i1):
    return str(i1)

conf = (SparkConf())
sc = pyspark.SparkContext(conf=conf)

spark = SQLContext(sc)
df1 = spark.read.csv('file://'+dataset_path, header = True, inferSchema = True,encoding='utf8')
df1.show()
df1.printSchema()
feb_udf = udf(lambda a: get_feb(a), BooleanType())
jan_udf = udf(lambda a: get_jan(a), BooleanType())
c_udf = udf(lambda a: convert_to_float(a), FloatType())


df01 = df1.withColumn('i_total_amount',c_udf(df1['total_amount']))
df02 = df01.withColumn('i_fare_amount',c_udf(df01['fare_amount']))
df03 = df02.withColumn('i_tolls_amount',c_udf(df02['tolls_amount']))
df_train = df03.filter(jan_udf('lpep_pickup_datetime'))
df_evaluate = df03.filter(feb_udf('lpep_pickup_datetime'))

df1.printSchema()

df_train.show()
df_evaluate.show()
df_train.toPandas().to_csv('df_train_all.csv')
df_evaluate.toPandas().to_csv('df_evaluate_all.csv')


sc.stop()
