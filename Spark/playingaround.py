from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession

spark = SparkSession.builer.appName('PracticeSession').getOrCreate()

#df_pyspark = spark.read.option('header', 'true').csv('test1.csv', inferSchema=True)
df_pyspark = spark.read.csv('test1.csv', inferSchema=True, header=True)
df_pyspark.printSchema()  # or df_pyspark.dtypes
df_pyspark.show()

df_pyspark.select(['Name', 'Experience']).show()
df_pyspark.describe().show()


# add column
df_pyspark.withColumn('Experience after 2 years', df_pyspark['Experience']+2)

# drop column
df_pyspark.drop('Experience after 2 years')

# rename column
df_pyspark.withColumnRenamed('Name', 'New Name')


# handling nans
df_pyspark.na.drop()  # drop any row with a na; use how, thresh and subset to control
df_pyspark.na.fill('Missing Value')

# imputing columns
imputer = Imputer(
    inputCols=['Age', 'Experience', 'Salary'],
    outputCols=["{}_imputed".format(c)
                for c in ['Age', 'Experience', 'Salary']]
).setStrategy('mean')

imputer.fit(df_pyspark).transform(df_pyspark).show()


# filtering
df_pyspark.filter('Salary<=20000').select(['Name', 'Age']).show()
