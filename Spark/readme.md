# Spark
These are my notes from the [RealPython tutorial on PySpark](https://realpython.com/pyspark-intro/).

## Installation
PySpark can be installed via pypi: `python -m pip install pyspark`.  
Alternatively, it can be downloaded from [Apache](https://spark.apache.org/downloads.html).

There are some good [installation instructions](https://towardsdatascience.com/installing-apache-pyspark-on-windows-10-f5f0c506bea1) if I ever need to set it up from scratch.

## Requirements
Remember that PySpark requires Java to be installed, otherwise I get an error:
```
Java not found and JAVA_HOME environment variable is not set.
Install Java and set JAVA_HOME to point to the Java installation directory.
```

If running with virtual environments, it will give an error about not being able to find python3 if `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` environmental variables are not set:
```
Missing Python executable 'python3', defaulting to 'C:\Users\x\AppData\Local\Programs\Python\Python37\Lib\site-packages\pyspark\bin\..' for SPARK_HOME environment variable. Please install Python or specify the correct Python executable in PYSPARK_DRIVER_PYTHON or PYSPARK_PYTHON environment variable to detect SPARK_HOME safely.
```

One workaround is to specify it in the code before calling SparkContext:
```
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
```

On Windows, I may need to remember to disable the redirects in `Manage App Execution Aliases`.