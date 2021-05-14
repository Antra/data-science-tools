import pyspark
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = pyspark.SparkContext('local[*]')

txt = sc.textFile(
    'file:///C:\Projects\data-science-tools\Spark\data\python_copyright.txt')
print(txt.count())

python_lines = txt.filter(lambda line: 'python' in line.lower())
print(python_lines.count())

with open('results.txt', 'w') as file_obj:
    file_obj.write(f'Number of lines: {txt.count()}\n')
    file_obj.write(f'Number of lines with python: {python_lines.count()}\n')
