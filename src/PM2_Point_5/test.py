import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import unicodecsv
import seaborn as sb
from scipy.interpolate import spline
df = pd.read_csv("C:\\KnownSec\\PM2.5\\PRSA_data_2010.1.1-2014.12.31.csv")
df.info()
print(df.loc[1050:1066,['No','year','month','day','hour','pm2.5','cbwd']])