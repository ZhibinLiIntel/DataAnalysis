import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\KnownSec\\Bank\\bank-full.csv',sep=';')


sub_0 = df.job[df.y == 'no'].value_counts()
sub_1 = df.job[df.y == 'yes'].value_counts()
print("not sub: ")
print(sub_0)
print("subscribed: ")
print(sub_1)
df_job = pd.DataFrame({'subscribed':sub_1,'not sub':sub_0})
df_job.plot(kind='bar',stacked=True)
plt.show()


