import pandas as pd
import numpy as np
data=pd.read_csv("data.csv",encoding='ISO-8859-1')
data.head()

data.isnull().sum()

data.shape
data.dropna(subset=['avg_speed(km/h)']).shape
data.dropna(subset=['avg_speed(km/h)'],inplace=True)
data.shape
data.columns

fixing outliers
data['avg_speed(km/h)'].describe()[6]

def count_outliers(data,column):
    count=0
    q1=data[column].describe()[4]
    q3=data[column].describe()[6]
    iqr=q3-q1
    for i in data[column]:
        if (i<q1-(1.5*iqr)) or (i>q3+(1.5*iqr)):
            count+=1
    return count
            
    
outlier_columns=[
       'trip_distance(km)', 'quantity(kWh)',
        'consumption(kWh/100km)',
       'avg_speed(km/h)', 'ecr_deviation']
for column in outlier_columns:
    print("No of outliers in {} are {}".format(column,count_outliers(data,column)))
    
def remove_outliers(data,column_list):
    for column in column_list:
        q1=data[column].describe()[4]
        q3=data[column].describe()[6]
        iqr=q3-q1
        for i in data[column]:
            if (i<q1-(1.5*iqr)) or (i>q3+(1.5*iqr)):
                data = data.loc[data[column] != i]
    return data
            
    
data.shape

data=remove_outliers(data,outlier_columns)
    
data.shape

for column in outlier_columns:
    print("No of outliers in {} are {}".format(column,count_outliers(data,column)))
    
data.to_csv("data_cleaned.csv",index=False)
