import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

data=pd.read_csv("data/ev-data.csv",encoding='ISO-8859-1')

# Countplots

plt.figure(figsize=(20, 10))
plt.suptitle("Countplots")
plt.subplot(2, 3, 1)
plt.title("City counts")
sns.countplot(data['city'])

plt.subplot(2, 3, 2)
plt.title("Motor way counts")
sns.countplot(data['motor_way'])

plt.subplot(2, 3, 3)
plt.title("country_roads counts")
sns.countplot(data['country_roads'])

plt.subplot(2, 3, 4)
plt.title("driving_style counts")
sns.countplot(data['driving_style'])

plt.subplot(2, 3, 5)
plt.title("A/C counts")
sns.countplot(data['A/C'])

plt.subplot(2, 3, 6)
plt.title("park_heating counts")
sns.countplot(data['park_heating'])

plt.show()

# Quantity(kWh)

plt.figure(figsize=(20, 10))
plt.suptitle("quantity(kWh)")
plt.subplot(1, 2, 1)
plt.title("Distribution plot")
sns.distplot(data['quantity(kWh)'], kde = True)

plt.subplot(1, 2, 2)
plt.title("Box plot")
sns.boxplot(y=data['quantity(kWh)'])

plt.show()

# Distance(KM)

plt.figure(figsize=(20, 10))
plt.suptitle("Distance(KM)")
plt.subplot(1, 2, 1)
plt.title("Distribution plot")
sns.distplot(data['trip_distance(km)'], kde = True)

plt.subplot(1, 2, 2)
plt.title("Box plot")
sns.boxplot(y=data['trip_distance(km)'])

plt.show()

# Consumption(kWh/100km)

plt.figure(figsize=(20, 10))
plt.suptitle("consumption(kWh/100km)")
plt.subplot(1, 2, 1)
plt.title("Distribution plot")
sns.distplot(data['consumption(kWh/100km)'], kde = True)

plt.subplot(1, 2, 2)
plt.title("Box plot")
sns.boxplot(y=data['consumption(kWh/100km)'])

plt.show()

# Average Speed (km/h)

plt.figure(figsize=(20, 10))
plt.suptitle("avg_speed(km/h)")
plt.subplot(1, 2, 1)
plt.title("Distribution plot")
sns.distplot(data['avg_speed(km/h)'], kde = True)

plt.subplot(1, 2, 2)
plt.title("Box plot")
sns.boxplot(y=data['avg_speed(km/h)'])

plt.show()

# Average Speed

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.title("Average speeds vs city")
sns.violinplot(x = 'city', y = 'avg_speed(km/h)', data = data)

plt.subplot(2, 2, 2)
plt.title("Average speeds vs motor_way")
sns.violinplot(x = 'motor_way', y = 'avg_speed(km/h)', data = data)

plt.subplot(2, 2, 3)
plt.title("Average speeds vs country_roads")
sns.violinplot(x = 'country_roads', y = 'avg_speed(km/h)', data = data)

plt.subplot(2, 2, 4)
plt.title("Average speeds vs driving_style")
sns.violinplot(x = 'driving_style', y = 'avg_speed(km/h)', data = data)

plt.show()

# Average Speed vs Driving Range

plt.figure(figsize=(12, 5))
plt.title("average spee vs driving range")
sns.regplot(x = 'avg_speed(km/h)', y = 'trip_distance(km)', data = data)
plt.show()

# Car battery vs Driving Range

plt.figure(figsize=(20, 8))
plt.suptitle("car battery info. vs driving range")
plt.subplot(2, 2, 1)
plt.title("quantity(kWh) vs trip_distance(km)")
sns.scatterplot(x = 'quantity(kWh)', y = 'trip_distance(km)', data = data)

plt.subplot(2, 2, 2)
plt.title("consumption(kWh/100km) vs trip_distance(km)")
sns.scatterplot(x = 'consumption(kWh/100km)', y = 'trip_distance(km)', data = data)
plt.show()

# Consumption (kWh/100km) vs a/c and park heating

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.title("consumption(kWh/100km) vs a/c")
sns.boxplot(x = 'A/C', y = 'consumption(kWh/100km)', data = data)

plt.subplot(2, 2, 2)
plt.title("consumption(kWh/100km) vs park_heating")
sns.boxplot(x = 'park_heating', y = 'consumption(kWh/100km)', data = data)

# Correlition

data.corr()
sns.heatmap(data.corr())
plt.show()
