import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')

train.SalePrice.describe()

#plt.scatter(train['GarageArea'], train['SalePrice'], alpha=.75,color='b') #alpha helps to show overlapping data
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')




data = train.select_dtypes(include=[np.number]).interpolate().dropna()

new_data = np.array(data['GarageArea'])
new_data = sorted(new_data)

q1, q3 = np.percentile(new_data, [25,75])
iqr = q3 - q1
upper_boundary = q3 + 1.5 * iqr
lower_boundary = q1 - 1.5 *iqr


plt.scatter(upper_boundary, lower_boundary,color='red')
# plt.xlim(150,1000)
# plt.ylim(0,500000)
plt.show()
