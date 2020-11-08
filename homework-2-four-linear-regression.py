import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataframe = pd.read_csv('fatigue_data.csv',index_col='Sl. No.')
dataframe.info
dataframe.head()

xx=dataframe.drop(dataframe.columns[17:20],axis=1)
#print (xx.shape)
#xx

X=dataframe.drop(dataframe.columns[16:20], axis=1)
y=dataframe['Fatigue']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

lr_model=linear_model.LinearRegression()
#lr_model=linear_model.Lasso()
#lr_model=linear_model.Ridge()
#lr_model=linear_model.ElasticNet()

lr_model.fit(X_train, y_train)

y_pred=lr_model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=math.sqrt(mse)
r2=r2_score(y_test,y_pred)
print('MSE:{}'.format(mse))
print('RMSE:{}'.format(rmse))
print('MAE:{}'.format(mae))
print('R2:{}'.format(r2))

plt.plot(y_test, y_pred, 'ko')
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.grid(True)
plt.xlim((100, 1500))
plt.ylim((100, 1500))
plt.plot([100,1500],[100,1500], color='blue', linewidth=3.0, linestyle='-')
my_x_ticks = np.arange(100, 1600, 200)
my_y_ticks = np.arange(100, 1600, 200)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.axis('scaled')
plt.show()

print(lr_model.coef_)
