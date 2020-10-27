from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

lr_normal = linear_model.LinearRegression()
lr_lesso = linear_model.Lasso()
lr_ridge = linear_model.Ridge()
lr_elasticNet = linear_model.ElasticNet()

X = [[3.4], [1.8], [4.6], [2.3], [3.1], [5.5], [0.7], [3.0], [2.6], [4.3], [2.1], [1.1], [6.1],
[4.8], [3.8]]
y = [[26.2], [17.8], [31.3], [23.1], [27.5], [36.0], [14.1], [22.3], [19.6], [31.3],
[24.0], [17.3], [43.2], [36.4], [26.1]]

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted_normal = cross_val_predict(lr_normal, X, y, cv=10)
predicted_lesso = cross_val_predict(lr_lesso, X, y, cv=10)
predicted_ridge = cross_val_predict(lr_ridge, X, y, cv=10)
predicted_elasticNet = cross_val_predict(lr_elasticNet, X, y, cv=10)

r2_normal = r2_score(y,predicted_normal)
r2_lesso = r2_score(y,predicted_lesso)
r2_ridge = r2_score(y,predicted_ridge)
r2_elasticNet = r2_score(y,predicted_elasticNet)

print('R2_normal:{}'.format(r2_normal))
print('R2_lesso:{}'.format(r2_lesso))
print('R2_ridge:{}'.format(r2_ridge))
print('R2_elasticNet:{}'.format(r2_elasticNet))
