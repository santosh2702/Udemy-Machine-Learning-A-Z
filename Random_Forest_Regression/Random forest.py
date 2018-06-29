#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2].values


"""from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)"""

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""




#fitting Randomforest regression  to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,Y)

#predict a new result with linear regression


Y_pred=regressor.predict(6.5)


#visualising thre linear rehgression result
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth of Bluff(Random Forest Regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

