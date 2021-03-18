import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("http://bit.ly/w-data")
data.head()

X=data[['Hours']]
Y=data[['Scores']]

plt.scatter(X,Y)
plt.xlabel('Hours')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.grid()
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, Y)
plt.plot(X, line)
plt.grid()
plt.show()

y_pred = regressor.predict(X_test) 
y_pred

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("R^2 Score :",score)

h1=[9.25]
s1=regressor.predict([h1])
print('Predicted score if a student studies for 9.25 hrs/ day:',s1[0][0])
