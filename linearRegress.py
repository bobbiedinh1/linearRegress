import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values
X = X.reshape(-1,1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)
# X = X.reshape(X.size)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# plt.scatter(X_test, y_test, color = "red")
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Test Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# Assuming you have already defined X_train, y_train, X_test, y_test, and regressor

# Create a figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # Adjust figsize as needed

# Plot for Training Set
axes[0].scatter(X_train, y_train, color='red')
axes[0].plot(X_train, regressor.predict(X_train), color='blue')
axes[0].set_title('Salary vs Experience (Training Set)')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary')

# Plot for Test Set
axes[1].scatter(X_test, y_test, color='red')
axes[1].plot(X_test, regressor.predict(X_test), color='blue')
axes[1].set_title('Salary vs Experience (Test Set)')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Salary')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()