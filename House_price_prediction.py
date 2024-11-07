import numpy as np
import pandas as pd
import csv
from sklearn. linear_model import LinearRegression
data = pd.read_csv("F:\Project\house_price_regression_dataset.csv")
X = data[['Square_Footage', 'Num_Bathrooms' ]] . values
y = data['House_Price' ] . values
model = LinearRegression()
model. fit(X, y)
new_house = np.array([ [int(input("Enter the Square_Footage:")), int(input("Enter the Num_Bathrooms:"))]]) # input from user
predicted_price = model.predict(new_house) # prediction of the user
print(f"Predicted price for the house: ${predicted_price[0] :,.2f}") # output
