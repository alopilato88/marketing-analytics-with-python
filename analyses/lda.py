import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Read data
data_lda = pd.read_csv("data/Chapter 4/segmentation_result.csv")

# Create predictor and outcome data 
pred_names = ["married", "own_home", "household_size", "income", "age"]
outcome_name = ["segment"]

data_x = data_lda[pred_names]
data_y = data_lda[outcome_name].to_numpy().reshape(2000,)

# fit <- lda(segment ~ married + own_home + household_size + income + age, data = seg)
model_lda = LinearDiscriminantAnalysis()

model_lda.fit(data_x, data_y)

