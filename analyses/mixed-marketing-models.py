# Import packages
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Read data
data_mm = pd.read_csv("data/Chapter 15/mmix_data.csv")

# Describe data
data_mm.describe()

data_mm["ln_quantity"] = np.log(data_mm["quantity"])
data_mm["ln_price"] = np.log(data_mm["price"])
data_mm["ln_digital_ad"] = np.log(data_mm["digital_ad"])
data_mm["ln_digital_search"] = np.log(data_mm["digital_search"])
data_mm["ln_print"] = np.log(data_mm["print"] + 1)
data_mm["ln_tv"] = np.log(data_mm["tv"])
data_mm["lln_quantity"] = data_mm["ln_quantity"].shift(1, fill_value = 0)
data_mm["weekdays"] = pd.to_datetime(data_mm["date"]).dt.day_name()
data_mm["ln_digital"] = np.log(data_mm["digital_ad"] + data_mm["digital_search"])

# Check correlations among variables
data_mm[["ln_quantity", "lln_quantity", "ln_price", "ln_digital_ad", 
"ln_digital_search", "ln_print", "ln_tv"]].corr()

# Estimate mixed marketing model using statsmodels formula api
model_mmix = smf.ols("ln_quantity ~ lln_quantity + ln_price + ln_digital + ln_print + ln_tv + weekdays", 
                     data = data_mm).fit()

# Make predictions
y_pred = np.exp(model_mmix.predict())
