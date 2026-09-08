# Import packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Set the seed for reporducibility
np.random.seed(1)

# Read data 
data_cluster = pd.read_csv("data/Chapter 3/retail_segmentation.csv")

# Create predictor data
pred_names = ["avg_order_size", "avg_order_freq", "crossbuy", "multichannel", "per_sale", 
               "tenure", "avg_mktg_cnt", "return_rate"]

data_x = data_cluster[pred_names]

# Instantiate KMeans model
model_k_6 = KMeans(n_clusters = 6)

model_k_6.fit(data_x)

# Add cluster ids back to data_cluster
data_cluster["cluster_id"] = model_k_6.labels_

# Look at cluster means
data_cluster.groupby("cluster_id")["avg_order_size"].mean()
data_cluster.groupby("cluster_id")["Cust_No"].count()
