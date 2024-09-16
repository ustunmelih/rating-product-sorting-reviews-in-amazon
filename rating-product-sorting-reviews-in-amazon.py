
# Importing Libraries
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import sys
import numpy as np
import seaborn as sns
import warnings
import datetime as dt
import matplotlib.pyplot as plt


# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Data
df = pd.read_csv("Case/rating-product-sorting-reviews-in-amazon/amazon_review.csv")


# check the data    
def check_df(dataframe, head=5):
    print("SHAPE".center(50,'*'))
    print(dataframe.shape)
    print("INFO".center(50,"*"))
    print(dataframe.info())
    print("MISSING VALUES".center(50,'*'))
    print(dataframe.isnull().sum())
    print(" DUPLICATED VALUES ".center(70,'-'))
    print(dataframe.duplicated().sum())
    print(" UNIQUE VALUES ".center(70,'-'))
    print(dataframe.nunique())

check_df(df)

df.head()


# Data Preprocessing
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.drop(["asin", "unixReviewTime"], axis=1, inplace=True)
current_date = df["reviewTime"].max()
df["day_diff"] = (current_date - df["reviewTime"]).dt.days

#Time based Weighted Average
df["day_diff"].quantile([.25, .5, .75]) 
print("overall".center(50,'-'))
df["overall"].mean()
print("avg_280".center(50,'-'))
df.loc[(df["day_diff"] <= 280), "overall"].mean()
print("avg_280/430".center(50,'-'))
df.loc[(df["day_diff"] > 280) & (df["day_diff"] <= 430), "overall"].mean()
print("avg_430/600".center(50,'-'))
df.loc[(df["day_diff"] > 430) & (df["day_diff"] <= 600), "overall"].mean()
print("avg_600".center(50,'-'))
df.loc[df["day_diff"] > 600, "overall"].mean()

#Time based Weighted Average function
def time_based_weighted_average(data, w1=28, w2=26, w3=24, w4=22):
    avg_280 = data.loc[(data["day_diff"] <= 280), "overall"].mean() *  w1 / 100
    avg_280_430 = data.loc[(data["day_diff"] > 280) & (data["day_diff"] <= 430), "overall"].mean() * w2 / 100
    avg_430_600 =data.loc[(data["day_diff"] > 430) & (data["day_diff"] <= 600), "overall"].mean() * w3 / 100
    avg_600 = data.loc[data["day_diff"] > 600, "overall"].mean() * w4 / 100
    overall = avg_280 + avg_280_430 + avg_430_600 + avg_600
    return avg_280, avg_280_430, avg_430_600, avg_600, overall

time_based_weighted_average(df)

#Weighted Rating
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = score_up_down_diff(df["helpful_yes"], df["helpful_no"])



def score_average_rating(up, down):
    if (up + down).empty:
        return 0
    return up / (up + down)


df["score_average_rating"] = score_average_rating(df["helpful_yes"], df["helpful_no"])

#Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + \
                                                    z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)