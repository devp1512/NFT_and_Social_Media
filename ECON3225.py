#!/usr/bin/env python
# coding: utf-8

import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import pandas as pd
xls = pd.ExcelFile("C:/Users/Dev/Desktop/Extracted Data (2803).xlsx")

data_bits = []
start = 0
for i in range(1,11):
    if i%2 == 0:
        df = pd.read_excel(xls, f'Sheet{i}').set_index('Collection #')
        data_bits.append(df)
    else:
        df = pd.read_excel(xls, f'Sheet{i}').set_index('Collection #')
        df.drop(index=df.index[-1], axis=0, inplace=True)
        data_bits.append(df)

data = pd.concat(data_bits)
sample_size = 10000
collection_name_list = (data['Collection Name'].tolist())[start:start+500]

convert_verify = lambda x: 0 if x == False else 1

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

date_scraped = datetime.today().strftime("%Y-%m-%d")
date_scraped_1d_ago = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")



for collection_name in collection_name_list:
    
    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{collection_name} since:2022-03-27 until:2022-03-28').get_items()):
        if i>sample_size:
            break
        elif tweet.retweetedTweet == None and tweet.quotedTweet == None and sia.polarity_scores(tweet.content)['compound'] != 0:
            tweets_list.append([tweet.date, tweet.content, normalize_text(tweet.content), tweet.user.username, convert_verify(tweet.user.verified), 
                                tweet.user.followersCount, tweet.user.friendsCount, tweet.user.statusesCount, 
                                tweet.user.favouritesCount, tweet.user.listedCount, tweet.user.mediaCount, 
                                tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.quoteCount, sia.polarity_scores(normalize_text(tweet.content))['compound']])
            
    # Creating a dataframe from the tweets list above
    df = pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Cleaned Text', 'Username', 'User Verified', 'followersCount', 
                                                'friendsCount', 'statusesCount', 'favouritesCount', 'listedCount', 
                                                'mediaCount','Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Sentiment Score'])
    
    df.to_pickle(f"C:/Users/Dev/Pickle/{start}.{collection_name}.pkl")
    start += 1
    
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import pandas as pd
xls = pd.ExcelFile("New Extracted Data (2803).xlsx")

data_bits = []
start = 0
for i in range(1,11):
    if i%2 == 0:
        df = pd.read_excel(xls, f'Sheet{i}').set_index('Collection #')
        data_bits.append(df)
    else:
        df = pd.read_excel(xls, f'Sheet{i}').set_index('Collection #')
        df.drop(index=df.index[-1], axis=0, inplace=True)
        data_bits.append(df)

data = pd.concat(data_bits)

convert_verify = lambda x: 0 if x == False else 1

all_accounts = []

account_names_list = list(data["Twitter username"])

for account_name in account_names_list:
    
    if account_name != 'Unknown':
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{account_name}').get_items()):
            if i>0:
                break
            all_accounts.append([tweet.user.username, convert_verify(tweet.user.verified), tweet.user.followersCount, tweet.user.friendsCount, tweet.user.statusesCount, 
                                tweet.user.favouritesCount, tweet.user.listedCount, tweet.user.mediaCount])
    else:
        all_accounts.append(['Unknown', None, None, None, None, None, None, None])

all_accounts = pd.DataFrame(all_accounts, columns=['Username', 'User Verified', 'followersCount', 
                                                'friendsCount', 'statusesCount', 'favouritesCount', 'listedCount', 
                                                'mediaCount'])

all_accounts.to_pickle(f"C:/Users/Dev/ECON3225/collections.pkl")

collections = pd.read_pickle("collections.pkl")
collections = collections.iloc[: , 1:]
collections.index += 1
df_concat = pd.concat([data, collections], axis=1)

df_concat.to_csv("collections_data.csv")
df_concat.to_pickle("collections_data.pkl")

import os
import re
import pandas as pd
import numpy as np
all_tweets = []

numbers = []
for file in os.listdir("C:/Users/Dev/ECON3225/Individual Collection Tweets"):
    numbers.append(int(re.search(r'(\d+)', file).group()))
dictionary = dict(zip(range(0, 500), numbers))

file_list = [i for i in range(0,500)]
for k,v in dictionary.items():
    file_list[v] = os.listdir("C:/Users/Dev/ECON3225/Individual Collection Tweets")[k]

for collection_name in file_list:
    df = pd.read_pickle(f"C:/Users/Dev/ECON3225/Individual Collection Tweets/{collection_name}")
    all_tweets.append(df)

all_tweets_df = pd.concat(all_tweets)

def find_percentiles(data):
    p_20 = np.percentile(data, 20)
    p_40 = np.percentile(data, 40)
    p_60 = np.percentile(data, 60)
    p_80 = np.percentile(data, 80)
    return [p_20, p_40, p_60, p_80]


def update(column, data, type):
    if type == "percentile":
        percentile = find_percentiles(all_tweets_df[column])
        data[column] = np.where(data[column] <= percentile[0], -2, data[column])
        data[column] = np.where((data[column] > percentile[0]) & (data[column] <= percentile[1]), -1, data[column])
        data[column] = np.where((data[column] > percentile[1]) & (data[column] <= percentile[2]), 0, data[column])
        data[column] = np.where((data[column] > percentile[2]) & (data[column] <= percentile[3]), 1, data[column])
        data[column] = np.where(data[column] > percentile[3], 2, data[column])
        
        data["Sentiment Score"] = np.where(data[column] == -2, data["Sentiment Score"]*0.5, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == -1, data["Sentiment Score"]*0.75, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 0, data["Sentiment Score"]*1, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 1, data["Sentiment Score"]*1.25, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 2, data["Sentiment Score"]*1.5, data["Sentiment Score"])
        
    if type == "iqr":
        median = np.median(all_tweets_df[column])
        iqr = np.subtract(*np.percentile(all_tweets_df[column], [75, 25]))
        data[column] = np.where((data[column] <= median+iqr), -2, data[column])
        data[column] = np.where((data[column] > median+iqr) & (data[column] <= median+2*iqr), -1, data[column])
        data[column] = np.where((data[column] > median+2*iqr) & (data[column] <= median+4*iqr), 0, data[column])
        data[column] = np.where((data[column] > median+4*iqr) & (data[column] <= median+8*iqr), 1, data[column])
        data[column] = np.where((data[column] > median+8*iqr), 2, data[column])
        
        data["Sentiment Score"] = np.where(data[column] == -2, data["Sentiment Score"]*0.8, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == -1, data["Sentiment Score"]*0.9, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 0, data["Sentiment Score"]*1.1, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 1, data["Sentiment Score"]*1.4, data["Sentiment Score"])
        data["Sentiment Score"] = np.where(data[column] == 2, data["Sentiment Score"]*1.8, data["Sentiment Score"])
        
    return data

mean_sentiment = []

for tweets in all_tweets:
    
    tweets['Sentiment Score'] = np.where(tweets['User Verified']==1, tweets['Sentiment Score']*1.8, tweets['Sentiment Score'])
    
    update("followersCount", tweets, "percentile")
    update("friendsCount", tweets, "percentile")
    update("Reply Count", tweets, "iqr")
    update("Retweet Count", tweets, "iqr")
    update("Like Count", tweets, "iqr")
    update("Quote Count", tweets, "iqr")
    
    if len(tweets['Sentiment Score']) == 0:
        mean_sentiment.append(0)
    else:
        mean_sentiment.append(np.mean(tweets["Sentiment Score"]))
        
mean_sentiment = pd.Series(mean_sentiment, name = "Sentiment")
mean_sentiment.index += 1

dataset = pd.read_pickle("collections_data.pkl")
new_dataset = pd.concat([dataset, mean_sentiment], axis=1)
new_dataset.to_csv("final.csv")
new_dataset.to_pickle("final.pkl")

import pandas as pd
import numpy as np
df = pd.read_pickle("final.pkl")

df['24h %'] = df['24h %'].str.rstrip('%').astype('float') / 100.0
df['7d %'] = df['7d %'].str.rstrip('%').astype('float') / 100.0
df["Owners"] = df["Owners"].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype("float")
df["Items"] = df["Items"].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype("float")
df["Floor Price"] = df["Floor Price"].replace({'---': np.nan, '< 0.01': 0.005}).astype("float")
df["Volume"] = df["Volume"].astype("str").str.replace(',', '').astype("float")

df.to_csv("cleaned_final.csv")
df.to_pickle("cleaned_final.pkl")

import pandas as pd
df = pd.read_pickle("cleaned_final.pkl")

import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
from sklearn import metrics
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler



model_df = df[["Floor Price","Volume","Owners","Items","User Verified","followersCount","friendsCount","statusesCount","favouritesCount","listedCount","mediaCount","Sentiment"]].dropna()
x = model_df[["Floor Price","Owners","Items","User Verified","followersCount","friendsCount","statusesCount","favouritesCount","listedCount","mediaCount","Sentiment"]]
y = model_df["Volume"]
x_numerical = x.drop(["User Verified"], axis=1)
list_numerical = x_numerical.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
scaler = StandardScaler().fit(x_train[list_numerical]) 
x_train[list_numerical] = scaler.transform(x_train[list_numerical])
x_test[list_numerical] = scaler.transform(x_test[list_numerical])


from itertools import combinations
sample_list = ["Floor Price","Owners","Items","User Verified","followersCount","friendsCount",
               "statusesCount","favouritesCount","listedCount","mediaCount","Sentiment"]
list_combinations = []
for n in range(len(sample_list) + 1):
    list_combinations += list(combinations(sample_list, n))
    
import statsmodels.api as sm
x_train = sm.add_constant(x_train)
est = sm.OLS(y_train, x_train).fit()
est.summary()

mse = []
data = []

for regressors in list_combinations[1:]:
    regressors = list(regressors)
    model = LinearRegression()
    model.fit(x_train[regressors], y_train)
    y_pred = model.predict(x_test[regressors])
    mse.append(metrics.mean_squared_error(y_test, y_pred))
    data.append({"coef": list(zip(x_train[regressors], model.coef_)),
                 "r_sq": model.score(x_train[regressors],y_train)})
    
print(mse[mse.index(min(mse))])
print(data[mse.index(min(mse))])

best_alpha_lasso = []
for regressors in list_combinations[1:]:
    regressors = list(regressors)
    model = LassoCV(cv=5, random_state=100, max_iter=10000)
    model.fit(x_train[regressors], y_train)
    best_alpha_lasso.append(model.alpha_)
      
mse_lasso = []
data_lasso = []
for index, regressors in enumerate(list_combinations[1:]):
    regressors = list(regressors)
    model = Lasso(best_alpha_lasso[index], random_state=100)
    model.fit(x_train[regressors], y_train)
    y_pred = model.predict(x_test[regressors])
    mse_lasso.append(metrics.mean_squared_error(y_test, y_pred))
    data_lasso.append({"coef": list(zip(x_train[regressors], model.coef_)),
                       "intercept": model.intercept_,
                 "r_sq": model.score(x_train[regressors],y_train)})
    
print(mse_lasso[mse_lasso.index(min(mse_lasso))])
print(data_lasso[mse_lasso.index(min(mse_lasso))])

best_alpha_ridge = []
for regressors in list_combinations[1:]:
    regressors = list(regressors)
    model = RidgeCV(cv=5)
    model.fit(x_train[regressors], y_train)
    best_alpha_ridge.append(model.alpha_)
    
mse_ridge = []
data_ridge = []

for index, regressors in enumerate(list_combinations[1:]):
    regressors = list(regressors)
    model = Ridge(best_alpha_ridge[index], random_state=100)
    model.fit(x_train[regressors], y_train)
    y_pred = model.predict(x_test[regressors])
    mse_ridge.append(metrics.mean_squared_error(y_test, y_pred))
    data_ridge.append({"coef": list(zip(x_train[regressors], model.coef_)),
                       "intercept": model.intercept_,
                 "r_sq": model.score(x_train[regressors],y_train)})
    
print(mse_ridge[mse_ridge.index(min(mse_ridge))])
print(data_ridge[mse_ridge.index(min(mse_ridge))])
    
