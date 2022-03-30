#!/usr/bin/env python
# coding: utf-8

# In[7]:


import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import pandas as pd
xls = pd.ExcelFile("C:/Users/Dev/Desktop/Extracted Data (2803).xlsx")

data_bits = []

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
collection_name_list = (data['Collection Name'].tolist())[0:20]
collection_name_list


# In[8]:


# Import the necessary modules for tweet scraping and cleaning

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from datetime import datetime
import tensorflow as tf
import time
import tweepy
import json
import contractions
import nltk
import keras
import pandas as pd
import re
import pickle
import unicodedata



# Set up similar functions for cleanup of scraped tweets

def remove_URL(text):
    '''Removes any URL that appears in a tweet'''

    return re.sub(r"http\S+", ' ', text)


def remove_accented_chars(text):
    '''Removes any accented characters'''

    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore'
            ).decode('utf-8', 'ignore')


def expand_contractions(text):
    '''Expands contractions into their full form, like I'm into I am'''

    return contractions.fix(text)


def remove_hash_at(text):
    '''Removes any instances of @ or #'''

    return re.sub("#\w*|@\w*", ' ', text)


def remove_special_characters(text, remove_digits=True):
    '''Removes any special characters and numbers (optional)'''

    pattern = (r'[^a-zA-Z0-9\s]'
                if not remove_digits else r'[^a-zA-Z\s]')
    return re.sub(pattern, '', text)


def simple_lemmatize(text):
    '''Text lemmatization'''
    
    return ' '.join([WordNetLemmatizer().lemmatize(word) for word in
                    text.split()])


def remove_stopwords(text):
    '''Remove stopwords if needed (Warning: Takes a very long time)'''
    
    tokens = [token.strip() for token in
              WordPunctTokenizer().tokenize(text)]
    filtered_tokens = [token for token in tokens if token.lower()
                       not in stopwords.words('english')]
    return ' '.join(filtered_tokens)


def normalize_text(
    text,
    URL_remove=True,
    accented_chars=True,
    contractions_exp=True,
    text_tidy=True,
    hash_at_remove=True,
    special_characters=True,
    lemmatize_text=True,
    stopwords_remove=False,
    ):
    '''Combines all text cleanup steps into one large function'''

    normalized_text = ''

    if URL_remove:
        text = remove_URL(text)
    if accented_chars:
        text = remove_accented_chars(text)
    if contractions_exp:
        text = expand_contractions(text)
    if text_tidy:
        text = text.lower()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    if hash_at_remove:
        text = remove_hash_at(text)
    if special_characters:
        text = remove_special_characters(text)
    if lemmatize_text:
        text = simple_lemmatize(text)
        text = re.sub(' +', ' ', text)
    if stopwords_remove:
        text = remove_stopwords(text)

    normalized_text += text + ' '

    return normalized_text


# In[9]:


convert_verify = lambda x: 0 if x == False else 1

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[ ]:



date_scraped = datetime.today().strftime("%Y-%m-%d")
date_scraped_1d_ago = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")



for collection_name in collection_name_list:
    
    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{collection_name} since:{date_scraped_1d_ago} until:{date_scraped}').get_items()):
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
    
    df.to_pickle(f"C:/Users/Dev/Pickle/{collection_name}.pkl")
    


# In[22]:


all_tweets = []

for collection_name in collection_name_list:
    df = pd.read_pickle(f"C:/Users/Dev/Pickle/{collection_name}.pkl")
    all_tweets.append(df)

all_tweets[1].shape[0]


# In[4]:


df = pd.read_pickle(f"C:/Users/Dev/Pickle/Azuki.pkl")
df


# In[44]:


from nltk.tokenize import TweetTokenizer
from collections import Counter
import re


tt = TweetTokenizer()
account_names_list = []
for tweets in all_tweets:
    ats = []
    tweet_content = tweets.loc[:, "Text"].tolist()
    for text in tweet_content:
        regex = re.compile("@\w*")
        at = [ans for ans in tt.tokenize(text) if regex.search(ans)]
        ats.extend(at)
    print(Counter(ats).most_common(2))
    try: 
        account_name = Counter(ats).most_common(1)[0][0]
        count = Counter(ats).most_common(1)[0][1]
        count2 = Counter(ats).most_common(2)[1][1]
        print(count/tweets.shape[0])
        print(count2/tweets.shape[0])
        if (count/tweets.shape[0])/(count2/tweets.shape[0]) > 10 or count/tweets.shape[0] > 0.5:
            account_names_list.append(account_name)
        else:
            account_names_list.append("Unknown")
    except:
        account_names_list.append("Unknown")
        
account_names_list


# In[31]:


all_accounts = []

for account_name in account_names_list:
    

    account_list = []
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

all_accounts


# In[ ]:





# In[ ]:





# In[46]:





# In[ ]:





# In[ ]:





# In[ ]:




