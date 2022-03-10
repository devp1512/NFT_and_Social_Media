#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd

options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")

driver = webdriver.Chrome(options=options)
driver.get("https://www.nftsocialranking.com/")
index = 1
dataset = []
while True:
    try:
        project_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[2]'
        meta_score_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[3]'
        follower_change_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[4]'
        no_of_activities_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[5]'
        no_of_mentions_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[6]'
        avg_like_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[7]'
        avg_reply_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[8]'
        avg_retweet_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[9]'
        avg_quote_heading = f'/html/body/div[1]/div[2]/div[4]/div/div/div/div/div/div[2]/table/tbody/tr[{index+1}]/td[10]'
        project_raw = driver.find_element(By.XPATH, project_heading)
        meta_score_raw = driver.find_element(By.XPATH, meta_score_heading)
        follower_change_raw = driver.find_element(By.XPATH, follower_change_heading)
        no_of_activities_raw = driver.find_element(By.XPATH, no_of_activities_heading)
        no_of_mentions_raw = driver.find_element(By.XPATH, no_of_mentions_heading)
        avg_like_raw = driver.find_element(By.XPATH, avg_like_heading)
        avg_reply_raw = driver.find_element(By.XPATH, avg_reply_heading)
        avg_retweet_raw = driver.find_element(By.XPATH, avg_retweet_heading)
        avg_quote_raw = driver.find_element(By.XPATH, avg_quote_heading)
        dataset += [project_raw.text, meta_score_raw.text, follower_change_raw.text, no_of_activities_raw.text, no_of_mentions_raw.text, 
              avg_like_raw.text, avg_reply_raw.text, avg_retweet_raw.text, avg_quote_raw.text]
        index += 1
    except:
        break        

driver.quit()
        
print(dataset)

