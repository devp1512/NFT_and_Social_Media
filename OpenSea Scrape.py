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
driver.get("https://opensea.io/rankings")
index = 1

project_heading = f'//*[@id="main"]/div/div[2]/div/div[2]/div[9]/[@*]'
project_raw = driver.find_element(By.XPATH, project_heading)

while True:
    try:
        print("2")
        project_heading = f'//*[@id="main"]/div/div[2]/div/div[2]/div[9]/a'
        print("2")
        project_raw = driver.find_element(By.XPATH, project_heading)
        print("2")
        print(project_raw.text)
        print("2")
        index += 1
    except:
        break

driver.quit()
        


# In[ ]:


#import 1 collection
import requests
import json
lst = ["doodles-official", "cyberbrokers"]
final = []
for i in lst:
    url = f"https://api.opensea.io/api/v1/collection/{i}"
    response = requests.request("GET", url)
    x = json.loads(response.text)
    final += x['collection']['stats']


# In[ ]:


#import many collections
import requests
url = "https://api.opensea.io/api/v1/collections?offset=0&limit=1"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
print(response.text)

