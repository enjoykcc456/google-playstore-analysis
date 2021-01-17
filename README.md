# Google Play Store Apps Exploratory Data Analysis (EDA)

## Introduction
Google Play Store or formerly Android Market, is a digital distribution service developed and operated by Google. It is an official apps store that provides variety content such as apps, books, magazines, music, movies and television programs. It serves an as platform to allow users with 'Google certified' Android operating system devices to donwload applications developed and published on the platform either with a charge or free of cost. With the rapidly growth of Android devices and apps, it would be interesting to perform data analysis on the data to obtain valuable insights. 

The dataset that is going to be used is 'Google Play Store Apps' from Kaggle. It contains 10k of web scraped Play Store apps data for analysing the Android market. The tools that are going to be used for this EDA would be numpy, pandas, matplotlib and seaborn which I have learnt from [the course](http://zerotopandas.com). 

## Data Preparation and Cleaning

In this section, we will be loading the Google Store Apps data stored in csv using pandas which is a fast and powerful python library for data analysis and easy data manipulation in pandas DataFrame object. It is usually used for working with tabular data (e.g data in spreadsheet) in various formats such as CSV, Excel spreadsheets, HTML tables, JSON etc. We will then perform some data preparation and also cleaning on it.


```python
# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
# allow matplotlib to plot inline with frontends like Jupyter
```


```python
# look at the files in the dataset
import os
os.listdir('/kaggle/input/google-play-store-apps')
```




    ['license.txt', 'googleplaystore_user_reviews.csv', 'googleplaystore.csv']




```python
# load the apps and reviews data into pandas dataframe
apps_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
```


```python
# look at the first 10 records in the apps dataframe
apps_df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Photo Editor &amp; Candy Camera &amp; Grid &amp; ScrapBook</td>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>19M</td>
      <td>10,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>January 7, 2018</td>
      <td>1.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Coloring book moana</td>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>14M</td>
      <td>500,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Pretend Play</td>
      <td>January 15, 2018</td>
      <td>2.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U Launcher Lite – FREE Live Cool Themes, Hide ...</td>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>8.7M</td>
      <td>5,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>August 1, 2018</td>
      <td>1.2.4</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sketch - Draw &amp; Paint</td>
      <td>ART_AND_DESIGN</td>
      <td>4.5</td>
      <td>215644</td>
      <td>25M</td>
      <td>50,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Teen</td>
      <td>Art &amp; Design</td>
      <td>June 8, 2018</td>
      <td>Varies with device</td>
      <td>4.2 and up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pixel Draw - Number Art Coloring Book</td>
      <td>ART_AND_DESIGN</td>
      <td>4.3</td>
      <td>967</td>
      <td>2.8M</td>
      <td>100,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Creativity</td>
      <td>June 20, 2018</td>
      <td>1.1</td>
      <td>4.4 and up</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Paper flowers instructions</td>
      <td>ART_AND_DESIGN</td>
      <td>4.4</td>
      <td>167</td>
      <td>5.6M</td>
      <td>50,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>March 26, 2017</td>
      <td>1.0</td>
      <td>2.3 and up</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Smoke Effect Photo Maker - Smoke Editor</td>
      <td>ART_AND_DESIGN</td>
      <td>3.8</td>
      <td>178</td>
      <td>19M</td>
      <td>50,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>April 26, 2018</td>
      <td>1.1</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Infinite Painter</td>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>36815</td>
      <td>29M</td>
      <td>1,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>June 14, 2018</td>
      <td>6.1.61.1</td>
      <td>4.2 and up</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Garden Coloring Book</td>
      <td>ART_AND_DESIGN</td>
      <td>4.4</td>
      <td>13791</td>
      <td>33M</td>
      <td>1,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>September 20, 2017</td>
      <td>2.9.2</td>
      <td>3.0 and up</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kids Paint Free - Drawing Fun</td>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>121</td>
      <td>3.1M</td>
      <td>10,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Creativity</td>
      <td>July 3, 2018</td>
      <td>2.8</td>
      <td>4.0.3 and up</td>
    </tr>
  </tbody>
</table>
</div>




```python
# look at the random 10 records in the apps dataframe
apps_df.sample(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9971</th>
      <td>AÖF Ev İdaresi 1. Sınıf</td>
      <td>FAMILY</td>
      <td>NaN</td>
      <td>2</td>
      <td>11M</td>
      <td>1,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Education</td>
      <td>July 15, 2018</td>
      <td>3.0</td>
      <td>4.1 and up</td>
    </tr>
    <tr>
      <th>10404</th>
      <td>Top Puzzles Volvo FH Trucks</td>
      <td>FAMILY</td>
      <td>NaN</td>
      <td>4</td>
      <td>8.1M</td>
      <td>500+</td>
      <td>Free</td>
      <td>0</td>
      <td>Teen</td>
      <td>Puzzle</td>
      <td>November 20, 2016</td>
      <td>1.0</td>
      <td>4.0 and up</td>
    </tr>
    <tr>
      <th>3893</th>
      <td>4 in a Row</td>
      <td>GAME</td>
      <td>3.8</td>
      <td>4257</td>
      <td>Varies with device</td>
      <td>500,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Board</td>
      <td>May 13, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>乐屋网: Buying a house, selling a house, renting ...</td>
      <td>HOUSE_AND_HOME</td>
      <td>3.7</td>
      <td>2248</td>
      <td>15M</td>
      <td>100,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>House &amp; Home</td>
      <td>August 3, 2018</td>
      <td>v3.1.1</td>
      <td>4.0 and up</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>Fruits Bomb</td>
      <td>GAME</td>
      <td>4.4</td>
      <td>74695</td>
      <td>17M</td>
      <td>10,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Casual</td>
      <td>July 6, 2018</td>
      <td>3.3.3179</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>4973</th>
      <td>Ad Removal: thereisonlywe</td>
      <td>PRODUCTIVITY</td>
      <td>3.4</td>
      <td>16</td>
      <td>293k</td>
      <td>100+</td>
      <td>Paid</td>
      <td>$6.49</td>
      <td>Everyone</td>
      <td>Productivity</td>
      <td>May 3, 2014</td>
      <td>1.0</td>
      <td>2.2 and up</td>
    </tr>
    <tr>
      <th>2485</th>
      <td>OMD Protocols</td>
      <td>MEDICAL</td>
      <td>NaN</td>
      <td>0</td>
      <td>Varies with device</td>
      <td>10+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Medical</td>
      <td>July 27, 2018</td>
      <td>1.0</td>
      <td>Varies with device</td>
    </tr>
    <tr>
      <th>7507</th>
      <td>CL Pro Client for Craigslist</td>
      <td>SHOPPING</td>
      <td>3.6</td>
      <td>48</td>
      <td>2.2M</td>
      <td>5,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Shopping</td>
      <td>September 10, 2016</td>
      <td>Bowser4Craigslist</td>
      <td>4.0 and up</td>
    </tr>
    <tr>
      <th>8855</th>
      <td>DT Fieldlink</td>
      <td>BUSINESS</td>
      <td>3.0</td>
      <td>2</td>
      <td>49M</td>
      <td>500+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Business</td>
      <td>September 19, 2017</td>
      <td>2.1.1</td>
      <td>4.1 and up</td>
    </tr>
    <tr>
      <th>3950</th>
      <td>B</td>
      <td>FINANCE</td>
      <td>3.7</td>
      <td>800</td>
      <td>32M</td>
      <td>50,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>June 21, 2018</td>
      <td>2.16.5426</td>
      <td>4.4 and up</td>
    </tr>
  </tbody>
</table>
</div>



## Description of App Dataset columns
1. App : The name of the app
2. Category : The category of the app
3. Rating : The rating of the app in the Play Store
4. Reviews : The number of reviews of the app
5. Size : The size of the app
6. Install : The number of installs of the app
7. Type : The type of the app (Free/Paid)
8. The price of the app (0 if it is Free)
9. Content Rating :The appropiate target audience of the app
10. Genres: The genre of the app
11. Last Updated : The date when the app was last updated
12. Current Ver : The current version of the app
13. Android Ver : The minimum Android version required to run the app


```python
# type of Category
apps_df['Category'].unique()
```




    array(['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',
           'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',
           'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',
           'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',
           'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',
           'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',
           'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',
           'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION',
           '1.9'], dtype=object)




```python
# type of Type
apps_df['Type'].unique()
```




    array(['Free', 'Paid', nan, '0'], dtype=object)




```python
# type of Content Rating
apps_df['Content Rating'].unique()
```




    array(['Everyone', 'Teen', 'Everyone 10+', 'Mature 17+',
           'Adults only 18+', 'Unrated', nan], dtype=object)




```python
# type of Genres
apps_df['Genres'].unique()
```




    array(['Art & Design', 'Art & Design;Pretend Play',
           'Art & Design;Creativity', 'Art & Design;Action & Adventure',
           'Auto & Vehicles', 'Beauty', 'Books & Reference', 'Business',
           'Comics', 'Comics;Creativity', 'Communication', 'Dating',
           'Education;Education', 'Education', 'Education;Creativity',
           'Education;Music & Video', 'Education;Action & Adventure',
           'Education;Pretend Play', 'Education;Brain Games', 'Entertainment',
           'Entertainment;Music & Video', 'Entertainment;Brain Games',
           'Entertainment;Creativity', 'Events', 'Finance', 'Food & Drink',
           'Health & Fitness', 'House & Home', 'Libraries & Demo',
           'Lifestyle', 'Lifestyle;Pretend Play',
           'Adventure;Action & Adventure', 'Arcade', 'Casual', 'Card',
           'Casual;Pretend Play', 'Action', 'Strategy', 'Puzzle', 'Sports',
           'Music', 'Word', 'Racing', 'Casual;Creativity',
           'Casual;Action & Adventure', 'Simulation', 'Adventure', 'Board',
           'Trivia', 'Role Playing', 'Simulation;Education',
           'Action;Action & Adventure', 'Casual;Brain Games',
           'Simulation;Action & Adventure', 'Educational;Creativity',
           'Puzzle;Brain Games', 'Educational;Education', 'Card;Brain Games',
           'Educational;Brain Games', 'Educational;Pretend Play',
           'Entertainment;Education', 'Casual;Education',
           'Music;Music & Video', 'Racing;Action & Adventure',
           'Arcade;Pretend Play', 'Role Playing;Action & Adventure',
           'Simulation;Pretend Play', 'Puzzle;Creativity',
           'Sports;Action & Adventure', 'Educational;Action & Adventure',
           'Arcade;Action & Adventure', 'Entertainment;Action & Adventure',
           'Puzzle;Action & Adventure', 'Strategy;Action & Adventure',
           'Music & Audio;Music & Video', 'Health & Fitness;Education',
           'Adventure;Education', 'Board;Brain Games',
           'Board;Action & Adventure', 'Board;Pretend Play',
           'Casual;Music & Video', 'Role Playing;Pretend Play',
           'Entertainment;Pretend Play', 'Video Players & Editors;Creativity',
           'Card;Action & Adventure', 'Medical', 'Social', 'Shopping',
           'Photography', 'Travel & Local',
           'Travel & Local;Action & Adventure', 'Tools', 'Tools;Education',
           'Personalization', 'Productivity', 'Parenting',
           'Parenting;Music & Video', 'Parenting;Education',
           'Parenting;Brain Games', 'Weather', 'Video Players & Editors',
           'Video Players & Editors;Music & Video', 'News & Magazines',
           'Maps & Navigation', 'Health & Fitness;Action & Adventure',
           'Educational', 'Casino', 'Adventure;Brain Games',
           'Trivia;Education', 'Lifestyle;Education',
           'Books & Reference;Creativity', 'Books & Reference;Education',
           'Puzzle;Education', 'Role Playing;Education',
           'Role Playing;Brain Games', 'Strategy;Education',
           'Racing;Pretend Play', 'Communication;Creativity',
           'February 11, 2018', 'Strategy;Creativity'], dtype=object)




```python
# look at the info of the dataframe
apps_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10841 entries, 0 to 10840
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   App             10841 non-null  object 
     1   Category        10841 non-null  object 
     2   Rating          9367 non-null   float64
     3   Reviews         10841 non-null  object 
     4   Size            10841 non-null  object 
     5   Installs        10841 non-null  object 
     6   Type            10840 non-null  object 
     7   Price           10841 non-null  object 
     8   Content Rating  10840 non-null  object 
     9   Genres          10841 non-null  object 
     10  Last Updated    10841 non-null  object 
     11  Current Ver     10833 non-null  object 
     12  Android Ver     10838 non-null  object 
    dtypes: float64(1), object(12)
    memory usage: 1.1+ MB


By diagnosing the data frame, we know that:
1. There are 13 columns of properties with 10841 rows of data.
2. Column 'Reviews', 'Size', 'Installs' and 'Price' are in the type of 'object'
3. Values of column 'Size' are strings representing size in 'M' as Megabytes, 'k' as kilobytes and also 'Varies with devices'.
4. Values of column 'Installs' are strings representing install amount with symbols such as ',' and '+'.
5. Values of column 'Price' are strings representing price with symbol '$'.

Hence, we will need to do some data cleaning.

### Some Data Cleaning


```python
# 1) clean the 'Reviews' data and change the type 'object' to 'float'
reviews = [i for i in apps_df['Reviews']]

def clean_reviews(reviews_list):
    """
    As 'M' has been found the in reviews data, this function
    replace it with million
    """
    cleaned_data = []
    for review in reviews_list:
        if 'M' in review:
            review = review.replace('M', '')
            review = float(review) * 1000000  # 1M = 1,000,000
        cleaned_data.append(review)
    return cleaned_data

apps_df['Reviews'] = clean_reviews(reviews)
apps_df['Reviews'] = apps_df['Reviews'].astype(float)
```


```python
# 2) clean the 'Size' data and change the type 'object' to 'float'

# found value with '1,000+' in one of record, remove it from data_frame as uncertain whether it is 'M' or 'k'
index = apps_df[apps_df['Size'] == '1,000+'].index
apps_df.drop(axis=0, inplace=True, index=index)

sizes = [i for i in apps_df['Size']]

def clean_sizes(sizes_list):
    """
    As sizes are represented in 'M' and 'k', we remove 'M'
    and convert 'k'/kilobytes into megabytes
    """
    cleaned_data = []
    for size in sizes_list:
        if 'M' in size:
            size = size.replace('M', '')
            size = float(size)
        elif 'k' in size:
            size = size.replace('k', '')
            size = float(size)
            size = size/1024  # 1 megabyte = 1024 kilobytes
        # representing 'Varies with device' with value 0
        elif 'Varies with device' in size:
            size = float(0)
        cleaned_data.append(size)
    return cleaned_data

apps_df['Size'] = clean_sizes(sizes)
apps_df['Size'] = apps_df['Size'].astype(float)
```


```python
# 3) clean the 'Installs' data and change the type 'object' to 'float'
installs = [i for i in apps_df['Installs']]

def clean_installs(installs_list):
    cleaned_data = []
    for install in installs_list:
        if ',' in install:
            install = install.replace(',', '')
        if '+' in install:
            install = install.replace('+', '')
        install = int(install)
        cleaned_data.append(install)
    return cleaned_data
        
apps_df['Installs'] = clean_installs(installs)
apps_df['Installs'] = apps_df['Installs'].astype(float)
```


```python
# 4) clean the 'Price' data and change the type 'object' to 'float'
prices = [i for i in apps_df['Price']]

def clean_prices(prices_list):
    cleaned_data = []
    for price in prices_list:
        if '$' in price:
            price = price.replace('$', '')
        cleaned_data.append(price)
    return cleaned_data

apps_df['Price'] = clean_prices(prices)
apps_df['Price'] = apps_df['Price'].astype(float)
```


```python
# look at the random 10 records in the apps dataframe to verify the cleaned columns
apps_df.sample(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>358</th>
      <td>Firefox Focus: The privacy browser</td>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>36880.0</td>
      <td>4.0</td>
      <td>1000000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Communication</td>
      <td>July 6, 2018</td>
      <td>5.2</td>
      <td>5.0 and up</td>
    </tr>
    <tr>
      <th>10831</th>
      <td>payermonstationnement.fr</td>
      <td>MAPS_AND_NAVIGATION</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>9.8</td>
      <td>5000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Maps &amp; Navigation</td>
      <td>June 13, 2018</td>
      <td>2.0.148.0</td>
      <td>4.0 and up</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>Makeup Photo Editor: Makeup Camera &amp; Makeup Ed...</td>
      <td>PHOTOGRAPHY</td>
      <td>4.4</td>
      <td>10525.0</td>
      <td>25.0</td>
      <td>1000000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Photography</td>
      <td>July 27, 2018</td>
      <td>8.9.9</td>
      <td>4.0 and up</td>
    </tr>
    <tr>
      <th>5416</th>
      <td>PES 2018 PRO EVOLUTION SOCCER</td>
      <td>SPORTS</td>
      <td>4.4</td>
      <td>1721943.0</td>
      <td>26.0</td>
      <td>10000000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Sports</td>
      <td>June 27, 2018</td>
      <td>2.3.2</td>
      <td>5.0 and up</td>
    </tr>
    <tr>
      <th>6124</th>
      <td>Faketalk - Chatbot</td>
      <td>GAME</td>
      <td>4.2</td>
      <td>63056.0</td>
      <td>8.1</td>
      <td>1000000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Teen</td>
      <td>Word</td>
      <td>July 3, 2018</td>
      <td>1.9.7</td>
      <td>2.3 and up</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Fishing Brain &amp; Boating Maps Marine</td>
      <td>DATING</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>6.9</td>
      <td>500.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Dating</td>
      <td>July 23, 2018</td>
      <td>1.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>7208</th>
      <td>QUI EST CE ?</td>
      <td>FAMILY</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>5.4</td>
      <td>1000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Puzzle</td>
      <td>October 11, 2015</td>
      <td>1.0</td>
      <td>2.2 and up</td>
    </tr>
    <tr>
      <th>843</th>
      <td>ClassDojo</td>
      <td>EDUCATION</td>
      <td>4.4</td>
      <td>148549.0</td>
      <td>59.0</td>
      <td>10000000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Education;Education</td>
      <td>August 3, 2018</td>
      <td>4.21.1</td>
      <td>4.1 and up</td>
    </tr>
    <tr>
      <th>9244</th>
      <td>AP Stamps and Registration</td>
      <td>BOOKS_AND_REFERENCE</td>
      <td>3.4</td>
      <td>82.0</td>
      <td>2.7</td>
      <td>10000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Books &amp; Reference</td>
      <td>March 27, 2018</td>
      <td>2.0</td>
      <td>3.0 and up</td>
    </tr>
    <tr>
      <th>6101</th>
      <td>BF 4 Guns</td>
      <td>FAMILY</td>
      <td>4.0</td>
      <td>1542.0</td>
      <td>13.0</td>
      <td>50000.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>June 9, 2015</td>
      <td>3.0</td>
      <td>3.0 and up</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check on null values
apps_df.isna().sum()
```




    App                  0
    Category             0
    Rating            1474
    Reviews              0
    Size                 0
    Installs             0
    Type                 1
    Price                0
    Content Rating       0
    Genres               0
    Last Updated         0
    Current Ver          8
    Android Ver          2
    dtype: int64



Here, we realized that there are 1474 rows having null values under column 'Rating'. 
Hence, we decided to replace the null values with median of overall 'Rating' values.


```python
def replace_with_median(series):
    """
    Given a series, replace the rows with null values 
    with median values
    """
    return series.fillna(series.median())

apps_df['Rating'] = apps_df['Rating'].transform(replace_with_median)
apps_df['Rating'] = apps_df['Rating'].astype(float)
```


```python
# remove the record where 'Type' is having null value
index = apps_df[apps_df['Type'].isna()].index
apps_df.drop(axis=0, inplace=True, index=index)
```


```python
# check on null values
apps_df.isna().sum()
```




    App               0
    Category          0
    Rating            0
    Reviews           0
    Size              0
    Installs          0
    Type              0
    Price             0
    Content Rating    0
    Genres            0
    Last Updated      0
    Current Ver       8
    Android Ver       2
    dtype: int64




```python
# grouping the data starting with App and Reviews
apps_df = apps_df.groupby(['App', 'Reviews', 'Category', 'Rating', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 
                           'Last Updated', 'Current Ver', 'Android Ver'], as_index=False)
# reassign Installs values with their mean
apps_df = apps_df['Installs'].mean()
# sort the dataframe by Reviews descendingly
apps_df.sort_values(by='Reviews', ascending=False, inplace=True)
# drop duplicate rows based on App 
apps_df.drop_duplicates(subset=['App'], inplace=True)
apps_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Reviews</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4534</th>
      <td>Facebook</td>
      <td>78158306.0</td>
      <td>SOCIAL</td>
      <td>4.1</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Teen</td>
      <td>Social</td>
      <td>August 3, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>9661</th>
      <td>WhatsApp Messenger</td>
      <td>69119316.0</td>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Communication</td>
      <td>August 3, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>5731</th>
      <td>Instagram</td>
      <td>66577446.0</td>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Teen</td>
      <td>Social</td>
      <td>July 31, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>6546</th>
      <td>Messenger – Text and Video Chat for Free</td>
      <td>56646578.0</td>
      <td>COMMUNICATION</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Communication</td>
      <td>August 1, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>2701</th>
      <td>Clash of Clans</td>
      <td>44893888.0</td>
      <td>GAME</td>
      <td>4.6</td>
      <td>98.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone 10+</td>
      <td>Strategy</td>
      <td>July 15, 2018</td>
      <td>10.322.16</td>
      <td>4.1 and up</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4372</th>
      <td>FG Autumn Photo Puzzle</td>
      <td>0.0</td>
      <td>FAMILY</td>
      <td>4.3</td>
      <td>4.6</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Puzzle</td>
      <td>August 23, 2017</td>
      <td>1.0</td>
      <td>4.0 and up</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>4369</th>
      <td>FE Other Disciplines Engineering Exam Prep</td>
      <td>0.0</td>
      <td>FAMILY</td>
      <td>4.3</td>
      <td>21.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Education</td>
      <td>July 27, 2018</td>
      <td>5.33.3669</td>
      <td>5.0 and up</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>9033</th>
      <td>Thyroid Nodules</td>
      <td>0.0</td>
      <td>MEDICAL</td>
      <td>4.3</td>
      <td>20.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Medical</td>
      <td>July 14, 2018</td>
      <td>1.0</td>
      <td>4.3 and up</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>185</th>
      <td>ACCEPT CE MARKING</td>
      <td>0.0</td>
      <td>PRODUCTIVITY</td>
      <td>4.3</td>
      <td>30.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Productivity</td>
      <td>June 28, 2018</td>
      <td>1.0</td>
      <td>4.1 and up</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>5862</th>
      <td>K-App Mitarbeiter Galeria Kaufhof</td>
      <td>0.0</td>
      <td>PRODUCTIVITY</td>
      <td>4.3</td>
      <td>19.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Productivity</td>
      <td>July 10, 2018</td>
      <td>3.27.1</td>
      <td>4.4 and up</td>
      <td>1.000000e+02</td>
    </tr>
  </tbody>
</table>
<p>9648 rows × 13 columns</p>
</div>




```python
# check on statistical information of the dataframe
apps_df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reviews</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Price</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.648000e+03</td>
      <td>9648.000000</td>
      <td>9648.000000</td>
      <td>9648.000000</td>
      <td>9.648000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.170487e+05</td>
      <td>4.192485</td>
      <td>17.820208</td>
      <td>1.098122</td>
      <td>7.806898e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.832460e+06</td>
      <td>0.496210</td>
      <td>21.503151</td>
      <td>16.861193</td>
      <td>5.379975e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.500000e+01</td>
      <td>4.000000</td>
      <td>2.900000</td>
      <td>0.000000</td>
      <td>1.000000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.745000e+02</td>
      <td>4.300000</td>
      <td>9.200000</td>
      <td>0.000000</td>
      <td>1.000000e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.949750e+04</td>
      <td>4.500000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>1.000000e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.815831e+07</td>
      <td>5.000000</td>
      <td>100.000000</td>
      <td>400.000000</td>
      <td>1.000000e+09</td>
    </tr>
  </tbody>
</table>
</div>

## Exploratory Analysis and Visualization

### Category


```python
# get the number of apps for each category
sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
sns.countplot(x='Category', data=apps_df)
plt.title('Number of Apps Per Category')
plt.xticks(rotation=90)
plt.ylabel('Number of Apps')
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_34_0.png)
    


From this plotting we know that most of the apps in the play store are from the categories of 'Family', 'Game' and also 'Tools.


```python
# get the number of installs for each category
categories = apps_df.groupby('Category')
category_installs_sum_df = categories[['Installs']].sum()
category_installs_sum_df = category_installs_sum_df.reset_index()  # to convert groupby object into dataframe

plt.figure(figsize=(10, 5))
sns.barplot(x='Category', y='Installs', data=category_installs_sum_df)
plt.xticks(rotation=90)
plt.ylabel('Installs (e+10)')
plt.title('Number of Installs For Each Category')
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_36_0.png)
    


From this distribution plotting of number of installs for each category, we can see that most of the apps being downloaded and installed are from the categories of 'Game' and 'Communication'.

### Rating


```python
# show the distribution of rating
plt.figure(figsize=(10, 5))
sns.countplot(x='Rating', data=apps_df)
plt.title('Rating Distribution')
plt.xticks(rotation=90)
plt.ylabel('Number of Apps')
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_39_0.png)
    


From this distribution plotting, it implies that most of the apps in the Play Store are having rating higher than 4 or in the range of 4 to 4.7. 


```python
# plot the graphs of reviews, size, installs and price per rating
rating_df = apps_df.groupby('Rating').sum().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].plot(rating_df['Rating'], rating_df['Reviews'], 'r')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Reviews')
axes[0].set_title('Reviews Per Rating')

axes[1].plot(rating_df['Rating'], rating_df['Size'], 'g')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Size')
axes[1].set_title('Size Per Rating')

axes[2].plot(rating_df['Rating'], rating_df['Installs'], 'g')
axes[2].set_xlabel('Rating')
axes[2].set_ylabel('Installs (e+10)')
axes[2].set_title('Installs Per Rating')

axes[3].plot(rating_df['Rating'], rating_df['Price'], 'k')
axes[3].set_xlabel('Rating')
axes[3].set_ylabel('Price')
axes[3].set_title('Price Per Rating')

plt.tight_layout(pad=2)
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_41_0.png)
    


From the above plottings, we can imply that most of the apps with higher rating range of 4.0 - 4.7 are having high amount of reviews, size, and installs. In terms of price, it doesn't reflect a direct relationship with rating, as we could see a fluctuation in term of pricing even at the range of high rating. 

### Application Type


```python
# application type distribution
plt.figure(figsize=(10, 5))
sns.countplot(apps_df['Type'])
plt.title('Type Distribution')
plt.ylabel('Number of Apps')
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_44_0.png)
    


From the plot we can imply that majority of the apps in the Play Store are Free apps.


```python
# show the distribution of apps in term of their rating, size and type
plt.figure(figsize=(12, 6))
sns.scatterplot(apps_df['Size'],
               apps_df['Rating'],
               hue=apps_df['Type'],
               s=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0b7b0fb590>




    
![png](google-play-store-eda_files/google-play-store-eda_46_1.png)
    


From this scatter plot, we can imply that majority of the free apps are small in size and having high rating. While for paid apps, we have quite equal distribution in term on size and rating.


```python
# correlation
apps_df.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reviews</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Price</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Reviews</th>
      <td>1.000000</td>
      <td>0.050280</td>
      <td>0.037812</td>
      <td>-0.007597</td>
      <td>0.625051</td>
    </tr>
    <tr>
      <th>Rating</th>
      <td>0.050280</td>
      <td>1.000000</td>
      <td>0.027338</td>
      <td>-0.018585</td>
      <td>0.034393</td>
    </tr>
    <tr>
      <th>Size</th>
      <td>0.037812</td>
      <td>0.027338</td>
      <td>1.000000</td>
      <td>-0.015033</td>
      <td>-0.007803</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>-0.007597</td>
      <td>-0.018585</td>
      <td>-0.015033</td>
      <td>1.000000</td>
      <td>-0.009418</td>
    </tr>
    <tr>
      <th>Installs</th>
      <td>0.625051</td>
      <td>0.034393</td>
      <td>-0.007803</td>
      <td>-0.009418</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(figsize=(8, 8))
sns.heatmap(apps_df.corr(), ax=axes, annot=True, linewidths=0.1, fmt='.2f', square=True)
plt.show()
```


    
![png](google-play-store-eda_files/google-play-store-eda_49_0.png)
 

## Asking and Answering Questions



```python
# 1. What is the top 5 apps on the basis of installs?
df = apps_df.sort_values(by=['Installs'], ascending=False)
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Reviews</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4534</th>
      <td>Facebook</td>
      <td>78158306.0</td>
      <td>SOCIAL</td>
      <td>4.1</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Teen</td>
      <td>Social</td>
      <td>August 3, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>5211</th>
      <td>Google Photos</td>
      <td>10859051.0</td>
      <td>PHOTOGRAPHY</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Photography</td>
      <td>August 6, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>5229</th>
      <td>Google+</td>
      <td>4831125.0</td>
      <td>SOCIAL</td>
      <td>4.2</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Teen</td>
      <td>Social</td>
      <td>July 26, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>5122</th>
      <td>Gmail</td>
      <td>4604483.0</td>
      <td>COMMUNICATION</td>
      <td>4.3</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Communication</td>
      <td>August 2, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>5221</th>
      <td>Google Street View</td>
      <td>2129707.0</td>
      <td>TRAVEL_AND_LOCAL</td>
      <td>4.2</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Travel &amp; Local</td>
      <td>August 6, 2018</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
      <td>1.000000e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'The 5 apps that have the most number of installs are: {", ".join(df["App"].head(5))}')
```

    The 5 apps that have the most number of installs are: Facebook, Google Photos, Google+, Gmail, Google Street View



```python
# 2. What is the top 5 reviewed apps?
df = apps_df.groupby(by=['App', 'Category', 'Rating'])[['Reviews']].sum().reset_index()
df = df.sort_values(by=['Reviews'], ascending=False)
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4324</th>
      <td>Facebook</td>
      <td>SOCIAL</td>
      <td>4.1</td>
      <td>78158306.0</td>
    </tr>
    <tr>
      <th>9031</th>
      <td>WhatsApp Messenger</td>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>69119316.0</td>
    </tr>
    <tr>
      <th>5395</th>
      <td>Instagram</td>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>66577446.0</td>
    </tr>
    <tr>
      <th>6158</th>
      <td>Messenger – Text and Video Chat for Free</td>
      <td>COMMUNICATION</td>
      <td>4.0</td>
      <td>56646578.0</td>
    </tr>
    <tr>
      <th>2562</th>
      <td>Clash of Clans</td>
      <td>GAME</td>
      <td>4.6</td>
      <td>44893888.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'The 5 apps that have the most number of total reviews are: {", ".join(df["App"].head(5))}')
```

    The 5 apps that have the most number of total reviews are: Facebook, WhatsApp Messenger, Instagram, Messenger – Text and Video Chat for Free, Clash of Clans



```python
# 3. What is the top 5 expensive apps?
df = apps_df.sort_values(by=['Price'], ascending=False)
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Reviews</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5653</th>
      <td>I'm Rich - Trump Edition</td>
      <td>275.0</td>
      <td>LIFESTYLE</td>
      <td>3.6</td>
      <td>7.3</td>
      <td>Paid</td>
      <td>400.00</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>May 3, 2018</td>
      <td>1.0.1</td>
      <td>4.1 and up</td>
      <td>10000.0</td>
    </tr>
    <tr>
      <th>5654</th>
      <td>I'm Rich/Eu sou Rico/أنا غني/我很有錢</td>
      <td>0.0</td>
      <td>LIFESTYLE</td>
      <td>4.3</td>
      <td>40.0</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>December 1, 2017</td>
      <td>MONEY</td>
      <td>4.1 and up</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5643</th>
      <td>I am Rich Plus</td>
      <td>856.0</td>
      <td>FAMILY</td>
      <td>4.0</td>
      <td>8.7</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>May 19, 2018</td>
      <td>3.0</td>
      <td>4.4 and up</td>
      <td>10000.0</td>
    </tr>
    <tr>
      <th>5648</th>
      <td>I am rich (Most expensive app)</td>
      <td>129.0</td>
      <td>FINANCE</td>
      <td>4.1</td>
      <td>2.7</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Teen</td>
      <td>Finance</td>
      <td>December 6, 2017</td>
      <td>2</td>
      <td>4.0.3 and up</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>5627</th>
      <td>I Am Rich Premium</td>
      <td>1867.0</td>
      <td>FINANCE</td>
      <td>4.1</td>
      <td>4.7</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>November 12, 2017</td>
      <td>1.6</td>
      <td>4.0 and up</td>
      <td>50000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'The top 5 most expensive apps in the store are: {", ".join(df["App"].head(5))}')
```

    The top 5 most expensive apps in the store are: I'm Rich - Trump Edition, I'm Rich/Eu sou Rico/أنا غني/我很有錢, I am Rich Plus, I am rich (Most expensive app), I Am Rich Premium



```python
# 4. What is the top 3 most installed apps in Game category?
df = apps_df[apps_df['Category'] == 'GAME']
df = df.sort_values(by=['Installs'], ascending=False)
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Reviews</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8619</th>
      <td>Subway Surfers</td>
      <td>27725352.0</td>
      <td>GAME</td>
      <td>4.5</td>
      <td>76.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone 10+</td>
      <td>Arcade</td>
      <td>July 12, 2018</td>
      <td>1.90.0</td>
      <td>4.1 and up</td>
      <td>1.000000e+09</td>
    </tr>
    <tr>
      <th>2480</th>
      <td>Candy Crush Saga</td>
      <td>22430188.0</td>
      <td>GAME</td>
      <td>4.4</td>
      <td>74.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Casual</td>
      <td>July 5, 2018</td>
      <td>1.129.0.2</td>
      <td>4.1 and up</td>
      <td>5.000000e+08</td>
    </tr>
    <tr>
      <th>6849</th>
      <td>My Talking Tom</td>
      <td>14892469.0</td>
      <td>GAME</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>Free</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>Casual</td>
      <td>July 19, 2018</td>
      <td>4.8.0.132</td>
      <td>4.1 and up</td>
      <td>5.000000e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'The top 3 most expensive apps in the GAME category are: {", ".join(df["App"].head(3))}')
```

    The top 3 most expensive apps in the GAME category are: Subway Surfers, Candy Crush Saga, My Talking Tom



```python
# 5. Which 5 apps from the 'FAMILY' category are having the lowest rating?
df = apps_df[apps_df['Category'] == 'FAMILY']
df = df.sort_values(by=['Rating'], ascending=True)
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Reviews</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Size</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Installs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4366</th>
      <td>FE Mechanical Engineering Prep</td>
      <td>2.0</td>
      <td>FAMILY</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>Free</td>
      <td>0.00</td>
      <td>Everyone</td>
      <td>Education</td>
      <td>July 27, 2018</td>
      <td>5.33.3669</td>
      <td>5.0 and up</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>9187</th>
      <td>Truck Driving Test Class 3 BC</td>
      <td>1.0</td>
      <td>FAMILY</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Paid</td>
      <td>1.49</td>
      <td>Everyone</td>
      <td>Education</td>
      <td>April 9, 2012</td>
      <td>1.0</td>
      <td>2.1 and up</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>8494</th>
      <td>Speech Therapy: F</td>
      <td>1.0</td>
      <td>FAMILY</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>Paid</td>
      <td>2.99</td>
      <td>Everyone</td>
      <td>Education</td>
      <td>October 7, 2016</td>
      <td>1.0</td>
      <td>2.3.3 and up</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>AC REMOTE UNIVERSAL-PRO</td>
      <td>402.0</td>
      <td>FAMILY</td>
      <td>1.6</td>
      <td>1.7</td>
      <td>Free</td>
      <td>0.00</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>December 11, 2015</td>
      <td>1.0</td>
      <td>2.2 and up</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>BG TV App</td>
      <td>6.0</td>
      <td>FAMILY</td>
      <td>1.7</td>
      <td>2.9</td>
      <td>Free</td>
      <td>0.00</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>December 21, 2017</td>
      <td>1.0</td>
      <td>4.1 and up</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'The 5 apps from the FAMILY category having the lowest rating are: {", ".join(df["App"].head(5))}')
```

    The 5 apps from the FAMILY category having the lowest rating are: FE Mechanical Engineering Prep, Truck Driving Test Class 3 BC, Speech Therapy: F, AC REMOTE UNIVERSAL-PRO, BG TV App


## Inferences and Conclusion

The Google Play Store Apps report provides some useful insights regarding the trending of the apps in the play store. As per the graphs visualizations shown above, most of the trending apps (in terms of users' installs) are from the categories like GAME, COMMUNICATION, and TOOL even though the amount of available apps from these categories are twice as much lesser than the category FAMILY. The trending of these apps are most probably due to their nature of being able to entertain or assist the user. Besides, it also shows a good trend where we can see that developers from these categories are focusing on the quality instead of the quantity of the apps.

Other than that, the charts shown above actually implies that most of the apps having good ratings of above 4.0 are mostly confirmed to have high amount of reviews and user installs. There are some spikes in term of size and price but it shouldn't reflect that apps with high rating are mostly big in size and pricy as by looking at the graphs they are most probably are due to some minority. Futhermore, most of the apps that are having high amount of reviews are from the categories of SOCIAL, COMMUNICATION and GAME like Facebook, WhatsApp Messenger, Instagram, Messenger – Text and Video Chat for Free, Clash of Clans etc. 

Eventhough apps from the categories like GAME, SOCIAL, COMMUNICATION and TOOL of having the highest amount of installs, rating and reviews are reflecting the current trend of Android users, they are not even appearing as category in the top 5 most expensive apps in the store (which are mostly from FINANCE and LIFESTYLE). As a conclsuion, we learnt that the current trend in the Android market are mostly from these categories which either assisting, communicating or entertaining apps. 
