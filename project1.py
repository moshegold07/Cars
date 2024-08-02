#!/usr/bin/env python
# coding: utf-8

# ##### מגישים:
# ##### משה גולדזנד ת.ז 312486046
# ##### מנחם פרל ת.ז 318836962

# #### Importing necessary libraries
# ###### Imports the required libraries for web scraping (requests, BeautifulSoup), data handling (pandas), and date-time operations (datetime).

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


# #### Define the function to scrape car data
# ###### Function Description
# The get_cars function scrapes car listings from a specific website, extracts relevant details, and returns a list of car dictionaries containing the extracted data. Here's a detailed explanation of what the function does:
# 
# Initialization:
# 
# The function initializes an empty list, car_list, to store the car data and sets an index for numbering the cars.
# It sets the base URL for scraping the car listings.
# Iterate through Pages:
# 
# The function iterates through multiple pages of car listings (adjustable range).
# For each page, it constructs the URL and sends a GET request to fetch the HTML content.
# Parse HTML Content:
# 
# The function uses BeautifulSoup to parse the HTML content of each page.
# It extracts car details such as manufacturer, model, and year from the HTML elements.
# Filter and Extract Details:
# 
# The function filters out non-Hyundai cars and those older than 2014.
# It extracts additional details like price, creation and republication dates, the number of pictures, and the description.
# 
# Navigate to Car Details Page:
# For each car, it navigates to the car's details page and extracts more specific information such as hand, gear type, engine capacity, engine type, previous ownership, current ownership, area, city, color, kilometers, and test expiry date.
# 
# Store Extracted Data:
# The extracted data is stored in a dictionary and appended to the car_list.
# The function handles any exceptions during the extraction process and continues to the next page if an error occurs.
# 
# Return the List:
# After iterating through all the pages, the function returns the car_list containing the extracted car data.
# This function effectively scrapes, filters, and extracts car data from a specified website, handling various edge cases and exceptions to ensure the data's accuracy and completeness.

# In[2]:


# Define the function to scrape car data
def get_cars():
    car_list = []
    index = 1
    
    # Base URL for scraping
    base_url = 'https://www.ad.co.il/car?sp261=13895&pageindex='
    for page in range(1, 9 + 1):  # Adjust the range as needed
        url = f"{base_url}{page}"
        response = requests.get(url)
        if not response.status_code == 200:
            print(f"Failed to retrieve page {page}, status code:", response.status_code)
            continue

        try:
            # Parse the results page and extract car data
            results_page = BeautifulSoup(response.content, 'html.parser')
            cars = results_page.find_all('div',{'class':'card-body p-md-3'}) 
            for car in cars:
                car_type = car.find('h2').get_text()
                car_type = car_type.split()
                manufactor = car_type[0]
                if manufactor != 'יונדאי': continue
                model = car_type[1]
                Year = car.find('p',class_="card-text my-1 mb-1 mt-0")
                
                try:
                    try:
                        Year = int(Year.get_text())
                    except:
                        Year = car.find('div',class_="card-text my-1 mb-1 mt-0 d-flex justify-content-between") 
                        Year = int(Year.get_text().split()[0])
                    if Year < 2014: continue
                except:
                    continue
                
                # Extract price and additional details
                try:
                    price = car.find('div',class_="price ms-1").get_text()
                except: 
                    price = None
                
                car_link = "https://www.ad.co.il" + car.find('a').get('href')
                new_response = requests.get(car_link)
                car_results = BeautifulSoup(new_response.content, 'html.parser')
                
                dates = car_results.find_all('div',{'class':'px-3'})
                count = 1
                for date in dates:
                    if count == 1:
                        Cre_date = date.get_text(strip=True).split()[2]
                        Cre_date = pd.to_datetime(Cre_date, dayfirst=True).date()
                    if count == 2:
                        Repub_date = date.get_text(strip=True).split()[3]
                        Repub_date = pd.to_datetime(Repub_date, dayfirst=True).date()
                    count += 1
                
                Pic_num = len(car_results.find_all('div',{'class':'justify-content-center px-1'}))
                
                try:
                    Description = car_results.find('p',class_="text-word-break").get_text()
                except:
                    Description = None
            
                # Extract specific information from the car details page
                details_table = car_results.find('table', class_='table table-sm mb-4')
                if details_table:
                    details = details_table.find_all('tr')
                    car_info = {}
                    for detail in details:
                        try:
                            datas = detail.find_all('td')
                            count = 1
                            for data in datas:
                                if count == 1:
                                    key = data.get_text(strip=True)
                                if count == 2:
                                    value = data.get_text(strip=True)
                                count += 1
                            car_info[key] = value
                        except AttributeError as e:
                            print(f"Error extracting key/value: {e}")
                            continue
                                                                       
                # Store the extracted data in a dictionary and add to the list                   
                car_dict = {
                    'index':index,
                    'manufacturer': manufactor,
                    'Year': Year,
                    'model': model,
                    'Hand': car_info.get('יד'),
                    'Gear': car_info.get('ת. הילוכים'),
                    'Engine_capacity': car_info.get('נפח'),
                    'Engine_type': car_info.get('סוג מנוע'),
                    'Prev_ownership': car_info.get('בעלות קודמת'),
                    'Curr_ownership': car_info.get('בעלות נוכחית'),
                    'Area': car_info.get('אזור'),
                    'City': car_info.get('עיר'),
                    'Price': price,
                    'Pic_num': Pic_num,
                    'Cre_date': Cre_date,
                    'Repub_date': Repub_date,
                    'Description': Description,
                    'Color': car_info.get("צבע"),
                    'Km': car_info.get('ק"מ'),
                    'Test': car_info.get("טסט עד"),
                }
                
                index+=1
                car_list.append(car_dict)
                
        except Exception as e:
            print(f"An error occurred on page {page}: {e}")
            continue
            
    return car_list


# #### Scrape car data

# In[3]:


cars = get_cars()


# #### Define categorical lists
# ###### Defines categorical lists for gear types, engine types, previous ownership, and current ownership.

# In[4]:


Gear_list  = pd.Categorical( ["אוטומטית", "ידנית", "טיפטרוניק", "רובוטית"])
Engine_type_list = pd.Categorical(["בנזין", "דיזל", "גז", "היבריד", "חשמלי"]) 
Prev_ownership_list = pd.Categorical(["פרטית" ,"חברה", "השכרה", "ליסינג", "מונית", "לימוד נהיגה", "ייבוא אישי", "ממשלתי", "אחר"])
cre_ownership_list = pd.Categorical(["פרטית" ,"חברה", "השכרה", "ליסינג", "מונית", "לימוד נהיגה", "ייבוא אישי", "ממשלתי", "אחר"])


# #### Ensure the data is part of a category
# ###### Validates the categorical data to ensure it falls within the defined categories. If not, it sets the value to None.

# In[5]:


for car in cars:
    if car.get('Gear') not in Gear_list:
        car['Gear'] = None

    if car.get('Engine_type') not in Engine_type_list:
        car['Engine_type'] = None

    if car.get('prev_ownership') in cre_ownership_list:
        car['prev_ownership'] = None

    if car.get('curr_ownership') in cre_ownership_list:
        car['curr_ownership'] = None


# #### Calculate remaining test days
# ###### Calculates the remaining days until the car's test expiry date and updates the Test field accordingly.

# In[6]:


for car in cars:
    if car.get('Test') != None:
        Test_date=(pd.to_datetime(car.get('Test'),dayfirst=True).date()+pd.offsets.MonthEnd(0)).date()
        today = datetime.today().date()
        car['Test'] = int((Test_date - today).days)
    else:
        car['Test'] = None


# #### Convert engine capacity to integers
# ###### Converts the engine capacity from string to integer, handling any exceptions by setting the value to None.

# In[7]:


for car in cars:
    try:
        car['Engine_capacity'] = int(car.get('Engine_capacity').replace(',', ''))
    except:
        car['Engine_capacity'] = None


# #### Convert Km and price to integers
# ###### Converts the kilometers (Km) and price from string to integer, handling any exceptions by setting the value to None.

# In[8]:


for car in cars:
    try:
        car['Km'] = int(car.get('Km').replace(',', ''))
    except:
        car['Km'] = None

    try:
        car['Price'] = float(car['Price'].replace('₪', '').replace(',', ''))
    except: 
        car['Price'] = None


# ##### Convert cleaned data to DataFrame
# ###### Converts the list of car dictionaries into a pandas DataFrame for easier data manipulation and analysis.

# In[9]:


df = pd.DataFrame(cars)


# #### Convert specific columns to categorical types
# ###### Converts specific columns to categorical types for better data handling and analysis.

# In[10]:


df['Gear'] = df['Gear'].astype('category')
df['Engine_type'] = df['Engine_type'].astype('category')
df['Prev_ownership'] = df['Prev_ownership'].astype('category')
df['Curr_ownership'] = df['Curr_ownership'].astype('category')


# #### Save to CSV 
# ###### save the DataFrame to a CSV file for future use.

# #### Display DataFrame

# In[11]:


df


# In[12]:


df.to_csv('cars.csv', index=False)


# In[ ]:




