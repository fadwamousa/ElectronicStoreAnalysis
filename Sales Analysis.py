#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ###### Merge 12 Months Tables into one Table 

# In[2]:


files = [file for file in os.listdir('Sales_Data')]
all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data , df])
all_months_data.to_csv('all_data.csv',index=False)


# In[3]:


df = pd.read_csv("all_data.csv")
df.head(1)


# In[4]:


df.dtypes


# #### rename the columns name

# In[5]:


df.rename(columns = {'Order ID':'Order_ID',
                     'Quantity Ordered':'Quantity_Ordered',
                     'Price Each':'Price_Each',
                     'Order Date':'Order_Date',
                     'Purchase Address':'Purchase_Address'},inplace=True)


# In[6]:


df.dtypes


# #### convert and transform datatypes

# - let's check about nan values in our columns

# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(inplace = True)


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[12]:


df.head(1)


# In[13]:


df['month'] = df['Order_Date'].str[0:2]


# In[16]:


df[df['Order_Date'].str[0:2] == 'Or']


# In[17]:


df = df[df['Order_Date'].str[0:2] != 'Or']


# In[18]:


df['month'] = df['month'].astype('int32')


# ##### Test

# In[19]:


df['month'].dtypes


# > The Question is what is the month that sales are the best?

# In[15]:


df.head(2)


# - Let's create new column called sales 

# In[20]:


df['Price_Each'].dtypes


# In[21]:


df['Price_Each'] = df['Price_Each'].astype(float)


# In[22]:


df['Price_Each'].dtypes


# In[23]:


df['Quantity_Ordered'].dtypes


# In[24]:


df['Quantity_Ordered'] = df['Quantity_Ordered'].astype(int)


# In[25]:


df['Quantity_Ordered'].dtypes


# In[26]:


df['sales'] = df['Quantity_Ordered'] * df['Price_Each']


# In[27]:


df.head(3)


# In[28]:


df_sales = df.groupby('month').sum()['sales']
#df_sales
pd.DataFrame(df_sales)


# - The best Sales was happend in 12 = 4.613443

# In[29]:


df_sales = df.groupby('month').sum()['sales'].max()
df_sales


# In[30]:


Result  = df.groupby('month').sum()
months  = range(1,13)

plt.bar(months ,Result['sales']);
plt.xlabel('Months')
plt.ylabel('Sales')


# ### What is the city have a higher number in sales ?

# - Let's add a new column called City

# In[31]:


df.head(3)


# In[32]:


df['city'] = df['Purchase_Address'].apply(lambda x:x.split(',')[1])


# In[33]:


df.head()


# - anthor way to do that

# In[34]:


def get_city(address):
    return address.split(',')[1]
df['Purchase_Address'].apply(lambda x:get_city(x))


# In[35]:


city = df.groupby('city').sum()['sales']
pd.DataFrame(city)


# - San Francisco is the best sales 

# In[36]:


df.groupby('city').sum()['sales'].max()


# In[37]:


Result  = df.groupby('city').sum()
#cities  = df.city.unique()
cities  = [city for city,df in df.groupby('city')]


# In[38]:


plt.bar(cities ,Result['sales']);
plt.xticks(cities,rotation = 'vertical')
plt.xlabel('cities');
plt.ylabel('Sales');
plt.tight_layout()


# #### what time should we display advertisment to maxmize of customers buying products? 

# In[39]:


df.head()


# In[40]:


df['Order_Date'].dtypes


# In[41]:


df['Order_Date'] = pd.to_datetime(df['Order_Date'])


# In[42]:


df['hour'] = df['Order_Date'].dt.hour


# In[43]:


df.head()


# In[49]:


df_hour = df.groupby('hour').sum()['sales'].sort_values(ascending = False).head(1)
pd.DataFrame(df_hour)


# In[51]:


hours = [hour for hour,df in df.groupby('hour')]
plt.plot(hours , df.groupby('hour').sum()['sales'].sort_values(ascending = True))
plt.grid()
plt.xticks(hours);
plt.xlabel('Hours');
plt.ylabel('count');


# In[52]:


hours = [hour for hour,df in df.groupby('hour')]


# In[56]:


plt.plot(hours, df.groupby('hour').count()['Order_Date'],color = 'black');
plt.grid()
plt.xticks(hours);
plt.xlabel('Hours');
plt.ylabel('count');
#plt.plot(hours, df.groupby('hour').count(),color = 'black');


# #### what products are most often soild? 

# In[50]:


df.head()


# In[57]:


df.Product.value_counts().sort_values(ascending = False).head(1)


# In[58]:


df.Product.value_counts().sort_values(ascending = False).plot();
plt.xticks(rotation = 'vertical');


# #### what products are most often soild together? 

# In[53]:


df.head()


# In[70]:


New_Data = df[df['Order_ID'].duplicated(keep=False)]


# In[71]:


df['grouped'] = df.groupby('Order_ID')['Product'].transform(lambda x:','.join(x))


# In[72]:


df.head()


# ### What product sold the most ?

# In[73]:


df.groupby('Product').sum()['Quantity_Ordered']


# In[75]:


df.groupby('Product').sum()['Quantity_Ordered'].sort_values(ascending = False).head(1)


# In[77]:




product_group=df.groupby('Product')
quantity_ordered=product_group.sum()['Quantity_Ordered']

products=[product for product ,df_o in product_group ]

plt.bar(products,quantity_ordered )

plt.xticks(products, rotation='vertical',size =8)

plt.ylabel('Quantity_Ordered')
plt.xlabel('Product ')
plt.show()


# In[78]:


df.groupby('Product').mean()['Price_Each']


# In[60]:


prices=df.groupby('Product').mean()['Price_Each']

fig , ax1=plt.subplots()

ax2=ax1.twinx()
ax1.bar(products,quantity_ordered)
ax2.plot(products,prices ,'b-', color='green')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered ', color='g')
ax1.set_ylabel('Price ', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=8)
plt.show()



# In[ ]:




