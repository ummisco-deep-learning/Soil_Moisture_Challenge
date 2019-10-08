#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import r2_score
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%%
# data = pd.read_csv("/Users/admin/Documents/ML/Concours/data/Train.csv")
# data["timestamp"] = data["timestamp"].astype('datetime64[ns]')
# print(data.head())
# print(data.dtypes)
# print(data.columns)


# #%%
# fig = plt.figure()

# plt.subplot(2, 2, 1)
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'r', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 2)
# plt.plot(data["timestamp"], data["Soil humidity 2"], 'green', label="field2")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 3)
# plt.plot(data["timestamp"], data["Soil humidity 3"], 'blue', label="field3")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 4)
# plt.plot(data["timestamp"], data["Soil humidity 4"], 'yellow', label="field4")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.show()

# #%%
# plt.subplot(2, 2, 1)
# plt.plot(data["timestamp"], data["Air temperature (C)"], 'r', label="air")
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'gray', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 2)
# plt.plot(data["timestamp"], data["Air humidity (%)"], 'green', label="humidite")
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'r', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 3)
# plt.plot(data["timestamp"], data["Pressure (KPa)"], 'blue', label="pression")
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'r', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.subplot(2, 2, 4)
# plt.plot(data["timestamp"], data["Wind speed (Km/h)"], 'yellow', label="vitess vent")
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'r', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')



# #%%

# # plt.hist(data["Irrigation field 1"])
# plt.plot(data["timestamp"], data["Soil humidity 1"], 'r', label="field1")
# plt.legend(loc='upper right')
# plt.xticks(rotation=45, ha='right')

# plt.show()

# #%%

# from statsmodels.graphics import tsaplots
# import statsmodels.api as sm
# from pylab import rcParams

# #rcParams['figure.figsize'] = 11, 9
# decomposition = sm.tsa.seasonal_decompose(data["Soil humidity 4"], freq=3)
# decomp_trend = decomposition.trend

# #fig = decomposition.plot()
# #fig = tsaplots.plot_acf(data["Soil humidity 1"], lags=40)

# plt.show()

#%%
def get_data_field(file_data_path):
    data = pd.read_csv(file_data_path)
    data["timestamp"] = data["timestamp"].astype('datetime64[ns]')
    cols = []
    for i in range(4):
        cols.append(['timestamp', 'Soil humidity '+ str(i+1), 'Irrigation field '+ str(i+1), 
                'Air temperature (C)','Air humidity (%)', 
                'Pressure (KPa)', 'Wind speed (Km/h)',
                'Wind gust (Km/h)', 'Wind direction (Deg)'
                ])
    
    data_field1, data_field2, data_field3, data_field4  = data[cols[0]], data[cols[1]], data[cols[2]], data[cols[3]]
    return data_field1[np.isfinite(data['Soil humidity 1'])], data_field2[np.isfinite(data['Soil humidity 2'])], data_field3[np.isfinite(data['Soil humidity 3'])], data_field4[np.isfinite(data['Soil humidity 4'])]
    

#%%
path =  "/Users/admin/Documents/ML/Concours/data/Train.csv"

data_field1, data_field2, data_field3, data_field4 = get_data_field(path)


#data_field1
data_field2
#data_field3.head()
#data_field4.head()
#%%
plt.subplot(2, 2, 1)
#plt.plot(data_field2["timestamp"], data["Air temperature (C)"], 'r', label="air")
plt.plot(data_field1["timestamp"], data_field1["Soil humidity 1"]/data_field1["Soil humidity 1"].max(), 'yellow', label="field1")
plt.plot(data_field1["timestamp"], data_field1["Irrigation field 1"]/4, 'black', label="irrigation")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 2)
plt.plot(data_field2["timestamp"], data_field2["Soil humidity 2"]/ data_field2["Soil humidity 2"].max(), 'green', label="field2")
#plt.plot(data_field2["timestamp"], data["Soil humidity 1"], 'r', label="field1")
plt.plot(data_field2["timestamp"],data_field2["Irrigation field 2"]/4, 'black', label="irrigation")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 3)
#plt.plot(data_field2["timestamp"], data["Pressure (KPa)"], 'blue', label="pression")
plt.plot(data_field3["timestamp"], data_field3["Soil humidity 3"]/data_field3["Soil humidity 3"].max(), 'b', label="field3")
plt.plot(data_field3["timestamp"],data_field3["Irrigation field 3"]/4, 'black', label="irrigation")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 4)
#plt.plot(data_field2["timestamp"], data["Wind speed (Km/h)"], 'yelloblack', label="vitess vent")
plt.plot(data_field4["timestamp"], data_field4["Soil humidity 4"]/data_field4["Soil humidity 4"].max(), 'r', label="field4")
plt.plot(data_field4["timestamp"],data_field4["Irrigation field 4"]/4, 'black', label="irrigation")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

#%%

plt.show()


#%% [markdown]
# ### Load the dataset :
# * Load train and test data into pandas DataFrames
# * Combine train and test data to process them together

#%%
# def get_data(field_numbe):
#     #get train data
#     train_data_path ='train.csv'
#     train = pd.read_csv(train_data_path)
    
#     #get test data
#     test_data_path ='test.csv'
#     test = pd.read_csv(test_data_path)
    
#     return train , test

# def get_combined_data():
#   #reading train data
#   train , test = get_data()

#   target = train.SalePrice
#   train.drop(['SalePrice'],axis = 1 , inplace = True)

#   combined = train.append(test)
#   combined.reset_index(inplace=True)
#   combined.drop(['index', 'Id'], inplace=True, axis=1)
#   return combined, target

#%%
from sklearn.model_selection import train_test_split

data_field1, data_field2, data_field3, data_field4 = get_data_field(path)
target = data_field1["Soil humidity 1"]
cols = ['Air temperature (C)','Air humidity (%)', 
                'Pressure (KPa)', 'Wind speed (Km/h)',
                'Wind gust (Km/h)', 'Wind direction (Deg)']
feature = data_field1[cols]
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.33, random_state=4)
combined = X_train.append(X_test)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
print ('combine:',  combined.shape)


#%% [markdown]
# let's define a function to get the columns that don't have any missing values 

#%%
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


#%%
num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')



#%% [markdown]
# Let's see how many columns we got

#%%
print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))


#%%
combined = combined[num_cols + cat_cols]
combined.hist(figsize = (12,10))
plt.show()


#%% [markdown]
# **The correlation between the features**

#%%
import seaborn as sb

C_mat = data_field1.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .9)
plt.show()


#%%
