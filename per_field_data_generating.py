import pandas as pd
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(root,'data')
train_df = pd.read_csv(os.path.join(data_directory,'Train.csv'))

def build_soil_df(constant_keys,inconsistent_keys,k):
    keys = constant_keys + [elem.format(k) for elem in inconsistent_keys]
    return train_df[keys]

def clean_soild_humidity_df(df,k):
    tmp_df = df.where(df['Soil humidity {}'.format(k)] >0)
    return tmp_df[np.isfinite(tmp_df['Soil humidity {}'.format(k)])]

def get_soil_df(constant_keys,inconsistent_keys,k):
    df = build_soil_df(constant_keys,inconsistent_keys,k)
    df = clean_soild_humidity_df(df,k)
    return df

def build_separate_datasets():
    i = 1
    while i < 5:
        df = get_soil_df(['timestamp','Air temperature (C)','Air humidity (%)','Pressure (KPa)','Wind speed (Km/h)','Wind gust (Km/h)','Wind direction (Deg)'],
                    ['Soil humidity {}', 'Irrigation field {}'],
                    i)
        df.to_csv(os.path.join(data_directory,'Train_field_{}.csv'.format(i)))
        i+=1
    print(" All datasets were separately generated")

build_separate_datasets()
