#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:30:21 2019

@author: root
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

dir = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(dir,"data\\Train.csv"))
columns = train.keys()

print(columns)

plt.figure(figsize=(12,9))
plt.plot(train["Soil humidity 1"], label="Slot 1")
plt.plot(train["Soil humidity 2"], label="Slot 2")
plt.plot(train["Soil humidity 3"], label="Slot 3")
plt.plot(train["Soil humidity 4"], label="Slot 4")
plt.legend()
plt.savefig("slot.png")
plt.show()

## How many time points without nan do we have for every field ?
length = train.shape[0]
na_1 = train["Soil humidity 1"].isna().sum()
na_2 = train["Soil humidity 2"].isna().sum()
na_3 = train["Soil humidity 3"].isna().sum()
na_4 = train["Soil humidity 4"].isna().sum()

time_points_1 = length - na_1
time_points_2 = length - na_2
time_points_3 = length - na_3
time_points_4 = length - na_4

time_points_str = "There are {} valid time points for the slot {}"

print("             ----------------             ")
print(time_points_str.format(time_points_1, 1))
print(time_points_str.format(time_points_2, 2))
print(time_points_str.format(time_points_3, 3))
print(time_points_str.format(time_points_4, 4))
print("             ----------------             ")

## How many days were collected ?
days_1 = int(time_points_1 * 5 / (50 * 24))
days_2 = int(time_points_2 * 5 / (50 * 24))
days_3 = int(time_points_3 * 5 / (50 * 24))
days_4 = int(time_points_4 * 5 / (50 * 24))

days_str = "Data from slot {} were roughly collected over {} days"

print("             ----------------             ")
print(days_str.format(1, days_1))
print(days_str.format(2, days_2))
print(days_str.format(3, days_3))
print(days_str.format(4, days_4))
print("             ----------------             ")
