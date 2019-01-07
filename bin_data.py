# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:47:07 2018

@author: Zhengqian Xu 
"""

## import library
import pandas as pd
import numpy as np


## main function
def main():
    #### load data
    df1 = pd.read_csv("returns.csv",index_col = 0)
    
    
    #### get the column name of data 
    ncol = list(df1)
    
    #### cumpute the standard deviation for each column
    sd = []

    for i in range(0,len(ncol)):
        sd1 = np.std(df1[ncol[i]])
        sd.append(sd1)

    #### create a dataframe for each column and its standard deviation
    dic1 = {'name': ncol[0:53], 'std': sd[0:53]}
    df3 = pd.DataFrame(data=dic1)
    
    #### sort df3 by the column std
    df4 = df3.sort_values(by=['std'])
    
    #### we partition the data into 3 groups with equal-depth, depth = 18
    low = df4.iloc[0:18,:]
    medi = df4.iloc[18:36,:]
    high = df4.iloc[36:53,:]
    print("\nlow risk company:\n",low['name'],"\n")
    print("\nmedian risk company:\n",medi['name'],"\n")
    print("\nhigh risk company:\n",high['name'],"\n")
    
    
    #### we divide original datat into 3 dataframes according to the different risk level
    dflow = df1[low['name'].values]
    dfmedian = df1[medi['name'].values]
    dfhigh = df1[high['name'].values]
    
    #### for each risk level, we calculate the mean of returns by rows - that is, everyday's mean of returns of this risk level group 
    dic2 = {'low': dflow.mean(axis=1).values,'median': dfmedian.mean(axis=1).values,'high':dfhigh.mean(axis=1).values}
    dfmean = pd.DataFrame(data=dic2)
    dfmean.to_csv("risk2.csv")

main()