#======================================================
# General Utility Functions
#======================================================
'''
Version 1.0
Utility functions for general applications
'''
# Import modules
import numpy as np
import pandas as pd
import sys
#import dill as pickle
import pickle
from datetime import datetime
main_dir = '/Users/younle/Documents/projects/promo_insights'
sys.path.append(main_dir)
from config.config import *


#------------------------------
# Utility Functions
#------------------------------
# Set title
def set_title(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 57
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        lr_buffer_len   = int((max_length - string_size) / 2)
        full_buffer_len = lr_buffer_len * 2 + string_size
        print('\n')
        print(full_buffer_len * '=')
        print(full_buffer_len * ' ')
        print(lr_buffer_len * ' ' + string + lr_buffer_len * ' ')
        print(full_buffer_len * ' ')
        print(full_buffer_len * '='+'\n\n')

# Set section
def set_section(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 100
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        full_buffer_len = string_size
        print('\n')
        print(full_buffer_len * '-')
        print(string)
        print(full_buffer_len * '-'+'\n')

# Print time taken
def print_dur(string, st):
    print(string, datetime.now() - st)

# Date conversion
def pdf_cast_date(df, date_field):
    #df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M'), list(df.loc[:, date_field])))
    #df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S'), list(df.loc[:, date_field])))
    df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M'), list(df.loc[:, date_field])))
    return df

# Create date table
def create_date_table(start='2000-01-01', end='2050-12-31'):
    start_ts = pd.to_datetime(start).date()

    end_ts = pd.to_datetime(end).date()

    # Record timetsamp is empty for now
    dates =  pd.DataFrame(columns=['Record_timestamp'],
        index=pd.date_range(start_ts, end_ts))
    dates.index.name = 'date'

    days_names = {
        i: name
        for i, name
        in enumerate(['Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday', 
                      'Sunday'])
    }

    dates['day'] = dates.index.dayofweek.map(days_names.get)
    dates['week'] = dates.index.week
    dates['month'] = dates.index.month
    dates['quarter'] = dates.index.quarter
    dates['year_half'] = dates.index.month.map(lambda mth: 1 if mth <7 else 2)
    dates['year'] = dates.index.year
    dates.reset_index(inplace=True)
    dates = dates.drop('Record_timestamp', axis=1)
    return dates


#==============================================================================
# Save Data
#==============================================================================
def save_data(data, save_file):
    # Define output file
    pickle.dump(data, open(save_file, 'wb')) 

def load_data(file_path):
    return pickle.load(open(file_path, 'rb'))
