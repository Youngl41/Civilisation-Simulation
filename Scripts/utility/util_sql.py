#======================================================
# Utility: Pandas Mayflower SQL execution
#======================================================
'''
Version 1.0
Pre-requisites:
    pip install psycopg2-binary
    conda install -c conda-forge sqlalchemy-redshift
'''
import datetime
import time

import numpy as np
# Import modules
import pandas as pd
import psycopg2
import sqlparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine

import pyodbc
#from config.aladdin_config import *


#------------------------------
# Main
#------------------------------
# Connection params
dbname=
user=
pass_'Thanos$123'

# Engine
connstr = 
engine = create_engine(connstr, connect_args={'sslmode': 'prefer'})

# Execute query
def get_table(sql_query, engine=engine):
    with engine.connect() as conn, conn.begin():
        df = pd.read_sql(sql_query, conn)
    return df

# Run query
def run_query(connstring, querypath, **kwargs):
    '''Gets query and parameters as inputs and returns a dataframe'''
    #AC = kwargs.pop('AC')
    #conn = pyodbc.connect(connstring, autocommit=AC)
    conn = create_engine(connstring, connect_args={'sslmode': 'prefer'}).raw_connection()
    cursor = conn.cursor()
    querysql = open(querypath, 'r').read()
    for name,value in kwargs.items():
        querysql = querysql.replace("{" + str(name) + "}",str(value))
    statements = sqlparse.split(querysql)
    for statement in statements[:-1]:
        print (statement[:50], end='...')
        cursor.execute(statement + ';') 
    df = pd.io.sql.read_sql(statements[-1] + ';', conn)
    conn.close
    return df


#------------------------------
# Unit Test
#------------------------------
