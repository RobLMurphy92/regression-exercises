import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from env import host, user, password



# Function for acquiring and prepping my student_grades df.

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')
    
    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = grades.dropna()
    
    # Convert all columns to int64 data types.
    df = df.astype('int')
    
    return df


import pandas as pd
import numpy as np
import os
from env import host, user, password
import seaborn as sns
from pydataset import data
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

###################### Acquire Telco_Churn Data ######################
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#####################################################################

def telco_churn_data():
    '''
    This function reads the telco_churn data filtered to those with two year contract from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = '''
    select customer_id, tenure, monthly_charges, total_charges
    FROM customers
    where contract_type_id =3;
    '''
    
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df


####################################################################



def get_telco_churn(cached=False):
    '''
    This function reads in telco_churn data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco_churn df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('telco_churn.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = telco_churn_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    return df


###############################################################################
#genreal split
def general_split(df, stratify_var):
    '''
    This function take in the telco_churn_data acquired by get_telco_churn,
    performs a split and stratifies total_charges column.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, stratify = stratify_var)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, stratify = stratify_var)
    return train, validate, test



######################################################################
#prepare
#######################train, validate, test split#####################
def telco_split(df):
    '''
    This function take in the telco_churn_data acquired by get_telco_churn,
    performs a split and stratifies total_charges column.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, stratify = 'churn')
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, stratify = 'churn' )
    return train, validate, test


#################### data prep###########################
def prep_telco(df):
    '''
    This function take in the telco_churn data acquired by get_telco(),
    Returns prepped train, validate,test dfs with total_charges a float64 dtype, dfs with Nan values replaced,
    tenure renamed to tenure_months, the column tenure_year added.
    '''
    # drop duplciates
    df.drop_duplicates(inplace =True)
    # creating df which has empty values both objects and nunmerics replaced with a nan value.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # replacing the NaN value in total_charges of the customers with 0 tenure with their monthly_charge.
    df.total_charges = df.total_charges.fillna(telco_df.monthly_charges)
    #total_charges is a dtype object, but needs to be changed to a float64.
    df = df.astype({'total_charges': 'float64'})
    #dropping original fields and renaming fields.
    df.rename(columns = {'tenure': 'tenure_months'}, inplace = True)
    #created a new field to change tenure from months to year.
    df['tenure_year'] =  round(df['tenure_months']/12,0)
    
    return df


#######################################
def wrangle_telco():
    '''
    This function reads in telco_churn data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco_churn df from
    a csv file, returns df.
    '''
    if os.path.isfile('telco_churn.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = telco_churn_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    '''    
    Returns prepped train, validate,test dfs with total_charges a float64 dtype, dfs with Nan values replaced,
    tenure renamed to tenure_months, the column tenure_year added.
    '''
    # drop duplciates
    df.drop_duplicates(inplace =True)
    # creating df which has empty values both objects and nunmerics replaced with a nan value.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # replacing the NaN value in total_charges of the customers with 0 tenure with their monthly_charge.
    df.total_charges = df.total_charges.fillna(df.monthly_charges)
    #total_charges is a dtype object, but needs to be changed to a float64.
    df = df.astype({'total_charges': 'float64'})
    #dropping original fields and renaming fields.
    df.rename(columns = {'tenure': 'tenure_months'}, inplace = True)
    #created a new field to change tenure from months to year.
    
    return df


# Generic splitting function for continuous target.

def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test
##################################################


    


###################### Prep Telco Data ######################

def telco_split(df):
    '''
    This function take in the telco_churn data acquired by get_telco_data,
    performs a split and stratifies churn column.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, 
                                        stratify=df.churn)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, 
                                   stratify=train_validate.churn)
    return train, validate, test






