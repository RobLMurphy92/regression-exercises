import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from pydataset import data
import math
import stat
import seaborn as sns


#model classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt

from sklearn.metrics import explained_variance_score

##########################################
def plot_residuals(df,feature, y):
    #residual is y prediciton - y original (target)
    reg_model = LinearRegression().fit(df[[feature]], df[y])
    # compute predictions and add to original dataframe
    df['yhat'] = reg_model.predict(df[[feature]])
    df['yhat_baseline'] = df[y].mean()
    df['residual'] = df.yhat - df[y]
    #residual baseline is baseline_prediction - baseline actual.
    df['residual_baseline'] = df['yhat_baseline'] - df[y]
    df.residual.plot.hist()
    
    plt.show()
    df.residual_baseline.plot.hist()
    plt.show
    




def residual_errors(df,feature, target):
    #make baseline prediction 
    df['yhat_baseline'] = df[target].mean()
    # generate parameters, i.e. create model
    reg_model = LinearRegression().fit(df[[feature]], df[target])
    # compute predictions and add to original dataframe
    df['yhat'] = reg_model.predict(df[[feature]])
    #Calculating residual and residual baseline
    df['residual'] = df['yhat'] - df[target]
    df['residual_baseline'] = df['yhat_baseline'] - df[target]
    #Residual and baseline residual squared
    df['residual^2'] = df.residual ** 2
    df['residual_baseline^2'] = df.residual_baseline ** 2
    # calculating SSE and SSE baseline
    #SUM SQUARED ERROR
    SSE = mean_squared_error(df[target], df.yhat)*len(df)
    SSE_baseline = mean_squared_error(df[target], df.yhat_baseline)*len(df)
    #MEAN SQUARED ERROR
    MSE = mean_squared_error(df[target], df.yhat)
    MSE_baseline = mean_squared_error(df[target], df.yhat_baseline)
    #Root Mean Squared ERROR
    RMSE = sqrt(mean_squared_error(df[target], df.yhat))
    RMSE_baseline = sqrt(mean_squared_error(df[target], df.yhat_baseline))
    #######
    ESS = sum((df.yhat - df[target].mean())**2)
    ESS_baseline = sum((df.yhat_baseline - df[target].mean())**2)
    #######
    ###########
    TSS = ESS + SSE
    TSS_baseline = ESS_baseline + SSE_baseline
    ###########
    R2 = ESS/TSS
    R2_baseline = ESS_baseline / SSE_baseline
    #evaluation of model and baseline
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE','ESS','TSS','R2']), columns=['metric'])
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline', 'ESS_baseline','TSS_baseline','R2_baseline']), columns=['metric'])
    df_eval['model_error'] = np.array([SSE, MSE, RMSE, ESS, TSS, R2])

################################################
    
    print('---------Model---------------')
    print(df_eval)
    print("")
    print('-----------------------------')
    print("Percent of variance in y explained by x = ", round(R2*100,1), "%")
    print("")
    











def baseline_mean_errors(df,feature, target):
    df['yhat_baseline'] = df[target].mean()
    # generate parameters, i.e. create model
    reg_model = LinearRegression().fit(df[[feature]], df[target])
    # compute predictions and add to original dataframe
    df['yhat'] = reg_model.predict(df[[feature]])
    #Calculating residual and residual baseline
    df['residual'] = df.yhat - df[target]
    df['residual_baseline'] = df['yhat_baseline'] - df[target]
    #Residual and baseline residual squared
    df['residual^2'] = df.residual ** 2
    df['residual_baseline^2'] = df.residual_baseline ** 2
    
    
    
    # calculating baseline
    #SUM SQUARED ERROR
    SSE_baseline = mean_squared_error(df[target], df.yhat_baseline)*len(df)
    #MEAN SQUARED ERROR
    MSE_baseline = mean_squared_error(df[target], df.yhat_baseline)
    #Root Mean Squared Eroor
    RMSE_baseline = sqrt(mean_squared_error(df[target], df.yhat_baseline))
    
    #evaluation of model and baseline
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])
    df_baseline_eval['model_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
    
    print(df_baseline_eval)

############









def better_than_baseline(df, feature, target): 
    df['yhat_baseline'] = df[target].mean()
    # generate parameters, i.e. create model
    reg_model = LinearRegression().fit(df[[feature]], tips_df[target])
    # compute predictions and add to original dataframe
    df['yhat'] = reg_model.predict(df[[feature]])
    #Calculating residual and residual baseline
    df['residual'] = df.yhat - df[target]
    df['residual_baseline'] = df['yhat_baseline'] - df[target]
    #Residual and baseline residual squared
    df['residual^2'] = df.residual ** 2
    df['residual_baseline^2'] = df.residual_baseline ** 2
    
    RMSE = sqrt(mean_squared_error(df[target], df.yhat))
    RMSE_baseline = sqrt(mean_squared_error(df[target], df.yhat_baseline))
    #######

    if RMSE > RMSE_baseline:
        print(f'\n Model performs better than baseline, Model:{RMSE:.2f}, Baseline:{RMSE_baseline: .2f}')
    else:
        print(f'\n Model did not perform better than the baseline, Model:{RMSE: .2f}, Baseline:{RMSE_baseline: .2f}')