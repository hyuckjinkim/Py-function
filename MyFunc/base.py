import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def mse(true, pred):
    return np.mean((pred-true)**2)

def mae(true, pred):
    return np.mean(np.abs(pred-true))

def mape(true, pred):
    res = np.mean(np.abs(pred-true)/true)*100
    res = min(100,res)
    res = max(0,res)
    if (np.isnan(res)) or (np.isinf(res)):
        return 100
    else:
        return res
    
def smape(true, pred):
    res = np.mean(np.abs(pred-true) / (np.abs(pred)+np.abs(true))) * 200
    res = min(100,res)
    res = max(0,res)
    if (np.isnan(res)) or (np.isinf(res)):
        return 100
    else:
        return res
    
def positive_transformation(x,bound=None):
    
    if bound is None:
        adj = -0.1
        lower_bound = 0 + adj
        upper_bound = max(x)*100 - adj
    else:
        lower_bound = bound[0]
        upper_bound = bound[1]
    
    transformed = np.log((x-lower_bound)/(upper_bound-x))
    
    return transformed, lower_bound, upper_bound
    
def inverse_positive_transformation(x, lower_bound, upper_bound):
    upper_bound = upper_bound
    lower_bound = lower_bound
    
    transformed = (upper_bound-lower_bound)*np.exp(x)/(1+np.exp(x))+lower_bound
    
    return transformed

def displays(data,n=5,sort_var=None,select_var=None,check_last_data=False):
    d = data.copy()
    
    if sort_var is not None:
        d = d.sort_values(sort_var)
        
    if select_var is None:
        select_var = d.columns
    
    print(f'> head({n})')
    display(d[select_var].head())
    if (check_last_data) & (1<=d.shape[0]<=n):
        display(np.array(d[select_var].tail().iloc[-1,:]))
        
    print('')
    
    if d.shape[0]>n:
        
        print(f'> tail({n})')
        display(d[select_var].tail())
        if check_last_data:
            display(np.array(d[select_var].tail().iloc[-1,:]))
            
def abline(intercept,slope,linewidth=2,linestyle='--',color='red'):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color)
    
def onehot_encoding(df, cat_type, ignore_features=['']):
    data = df.copy()

    cols = setdiff(data.columns, ignore_features + ['target'])
    cols = df.select_dtypes(include=[cat_type]).columns

    res_df = pd.get_dummies(data, columns=cols)

    return(res_df)

def setdiff(x, y):
    return(list(set(x)-set(y)))

def seed_everything(seed: int=0):
    # (참조) https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    # for later versions: 
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)