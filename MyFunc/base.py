# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

#----------------------------------------------------------------------------#
# > 설명 : print 함수를 사용할 때, 색상을 설정하는 함수
# > 예시 : print(f'{color.BOLD}{color.BLUE}text...{color.END}')
#----------------------------------------------------------------------------#
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
    
#----------------------------------------------------------------------------#
# > 설명 : mse, mae, mape, smape를 산출하는 함수
# > 상세
#    - mape, smape의 경우에는 예측률 확인을 위해서 0~100의 범위로 제한
#    - 아래의 산출식으로는 mape, smape > 0 이므로, min(100,res)는 제외해도 될 것으로 보임
#----------------------------------------------------------------------------#
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
    
#----------------------------------------------------------------------------#
# > 설명 : 양수의 값에 대해서, log변환 및 재변환하는 함수
# > 상세
#    - 이러한 변환을 하는 근거에 대해서 정확하게 확인을 하지 못함
#    - lower_bound는 양의 값에 대한 변환이므로 0
#    - upper_bound는 왜 100을 곱해주는지 잘 모르겠음
# > 예시
#     max_number = 10**2
#     x = np.array(sorted([-x for x in range(1,max_number)] + [0] + [x for x in range(1,max_number)]))
#     y = positive_transformation(x)[0]
#     plt.plot(x,y)
#----------------------------------------------------------------------------#
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

#----------------------------------------------------------------------------#
# > 설명 : dataframe의 head와 tail을 원하는 상황에 맞게 보여주는 함수
# > 상세
#    - n : head, tail의 데이터 수
#    - sort_var : sorting을 원하는 컬럼명 리스트
#    - select_var : 선택을 원하는 컬럼명 리스트
#    - check_last_data : 마지막 데이터를 확인할건지 여부
#----------------------------------------------------------------------------#
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
            
#----------------------------------------------------------------------------#
# > 설명 : R의 abline과 동일한 결과를 보여주는 함수
# > 상세
#    - intercept : y절편으로, 직선함수 y=a+bx의 a에 해당
#    - slope : 기울기로, 직선함수 y=a+bx의 b에 해당
#    - linewidth : 결과물인 직선의 두께 설정
#    - linestyle : 결과물인 직선의 선유형 설정
#    - color : 결과물인 직선의 색상 설정
#----------------------------------------------------------------------------#
def abline(intercept,slope,linewidth=2,linestyle='--',color='red'):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color)
    
#----------------------------------------------------------------------------#
# > 설명 : 원하는 타입의 컬럼들에 대해 onehot-encoding을 적용하는 함수
# > 상세
#    - df : dataframe
#    - cat_type : onehot-encoding을 적용 할 컬럼의 타입
#    - ignore_features : onehot-encoding이 적용되지 않도록 설정하는 컬럼
#----------------------------------------------------------------------------#
def onehot_encoding(df, cat_type, ignore_features=['']):
    data = df.copy()

    cols = setdiff(data.columns, ignore_features + ['target'])
    cols = df.select_dtypes(include=[cat_type]).columns

    res_df = pd.get_dummies(data, columns=cols)

    return(res_df)

#----------------------------------------------------------------------------#
# > 설명 : R의 setdiff와 동일한 결과를 보여주는 함수
#----------------------------------------------------------------------------#
def setdiff(x, y):
    return(list(set(x)-set(y)))

#----------------------------------------------------------------------------#
# > 설명 : 난수생성, 모델링 등에서 결과값의 변동이 없도록 하기위해 seed를 fix하는 함수
#----------------------------------------------------------------------------#
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