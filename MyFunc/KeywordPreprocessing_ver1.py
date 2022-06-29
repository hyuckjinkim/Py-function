# f-string과 carriage return을 같이 사용하기 위해서 만듦
def prt(x):
    sys.stdout.write(x)
    sys.stdout.flush()

#------------------------------------------------------------------------------------------------------------#
# (3-1) 키워드 오적재 케이스를 파악하는 함수
#------------------------------------------------------------------------------------------------------------#
def keyword_error_case(y):

    _case = [
        # (1) nvMid가 포함된 경우 : 값이 밀려들어가거나 해서 잘못 적재된 걸로 예상됨
        'case1' if x.find('nvMid')>=0 else
        
        # (2) \t 값으로 들어간 경우
        'case2' if (x=='\t') else
        
        # (3) dict 형태로 들어가있는 경우 (2,3번째는 키보드 자판으로 입력이 안됨(특수문자로 보임), 복붙해야 인식됨)
        'case3' if (x.find("': '")>=0) or (x.find("': F")>=0) or (x.find("': T")>=0) else
        
        # (4) 따옴표만 들어가있는 경우
        'case4' if (x=='""') or (x=='"') or (x=="''") or (x=="'") else
        
        # (5) ^가 들어간 경우
        'case5' if x.find('^')>=0 else
        
        # (6) `/`가 들어간 경우
        'case6' if x.find("`/`")>=0 else
        
        # (7) |가 들어간 경우 (/가 포함된 경우 제외 -> case10에서 가져옴)
        'case7' if (x.find('|')>=0) & (x.find('/')<0) else
        
        # (8) 숫자/가 들어간 경우
        'case8' if x.find('0/')>=0 or x.find('1/')>=0 or x.find('2/')>=0 or x.find('3/')>=0 or x.find('4/')>=0 or\
                   x.find('5/')>=0 or x.find('6/')>=0 or x.find('7/')>=0 or x.find('8/')>=0 or x.find('9/')>=0 else
        
        # (9) 숫자_가 들어간 경우
        'case9' if x.find('0_')>=0 or x.find('1_')>=0 or x.find('2_')>=0 or x.find('3_')>=0 or x.find('4_')>=0 or\
                   x.find('5_')>=0 or x.find('6_')>=0 or x.find('7_')>=0 or x.find('8_')>=0 or x.find('9_')>=0 else
        
        # (10) /가 들어간 경우
        'case10' if x.find('/')>=0 else
        
        # (11) 0.0, 1.0, ..., 10.0인 경우
        'case11' if (x=='0.0') or (x=='1.0') or (x=='2.0') or (x=='3.0') or (x=='4.0') or\
                    (x=='5.0') or (x=='6.0') or (x=='7.0') or (x=='8.0') or (x=='9.0') else
        
        # 정상으로 예상되는 것들
        'others'
        for x in y
    ]

    _res = pd.Series(_case).value_counts().sort_index()
    
    #------------------------------------------------------------------------------------------------------------#
    # case group과 table을 return
    #------------------------------------------------------------------------------------------------------------#
    # _case : case1 ~ case11(오적재 그룹), others(정상으로 예상되는 그룹)
    # _res  : 오적재 케이스 Freq.
    return np.array(_case),_res

#------------------------------------------------------------------------------------------------------------#
# (3-2) 키워드 전처리 케이스 파악하는 함수
#------------------------------------------------------------------------------------------------------------#
def check_1(df):

    # displays(chk,sort_var='key_len')

    print(f'기존 키워드 개수 : {df.keyword.astype(str).nunique():,}\n')

    quote_cnt = df[df["keyword"]!=df["keyword_new"]]["keyword_new"].nunique()
    print(f'따옴표 수정 필요한 키워드 건수 : {quote_cnt:,}')
    if quote_cnt>0:
        print(f'수정 전 키워드 건수 : {df.keyword.nunique():,}')
        print(f'수정 후 키워드 건수 : {df.keyword_new.nunique():,}')
        print(f'수정 전/후 차이 : {df.keyword.nunique() - df.keyword_new.nunique():,}')

        key = df[df['keyword']!=df['keyword_new']]['keyword_new'].unique()
        chk = df[df['keyword_new'].isin(key)]
        cnt_info = chk.keyword_new.value_counts().value_counts()

        print(f'  - 기존에 존재하던 키워드와 중복 O : {cnt_info[cnt_info.index==2].values[0]:,}')
        print(f'  - 기존에 존재하던 키워드와 중복 X : {cnt_info[cnt_info.index==1].values[0]:,}')
    
        check_count = cnt_info.sum() -\
            df[df["keyword"]!=df["keyword_new"]]["keyword_new"].nunique()
        check = '정상' if check_count==0 else '에러'

        print(f'  - 합계                      : {cnt_info.sum():,} (키워드 건수 확인 결과 : {check})')
    
#------------------------------------------------------------------------------------------------------------#
# (3-3) 전체 전처리 후 건수 변동 파악
#------------------------------------------------------------------------------------------------------------#
def check_2(df,keyword_df,final_df):
    
    print(f'> 건수 변동 : {df.shape[0]:,}건 -> {final_df.shape[0]:,}건 ({df.shape[0]-final_df.shape[0]:,}건 제거)')
    print(f'> 키워드 변동 : {df.keyword.nunique():,}건 -> {final_df.keyword.nunique():,}건', 
          f'({df.keyword.nunique()-final_df.keyword.nunique():,}건 제거)')

    key = keyword_df['keyword_tobe'][keyword_df["keyword_asis"]!=keyword_df["keyword_tobe"]]
    chk = keyword_df[keyword_df['keyword_tobe'].isin(key)]['keyword_tobe']

    print(f'> 따옴표 변동건 : {chk.value_counts().value_counts().sum():,}건',
          f'(기존 전체 {keyword_df.shape[0]:,}건)')
    
#------------------------------------------------------------------------------------------------------------#
# (3-4) 최종 키워드 전처리 함수 (건수 프린트 포함)
#------------------------------------------------------------------------------------------------------------#
# > Args :
#   - df : 전처리를 하기 이전의 데이터
#          함수 맨 앞부분에서 keyword의 unique를 들고와서 작업하기 때문에, 시간이 오래 걸리지 않음
#          (해당 키워드의 정상/비정상 여부를 확인하기 때문에, unique를 들고옴)
#          이후, 유의한 키워드만을 선택해서 최종 데이터셋을 return
#
# > Returns :
#  (1) keyword_df
#      - keyword_asis : 기존 키워드
#      - keyword_tobe : 따옴표 제거 후 키워드
#      - keyword_error_case : keyword_error_case에서 정의된 "확인 된" 오적재 케이스를 확인 (쿼리참조)
#      - keyword_special_char : 특수문자가 포함되었는지 여부
#      - keyword_only_number : 문자열이 모두 숫자인지 여부
#
#  (2) final_keyword_df : 전처리 후 키워드를 저장, 따옴표 포함된 키워드를 최종 데이터셋에 적용하기 위해 생성
#      - keyword_asis : 따옴표 포함된 키워드를 포함한 기존 키워드
#      - keyword_tobe : 따옴표를 수정한 키워드
#
#  (3) final_df : 전처리 모두 반영한 최종 데이터셋
#------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd

import datetime
import time
import os,sys
import re
import pytz


def KeywordPreprocessing(df):
    
    # 함수 시작/종료시간 display -> 한국시간으로 표시
    KST = pytz.timezone('Asia/Seoul')
    
    # 함수 시작시간
    start_time = str(datetime.datetime.now(KST))[:19]
    
    # 키워드를 unique로 가져와서 사용 (속도개선)
    d = pd.DataFrame({
        'keyword' : df.keyword.unique()
    })
    
    #------------------------------------------------------------------------------------------------------------#
    # (1) None 제거
    #------------------------------------------------------------------------------------------------------------#
    print('-'*100)
    print('> (1/9) None 제거 - 건수 :', d[d.keyword.astype(str)=='None'].shape[0])
    d = d[d.keyword.astype(str)!='None']
    
    #------------------------------------------------------------------------------------------------------------#
    # (2) Null 제거
    #------------------------------------------------------------------------------------------------------------#
    print('> (2/9) Null 제거 - 건수 :', d[d.keyword==''].shape[0])
    d = d[d.keyword!='']
    
    #------------------------------------------------------------------------------------------------------------#
    # (3) 따옴표 제거 (정상인 건에서, 따옴표가 붙은 건 제거)
    #------------------------------------------------------------------------------------------------------------#
    keyword_list = d.keyword.unique()

    # 따옴표 제거 룰 : 첫번째와 마지막이 따옴표인 경우 제거
    # (why?) 쌍따옴표가 있는 키워드는 의미없는 키워드라고 생각하고, 바로 쌍따옴표를 제거해버리면 되지만,
    #        추후에는 문제가 될 수도 있을거라 판단되기 때문에, 보수적으로 따옴표를 제거하도록 하였음.
    key_new = [k.replace('"','') if (k.find('"')==0) & (k.find('"',1)==len(k)-1) else
               k.replace("'",'') if (k.find("'")==0) & (k.find("'",1)==len(k)-1) else
               k for k in keyword_list]
    key_new = np.array(key_new)

    # 따옴표 제거 전 -> 제거 후로 수정하는 함수
    # 속도를 높이기 위하여, 데이터를 키워드 별로 쪼개서 작업
    def keyword_fix(keyword_df,old,new):

        k_df = keyword_df.copy()
        key = k_df.keyword.unique()
        
        # 키워드가 2개 이상인 경우 에러발생
        if k_df.keyword.nunique()>1:
            raise('Keywords must be unique')

        # 해당 키워드 데이터에 수정해야하는 키워드가 있는 경우, 키워드를 수정
        # 아닌 경우, 기존의 키워드 저장
        if np.where(old==key,1,0).sum() > 0:
            k_df['keyword_new'] = new[old==key][0]
        else:
            k_df['keyword_new'] = key[0]

        return k_df

    old = keyword_list[keyword_list!=key_new] # 수정해야하는 대상의 수정 전 키워드
    new = key_new[keyword_list!=key_new]      # 수정해야하는 대상의 수정 후 키워드

    # 키워드 별로 따옴표 제거한 변수 추가
    start_time_x = time.time()
    new_df = []
    iter = 0
    total = d.keyword.nunique()
    for key in d.keyword.unique():
        iter += 1
        prt(f'\r> (3/9) 따옴표 제거 : {iter:,} / {total:,} ({iter/total*100:.1f}%)    ')
        
        sub = d[d.keyword==key]
        sub = keyword_fix(sub,old,new)
        new_df.append(sub)
        
    new_df = pd.concat(new_df,axis=0)
    end_time_x = time.time()
    run_time_x = (end_time_x-start_time_x)/60
    prt(f'\r> (3/9) 따옴표 제거 : {iter:,} / {total:,} ({iter/total*100:.1f}%), Runtime : {run_time_x:.2f}Min')

    #------------------------------------------------------------------------------------------------------------#
    # (4) 예측된 오적재 케이스 확인
    #------------------------------------------------------------------------------------------------------------#
    print(f'\n> (4/9) 예측된 오적재 케이스 확인 (케이스 상세설명은 쿼리 참조)')
    case,res = keyword_error_case(new_df['keyword_new'])
    new_df['keyword_error_case'] = case
    print('-'*100)
    display(res)
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (5) 특수문자, only number 제거
    #------------------------------------------------------------------------------------------------------------#
    # > 수정
    #   - (2022-04-17) : 속도향상을 위해, (4)의 오적재케이스 제외한 키워드로 확인
    loc = new_df['keyword_error_case']=='others'
    keyword_list = new_df['keyword_new'][loc].unique()

    start_time_x = time.time()
    keyword_df = []
    iter = 0
    total = len(keyword_list)
    for k in keyword_list:
        iter += 1
        prt(f'\r> (5/9) 특수문자 및 only number 제거 : {iter:,} / {total:,} ({iter/total*100:.1f}%)     ')
        
        sub = new_df[new_df['keyword_new']==k]

        # [1] 특수문자 제거
        k_1 = re.sub('[^ㄱ-ㅎㅣ^가-힣+|^a-zA-Z0-9|^ |^-|^.|^%|^MLT-|^CLT-]','',k) # 정상 케이스 빼고 제거
        k_1 = re.sub('\^','',k_1)               # ^ 제외
        k_1 = re.sub('\|','',k_1)               # | 제외
        k_1 = re.sub('[ㄱ-ㅎ|ㅏ-ㅣ]','',k_1)      # 자음만 있는 경우 제외

        # [2] 숫자 + 대쉬(-,_)만 포함된 경우
        k_2 = re.sub(r'[0-9|-|_]','',k)

        # 데이터에 특수문자포함여부 / only number여부 저장
        sub['keyword_special_char'] = 1 if (k_1!=k)  else 0
        sub['keyword_only_number']  = 1 if (k_2=='') else 0

        keyword_df.append(sub)

    keyword_df = pd.concat(keyword_df,axis=0)
    end_time_x = time.time()
    run_time_x = (end_time_x-start_time_x)/60
    prt(f'\r> (5/9) 특수문자 및 only number 제거 : {iter:,} / {total:,} ({iter/total*100:.1f}%), Runtime : {run_time_x:.2f}Min')
    
    #------------------------------------------------------------------------------------------------------------#
    # (6) 최종 오적재 케이스 별 건수 파악
    #------------------------------------------------------------------------------------------------------------#
    print(f'\n> (6/9) 최종 오적재 케이스 별 건수 파악')
    print('-'*100)
    check_1(df,keyword_df)
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (7) 최종 키워드 전처리 (이슈있는 키워드 제거)
    #------------------------------------------------------------------------------------------------------------#
    print(f'> (7/9) 최종 키워드 전처리')
    final_keyword_df = keyword_df[
        (keyword_df['keyword_error_case'] == 'others') &\
        (keyword_df['keyword_special_char']==0) &\
        (keyword_df['keyword_only_number']==0)
    ]\
        [['keyword','keyword_new']].\
        reset_index(drop=True).\
        rename(columns={'keyword'    :'keyword_asis',
                        'keyword_new':'keyword_tobe'})
    
    #------------------------------------------------------------------------------------------------------------#
    # (8) 전처리 후, 최종 데이터 셋 생성
    #------------------------------------------------------------------------------------------------------------#
    df2 = df[df.keyword.isin(final_keyword_df['keyword_asis'])]

    start_time_x = time.time()
    final_df = []
    iter = 0
    total = df2.keyword.nunique()
    for k in df2.keyword.unique():
        iter += 1
        prt(f'\r> (8/9) 전처리 후, 최종 데이터셋 생성 : {iter:,} / {total:,} ({iter/total*100:.1f}%)  ')
        
        sub_df = df2[df2.keyword==k]
        key_df = final_keyword_df[final_keyword_df['keyword_asis']==k]

        if list(key_df['keyword_asis'].values != key_df['keyword_tobe'].values)[0]:
            sub_df['keyword'] = key_df['keyword_tobe'].values[0]
        
        final_df.append(sub_df)

    final_df = pd.concat(final_df,axis=0)
    end_time_x = time.time()
    run_time_x = (end_time_x-start_time_x)/60
    prt(f'\r> (8/9) 전처리 후, 최종 데이터셋 생성 : {iter:,} / {total:,} ({iter/total*100:.1f}%), Runtime : {run_time_x:.2f}Min')
    
    #------------------------------------------------------------------------------------------------------------#
    # (9) 전체 전처리 후, 건수 변동 파악
    #------------------------------------------------------------------------------------------------------------#
    print(f'\n> (9/9) 전체 전처리 후, 건수 변동 파악')
    print('-'*100)
    check_2(df,final_keyword_df,final_df)
    print('-'*100)
    
    res = (
        keyword_df.rename(columns={'keyword':'keyword_asis','keyword_new':'keyword_tobe'}).reset_index(drop=True),
        final_keyword_df.reset_index(drop=True),
        final_df.reset_index(drop=True),
    )
    
    # 함수 종료시간
    end_time = str(datetime.datetime.now(KST))[:19]
    
    fmt = '%Y-%m-%d %H:%M:%S'
    run_time = datetime.datetime.strptime(end_time,fmt) - datetime.datetime.strptime(start_time,fmt)
    
    print('')
    print('-'*100)
    print(f'> start time : {start_time}')
    print(f'>   end time : {end_time}')
    print(f'>   run time : {run_time}')
    print('-'*100)

    return res

# # Example
# keyword_df, final_keyword_df, final_df = KeywordPreprocessing(df)