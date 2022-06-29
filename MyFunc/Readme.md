# My UDFs (User Defined Functions)
- last updated : 2022.06.29

<br>

## 1. base.py
- Python에서 편의성을 위하여 사용하는 기본적인 함수들

<br>

## 2. convert_ipynb_to_py.ipynb
- ipynb 파일을 py 파일로 변환하는 쿼리

<br>

## 3. extract.py
- AWS Athena의 데이터 추출 쿼리

<br>

## 4. KeywordPreprocessing
- 데이터 내의 "keyword" 컬럼의 특수문자가 포함되거나,숫자로만 이루어지거나 등의 비정상적인 값들을 제외하는 쿼리
- KeywordPreprocessing_ver1.py는 이전 버전이고, KeywordPreprocessing_ver2.py는 최신 버전

<br>

## 5. MyModel.py
- 기본적인 모델링을 위해 만든 쿼리
- AutoML이 포함되어 있음

<br>

## 6. MyPickle.py
- pickle 형태로 저장하는 작업을 편하게 하기 위해 작성한 쿼리
- pandas의 to_csv와 비슷한 형태로 구성
- class의 형태로 업데이트 필요

<br>

## 7. runtime.py
- 쿼리의 실행시간을 보기위해 작성된 쿼리
- class의 형태로 작성되어 있지만, 고도화가 필요
