이번 chapter에서는 우리가 최근에 부동산 중개업 회사에 고용된 데이터 과학자인 마냥 종단간(end to end) 프로젝트를 수행해볼 것이다. 다음은 당신이 수행해야할 주된 단계들을 정렬해놓은 것이다.
 - 큰그림 보기
 - 데이터 가져오기
 - 데이터에 대한 이해를 얻기 위한 데이터 시각화 및 탐구
 - 학습모델 선정 및 학습시키기
 - 학습 모델 조율하기
 - 우리의 솔루션 선보이기
 - 우리의 시스템을 실행하고. 감시하며 유지하기.

# 실제 데이터로 해보기
기계학습을 배울 때에는 인공적으로 만든 데이터뿐만 아니라 실제 세계의 데이터세트를 이용헤 경험해보는 것이 좋다. 다행히도 데이터세트를 공개해놓은 수많은 사이트들이 존재한다. 다음은 데이터 세트를 수집하는데 도움이 될만한 몇몇 장소들이다.
 - 인기있는 공개 데이터 레포지토리
   - [미국 어바인 기계학습 레포지토리](http://archive.ics.uci.edu/ml/)
   - [Kaggle 데이터세트](https://www.kaggle.com/datasets)
   - [아마존 웹서비스(AWS) 데이터세트](http://aws.amazon.com/fr/datasets)
 - 메타 포털 사이트 (이 사이트들은 공개 데이터 세트 레포지토리들을 리스트화 하였음)
   - http://dataportals.org/
   - http://opendatamonitor.eu/
   - http://quandl.com/
 - 인기가 많은 공개 데이터 레포지토리들을 리스트한 다른 페이지들
   - [위키피디아의 기계학습 데이터 세트의 리스트](https://goo.gl/SJHN2k)
   - [Quora.com 질문](http://goo.gl/zDR78y)
   - [데이터 세트 서브레딧](https://www.reddit.com/r/datasets)

이 Chapter에서 우리는 StatLib 레포지토리의 캘리포니아 집 값 데이터 세트를 사용할 것이다. 이 데이터세트는 1990년 캘리포니아 인구조사로 나온 데이터에 기반을 한 것이다. 최근의 데이터는 아니지만, 학습에 쓰기에 질이 좋은 데이터 세트이기에 최근인 데이터 세트처럼 사용해보자. 교육 목적으로 요소(attribute)들을 좀 더 추가하였고, 몇가지 특징을 제거하였다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-1.png)

# 큰 그림을 보자
기계학습 하우징 회사에 오신 것을 환영합니다! 당신이 처음에 해주어야할 일은 캘리포니아 인구조사 데이터 세트를 활용하여 캘리포니아 집 값에 대한 모델을 구축해야합니다. 이 데이터는 캘리포니아에서 각각의 블럭그룹(Block Group)에 대하여 블럭 그룹 지역의 인구 밀도, 중간 값의 수입, 중간 값의 집 값등으로 이루어진 행렬로 되어있다. 블럭그룹이란 미국통계청에서 샘플 데이터로 지정한 최소 지리학적 단위이다.(전형적으로 한 블럭그룹당 600~3000명 정도의 인구밀도정도를 표현해준다.) 이제 이를 그냥 짧게 "구역"이라고 하겠다.

이 데이터 세트를 사용해 모델을 학습시킬 것이고, 어떠한 구역의 중간 값의 집 값을 예측할 수 있도록 할 것이다. 

## 문제에 대한 틀을 잡자
사장님에게 처음으로 물어볼 질문은 정확히 이 사업이 목적이 무엇인가이다. 아마 모델 구축이 최종 목표는 아닐 것이다. 회사는 이 학습 모델로 어떻게 사용하기를 원하고 어떻게 이익을 창출할지 원할까? 이는 중요한 사항인데, 이걸로 문제를 어떻게 정의할 것인지, 어떤 학습 모델을 사용할 것인지, 우리의 학습 모델을 평가하는데 어떤 수행 측정치를 사용할 것인지, 얼마나 노력을 들여서 학습시킬 지를 정하기 때문이다. 

그래서 사장님은 우리의 학습모델의 결과(중간 집 값 예측)가 다른 수 많은 신호들에 따라서 또 다른 기계학습 시스템에서도 잘 될 수 있는 것을 원한다고 답했다. 이러한 후속 시스템은 주어진 지역에 투자할 가치가 있는지 없는지 정할 것이다. 이는 직접적으로 수익에 영향을 끼치기에 결과가 올바르게 나와야하는 것이 아주 중요하다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-2.png)

다음으로 물어보아야할 짐문은 현재의 솔루션이 어떤지에 대해서이다. 이는 문제를 푸는 방법에 대한 통찰력 만큼이나 참고할 만한 행동이다. 사장님은 구역 집 값이 현재 주로 전문가들에게 평가받는다고 답할 것이다. 한 팀이 구역에 대한 최신 정보를 모으는데, 중간 집 값을 수집할 수 없을 떄에는 복잡한 규칙에 의거해 가격을 책정한다. 이는 시간도 많이 걸릴뿐더러 그 책정치도 완벽한 것이 아니다. 실제로 책정치가 10%정도 저렴하게 측정하기에 이런 회사들은 주어진 구역에 대한 구역의 중간 값을 예측 하는 모델을 학습시키기를 원하는 것이다. 인구조사 데이터 세트는 수천개의 구역의 중간 집 값 정보를 포함하고 있기 때문에 이러한 목적에 활용하기 딱 좋은 데이터 세트이다. 

좋다. 이제 이 정보들로 시스템을 구상을 시작할 준비가 되었디. 먼저 문제에 대한 틀을 잡아라. 다음으로 넘어가기 전에 어떤 방식으로 문제를 해결할지 한번 생각해보자. 지도학습, 비지도 학습, 강화학습, 실시간 학습, 비실시간 학습? 어떤 것을 활용할 것인가?

생각해보았는가? 그럼 이제 시작해보자. 일단 우리는 레이블된 학습데이터를 받았기 때문에 지도학습을 써야함은 분명하다. 더 나아가, 어떤 값을 예측해야하는 것이므로 회귀를 써야할 것이다. 좀 더 세부적으로, 시스템이 여러 특징들을 사용해서 예측할 것이기 때문에, 다변수 회귀(Multivariate regression) 문제이다(어떤 구역에 대한 인구 밀집도, 중간 소득 등). 이전 Chapter에서 삶의 만족도를 그 나라의 GDP당 소득 값으로 예측했는데 이를 단일변수 회귀(univariate regression) 문제라고 한다. 마지막으로 데이터가 연속적이지 않고, 빠르게 데이터를 바꾸어줄 필요도 없기 때문에 비실시간 학습(batch learning)이 이 프로젝트에 아주 적합할 것이다!
```
데이터가 너무 크면, (우리가 나중에 볼 MapReduce 기법을 활용해서) 다중 서버를 사용해 비실시간 학습을 나누어 실행할 수도 있다.
```
## 사용할 수행 측정치를 정하자
다음 단계로는 수행을 측정할 수 있는 방법을 고르는 것이다. 회귀 문제에 대하여 전형적인 수행 측정도는 Root Mean Square Error(RMSE:평균 제곱근 오차) 기법이다. 이는 예측을 수행할 때 얼마나 많은 에러가 시스템에 일반적으로 만들어지는지를 알 수 있는 아주 좋은 아이디어이다. RMSE를 연산하는 수학적 공식은 다음과 같다.
###### Equation 2-1. Root Mean Square Error(RMSE:평균 제곱근 오차)
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/Eq2-1.png)
* 표기법

  이 공식은 우리가 이 책을 진행하면서 몇몇 기본적인 기계학습 표기법을 소개한다.
  * *m* 은 RMSE로 측정하려는 데이터 셋 상의 예시의 개수
    - 예를들어, 만약 2000개 구역의 검증 데이터 세트로 평가하려고 한다면 *m* = 2000이다.
  * *x^(i)* 는 데이터 셋에서 *i*번째에 대한 (레이블 제외) 모든 특징 값들에 대한 벡터 값이다. *y^(i)* 는 이에 대한 레이블(예시에 대한 바람직한 값) 값이다.
    - 예를 들어, 데이터 세트의 첫번째의 경도가 -118.29도, 위도가 33.91도이며, 1416명의 거주자가 사는 동네의 중간의 소득값이 $38,372이며 이 지역의 중간 집 값이 $156,400이라면 다음과 같이 표기될 것이다.

      ![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/Notation1-1.png)    ![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/Notation1-2.png)

  * *X* 는 (레이블 제외) 데이터 세트에서 모든 경우에 대한 모든 특징의 값을 가지고 있는 행렬이다. 한 열당 예시 하나가 들어가 있으며 i번째 열은 
 *X^(i)* 를 전치한 전치행렬이다. 
    - 예를 들어, 첫번째 구역이 위에서 묘사했던 것처럼 되어있다면, 행렬 *X* 는 다음과 같을 것이다.
      ![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/Notation1-3.png)

  * *h* 는 시스템의 예측함수이며, 추측(hypothesis)라고 한다. 시스템이 어떤 경우에 대한 특징 벡터 *x^(i)* 가 주어지면, 그 경우에 대한 예측값 *y-hat^(1) = h(x^(i))* 를 출력한다.
    - 예를 들어, 

  * *RMSE(X,h)* 는 우리의 추측값 *h* 를 사용해서 예시 세트에 대해 측정된 손실 함수이다.


RMSE가 일반적으로 회귀 문제에서 많이 쓰이는 수행 측정 지표일지라도, 다른 상황에서는 다른 종류의 함수를 쓰는 것이 더 좋을 수도 있다. 예를들어, 수많은 이상점인 구역들이 있다. 이러한 경우는, Mean Absolute Error라는 함수의 사용을 고려해볼지모른다. (Average Absolute Devidation, 절대평균편차라고도 한다)
###### Equation 2-2. 절대 평균 편차
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/Eq2-2.png)

RMSE, MAE 둘 다 두 벡터, 예측값 벡터와 레이블값 벡터 사이의 거리를 측정하는 것이다. 그리고 측정법에는 여러가지 측정법과 Norm이 존재한다.

## 가정(Assumption)확인하기 
마지막으로 여태 만들었던 가정을 리스트화하고 확인하는 것은 좋은 연습이된다. 이는 초기에 심각한 이슈를 잡을 수도 있다. 예를 들면, 시스템이 출력한 어떤 구역의 가격이 후속 기계학습 시스템에 입력될 것이고, 우리는 이러한 가격이 앞으로 쓰여질 것을 가정할 수 있다. 하지만 만약 후속 시스템이 실제로 입력받는 카테고리를 그냥 "저렴", "평범", "비쌈"으로 가격을 변환하여, 가격 그 자체 값을 대신해서 사용된다면 어떨까? 이런 경우에는 가격의 값을 정확하게 맞도록 가져오는것은 전혀 중요하지 않다. 시스템은 그냥 올바른 카테고리를 받기를 원할 것이다. 그렇다면, 문제는 분류로 프레임이 맞추어져야한다. 우리들은 몇달동안 한 회귀문제에서 작동하는 것을 대신에 새롭게 솔루션을 다시 찾는 것을 원하지 않는다.

다행히도 후속 시스템 팀이 카테고리가 아닌 실제 가격 값이 필요하다고 한다! 좋다! 모든 준비가 끝났다. 이제부터 코딩을 해보자.


# 데이터 수집하기
이제 손을 더럽힐 시간이다! 랩탑을 들고 Jupyter notebook으로 예시 코드를 실행시켜보는 것을 두려워하지말자. 

## 작업공간 만들기
일단 파이썬이 설치가 되어 있어야한다. 설치가 되어있지 않다면 미리 설치를 해두자. 이제 기계학습 코드와 데이터셋을 위한 작업공간용 디렉토리를 만들자. 터미널을 열고 다음의 명령어를 입력해보자
```
$export ML_PATH="$HOME/ml" 꼭 디렉토리 이름이 ml일 필요는 없다.
$mkdir -p $ML_PATH
```
그리고 Jupyter, NumPy, Pandas, Matplotlib, scikit-learn이라고  하는 4개의 파이썬 라이브러리가 필요하다. 이미 이 모듈들을 호출하여 jupyter notebook을 실행시켜 보았다면 "데이터 다운로드" 섹션을 생략해도 좋다.

또한 프로젝트를 수행하면서 필요한 python 라이브러리들을 손쉽게 깔기위해서 Anaconda 혹은 pip라고 불리는 Python 패키징 시스템이 필요하다 만약 설치되어 있지 않다면 이 레포지토리 메인 페이지에 있는 [설치 섹션](https://github.com/Hahnnz/handson_ml-Kor)을 참고하라.
만약 설치된 pip 버전을 알고싶다면 다음 명령어를 입력하라
```
$ pip3 --version
pip 9.0.1 from [...]//lib/python3.5/site-packages (python 3.5)
```
가급적이면 최신의 pip 버전을 유지하고 pip 버전을 업그레이드 하고 싶다면 다음의 명령어를 입력하라
```
pip3 install --upgrade pip
```

```
독립적인 가상환경 만들기

만약 작업을 할 가상환경을 만들고 싶다면 다음의 pip 명령어로 virtualenv를 설치하자
    $ pip3 install --user --upgrade virtualenv
그럼 이제 아래의 명령어를 입력하여 독립적인 파이썬 환경을 만들수 있다.
    $ cd $ML_PATH
    $ virtualenv env
이제 매번 이 환경을 실행시킬때마다 아래의 명령어만 쳐주면 된다.
    $ cd $ML_PATH
    $ source env/bin/activate
이 환경이 실행되는 동안에 입력되는 pip 명령어에 의한 설치파일들은 모두 이 환경안에 설치가 되게 된다.
```

이제 다음의 명령어를 입력하여 필요한 모듈과 의존성 파일들을 설치하자.
```
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
```

설치가 잘 되었는지 확인하기 위해서는
```
$ python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
```

여기까지 문제가 없었다면 다음의 명령어로 Jupyter notebook을 실행시킬 수 있다
```
$ jupyter notebook
```

그렇다면 이제 포트 8888로 Jupyter 서버가 실행이 된 것이다! 이제 브라우저에 새 창이 뜨고 아까 만들어두었던 환경 파일 env가 보일 것이다. 이제 Python Notebook을 만들어보자. 우측 상단의 `New`라는 버튼을 클릭해서 Python3를 선택하면 새롭게 파일을 생성할 수 있다. (아래 그림 참고)
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-3.png)

Notebook은 Cell이라고하는 박스의 리스트이다. 각 Cell에는 실행가능한 코드나 일반 평서문등을 넣을 수 있다.
Cell 1번에 `print("hello world")`를 입력하고 실행해보아라. 우측 중앙에 플레이 버튼처럼 생긴 버튼을 누르면 선택한 Cell의 코드를 실행시킬 수 있다. 아니면 Shift + Enter로도 실행이 가능하다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-4.png)

Jupyter Notebook을 좀 더 체험해보고 싶다면 Help 메뉴에 있는 `User interface tour`를 사용해보자

## 데이터 다운로드
전형적인 환경에서, 우리의 데이터는 관계형 데이터베이스로 사용이 가능하며, 다중 테이블/문서/파일들로 널리 퍼져있다. 이를 사용하기 위해서는 먼저 사용 허가를 받아야하며, 데이터 스키마에 우리 자신을 친근하게 해야한다. 하지만 이 프로젝트에서는 생각보다 간단하다. 그냥 `housing.tgz`로 압축된 단일파일을 다운로드 받으면 되는데, 이는 `housing.csv`라는 컴마","로 값들이 나누어진 파일 (CSV:comma-separated value)을 다운로드하면 된다. 

데이터 다운로드를 위해 웹브라우저를 사용하여 다운로드를 받고, tgz파일을 실행해 압축을 풀어 CSV파일을 추출할 수 있다. 하지만 이를 수행하기 위한 함수를 작성해야한다. 만약 데이터가 규칙적으로 변한다면 데이터 다운로드 함수를 따로 작성해두면 필요할 때 마다 빠르게 실행할 수 있다는 장점이 있다. 혹은 이렇게 만들어둔 함수를 스케줄링하여 기간을 두어 자동으로 다운로드하게 할 수도 있을 것이다. 아래에 이에 대한 함수 코드가 있다. 

```
# 필요한 모듈 호출
import os
import tarfile
from six.moves import urllib

#다운로드 경로 설정
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# 하우징 데이터 세트를 설정된 경로에서 다운로드 할 수 있도록 해주는 함수
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```
우리는 이제 이 함수를 `fetch_housing_data()`부를 것이며, 이는 당신의 작업공간 디렉토리안에 `dataset/housing`이라는 디렉토리 안에 `housing.tgz`와 `housing.csv`를 다운로드 할 것이다.

자, 이제 pandas를 이용해서 데이터를 호출해보자. 다시 한번 더 데이터 호출 함수를 작성해보자.
```
# 필요한 모듈 호출
import pandas as pd

# 하우징 데이터 세트 불러오는 함수
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```
이 함수는 모든 데이터를 담아서 Pandas Dataframe object를 반환할 것이다.

## 데이터 구조 빠르게 훑어보기
DataFrame의 `head()`함수를 사용하여 한번 최상위 5개 열을 조회해보자.
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-5.png)

각각의 열은 하나의 구역을 나타낸다. 여기서 총 10개의 요소(Attribute)를 볼 수 있다.

`info()`함수는 데이터에 대한 빠른 설명을 얻는데 도움이 된다. 일반적으로, 열의 수, 각 attribute의 자료형, 값이 비어있지 않은 것의 개수를 표시해준다.
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-6.png)

데이터 셋에는 20640개의 예시가 존재하며, 기계학습 표준으로는 턱없이 부족하나, 초보에게 시작하기에는 딱 좋다. `total_bedrooms` 요소는 비어있지 않은 곳이 20433개이며 이는 207개의 구역에서 특징이 없다는 것을 의미한다. 이는 후에 다룰 필요가 있다.

`Ocean_proximity`를 제외한 모든 요소들은 숫자로 이루어져있다. `Ocean_proximity`의 자료형은 Object이며 Python Object의 어떤 종류에도 잘 데이터를 읽어올 수 있다. 하지만 이 데이터는 문자로 되어있다. 앞에서 최상위 5개 열을 보았을 때, `Ocean_proximity`가 있던 것을 보았을 것이다. 이는 `Ocean_proximity`가 카테고리 요소이며 우리는 얼마나 많은 구역이 이 카테고리에 걸려 있는지 보기위해서 `value_counts()`함수를 사용할 수 있다.
```
# "ocean_proximity"의 카테고리당 갯수 출력
housing["ocean_proximity"].value_counts()

<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
```
한 번 다른 것도 보자. `describe()` 함수가 이를 출력해서 보여줄 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-7.png)

`mean`, `min`, `max`는 따로 설명이 필요 없을 것이다. 그리고 Null 데이터는 무시될 수 있음을 알아두어라. `std`열은 표준편차(standard deviation)를 보여주며, 이는 값들이 얼마나 퍼져있는지를 표시해준 것이다. 25%, 50%, 75%는 백분율(percentiles)를 나타낸 것으로, 25%의 구역은 `Housing Median Age`가 18 이하이고, 50%는 29이하, 75%는 37이하이다. 이는 종종 25th percentile(1st quartile), 중간, 75th percentile(3st quartile)이라도 불린다. 

우리가 다루는 데이터의 타입에 대해 느낌을 받는 또 다른 방법은 각각의 숫자 요소들에 대해서 히스토그램을 plot하는 것이다. 히스토그램은 주어진 값의 법위(수평축)에 대한 예시의 수(수직축)를 나타낸다. 아니면 요소 하나당 점 하나로 찍는 방법도 있으며, 혹은 전체 데이터에 대해 `hist()`함수를 불러오는 것이다. 이는 각 숫자 요소들에 대해 히스토그램은 만들 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-8.png) 

예를 들어, 약 800개 이상의 지역이 `median_house_value`로 약 $100,000이라는 것을 볼 수 있다.
```
# Notebook안에 그림 출력
%matplotlib inline 
import matplotlib.pyplot as plt
#히스토그램 작성
housing.hist(bins=50, figsize=(20,15))
#그림 저장
save_fig("attribute_histogram_plots")
#그림 보기
plt.show()
```
```
hist()라는 함수는 matplotlib이라는 라이브러리에 속한 함수이기에 사용하고자 한다면 꼭 이 라이브러리를 먼저
호출을 해두어야 한다. show()라는 함수는 jupyter내에서 제공하는 함수이다.
```
위의 히스토그램에서 다음과 같은 점들을 알 수 있다.
1. 중간 소득 요소는 미국 달러로 표현된 것이 아니다. 데이터를 수집한 팀이 이를 확인한 후, 데이터가 상위 소득을(원래는 15.0001)15로, 하위 소득을(원래는 0.4999)5로 재조정 했다고 말해줄 것이다. 전처리된 요소로 작업하는 것은 기계학습에서는 아주 흔한 일이지만 아주 중요한 문제는 아니다. 그래도 어떻게 계산이 되는지 알 필요는 있다.
2. 중간 집 연식과 중간 집 값(value)가 완전히 포개어진다. 

## 실험 데이터 세트 만들기
이 단계에서 데이터의 일부분을 따로 빼놓는다는 소리는 이상하게 들릴지 모른다. 결국엔 우리는 데이터를 빠르게 한번만 보고 당연히 우리는 어떤 알고리즘을 사용할 지 정하기 전에 데이터에 대해 전체 그 이상을 알아야한다. 
그러나 우리의 뇌는 놀라운 패턴 감지 시스템을 가지고 있는데 이는 강하게 오버피팅을 일으키기 쉽다. 만약에 우리가 실험 데이터를 보게 된다면, 우리는 실험 데이터에서 겉보기에 재밌어 보이는 패턴을 우연이 만날지 모른다. 이는 특정 종류의 기계학습 모델을 고르도록 할 것이고 우리가 실험 데이터 세트를 사용해서 일반화 에러를 측정할 때, 우리의 측정치는 너무 낙관적일 것이고 우리는 기대했던 것 만큼 수행을 제대로 하지 못하는 모델을 출시할 것이다. 즉 다시 말하면 우리의 가설을 증명하기위해 이 가설이 좋다고 증명할 수 있는 데이터만을 뽑아와서 증명하는 것인데 이를 데이터 도용 편향(data snooping bias)라고 한다.

실험 데이터 세트를 만드는 것은 이론적으로 꽤 간단하다 : 그냥 데이터 세트의 20% 정도를 무작위로 골라서 따로 빼놓는 것이다.
```
import numpy as np

# 설명을 위해서 만든 것이다. Sklearn 자체에 train_test_split()함수를 가지고 있다.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```
그리고 이 함수는 다음과 같이 사용할 수 있다.
```
>>> train_set, test_set = split_train_test(housing, 0.2)
>>> print(len(train_set), "train +", len(test_set), "test")
16512 train + 4128 test
```
일단 작동은 하지만 완벽하지는 않다. 다시 한번 더 실행시켜보면, 이전과는 다른 실험 데이터 세트를 만들어 낼 것이다. 그래서 하나의 솔루션으로는 이렇게 뽑아낸 실험 데이터를 따로 저장해두거나, 아니면 랜덤 시드를 저장해주는 함수를 작성하는 것이다. 하지만 이 또한 데이터세트가 새로이 업데이트 된다면 의미가 없어져서 다시 만들어야한다. 그래서 이러한 점도 해결하기 위해 실험 데이터 세트로 갈 수 있는지 없는지 정해주는 식별자를 하나 만들어주는 것이다. 예시로 예시의 식별자에 대한 해쉬 값을 연산한 다음에, 그 값의 맨 마지막 해쉬 바이트를 저장하고, 그 값이 51 이하 (~20% of 256)는 실험 데이터 세트로 사용한다. 그래서 데이터 셋이 업데이트 되더라도 이전에 사용한 값들을 유지하면서 추가적으로 새롭게 업데이트 된 데이터 세트에 대해서도 실험 데이터 값을 따로 뺄 수 있는 것이다. 단 한가지 확실하게 해야할 것은 학습에 사용된 데이터가 아니여야한다는 것이다. 그래서 이를 구현한 코드는 다음과 같다.

```
#해시 연산을 하기위한 라이브러리 호출
import hashlib

#각 식별자에 대하여 정해진 비율에 따라서 해시 연산을 함
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

#해시연산된 값을 비율에 맞게 테스트 데이터로 가져옴
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
```
하지만 housing dataset에는 식별자로 쓸만한 것이 없다. 그래서 열의 인덱스를 식별자로 사용한다.
```
housing_with_id = housing.reset_index()   # `index`를 column에 추가
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```
그래도 여전히 문제가 있다. 새로운 데이터가 들어오면 기존 데이터 세트 끝에 리스트가 추가되고 삭제도 일어날 수도 있다. 그래서 이러한 값이 좀 더 안정적인 값을 갖게하기 위해서 housing 데이터의 경도 변수를 사용하는 것이다. 경도는 변하지 않고 고유적이기 때문에 좀 더 안전한 연산이 가능하다.
```
# 경도 변수를 추가해 인덱스를 넣음에 있어 더 안정적일 수 있도록한다.
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```
scikit-learn에도 다양한 방법으로 다중 서브셋 분할을 해주는 함수가 있다. 가장 간단한 함수가 `train_test_split`이라는 함수인데 이는 우리가 앞에서 구현한 함수와 꽤나 똑같은 방법을 수행한다. 그래서 이 함수는 다음과 같이 사용할 수 있다. 먼저 분할할 데이터 세트를 넣어주고 `random_state`라는 랜덤 시드와 같은 역할을 수행해주는 곳에 설정할 값을 넣어준 뒤, `test_size`라는 분할 비율을 입력하는 곳에 입력을 해서 사용하면 된다. 
```
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```
데이터 세트의 크기가 충분하다면 이러한 방법들이 잘 먹히겠지만, 표본이 부족하다면 별로 좋지 못한 결과를 도출할 수도 있다. 예로 미국에서 1000명에 대한 전화조사를 실시하였다. 미국내 전체 남자와 여자의 비율은 51.3%, 48.7%인데, 전화 조사를 통한 성별 비율이 이와 비슷하게 떨어진다는 보장이 없다. 무작위로 전화해서 조사했더니 남자가 800명 넘게 조사되었을 수도 있다. 이런 경우 성별에 영향을 받는 학습 모델이라면 큰 문제가 발생할 수도 있다.

우리에게 중간 집 값을 예측하는데 가장 중요한 것이 중간 소득 값이라고 말해준 전문가와 이야기하고 있다고 가정해보자. 그렇다면 전체 데이터 세트에서 소득에 대한 카테고리들을 모두 대변해줄 수 있는 실험 데이터 세트를 만들고 싶을 것이다. 중간 소득 값은 연속된 숫자의 값이기에, 먼저 소득 카테고리 요소를 만드는 것이 중요하다. 아래에 있는 중간 소득치 히스토그램을 보자. 대부분의 중간 소득치는 $20,000~$50,000에 뭉쳐있는데, 일부 중간 소득치는 $60,000이 넘어가는 것도 있다. 각 계층에 대하여 데이터 셋에 있는 예시들이 충분하게 해야하는 것은 매우 중요하다. 아니면 측정치가 매우 편향되어 나올 것인데, 이는 각 계층이 많은 게 중요한게 아니고 각 계층에 충분히 데이터들이 확보되어 있는지가 더 중요하다는 것이다. 다음의 코드는 (소득 카테고리의 수를 제한하도록) 1.5로 중간 소득을 나누어 소득 카테고리 요소를 만드는 코드이다.
```
# 소득 카테고리의 수를 제한하기 위해, 1.5로 나눈다.
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# 5이상을 5라고 레이블 한다.
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-9.png)

이제 소득 카테고리를 기반으로 하여 계층화 샘플링을 할 수 있게 되었다. 한번 Scikit-Learn의 `StratifiedShuffleSplit`클래스를 사용해보자.
```
rom sklearn.model_selection import StratifiedShuffleSplit

#사용방법은 위의 train_test_split과 많이 다르지는 않다.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#Housing 데이터셋의 income_cat을 기반으로 하여 학습 인덱스와 테스트 인덱스에 따라 데이터를 분할한다.
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
``` 
이제 기대했던대로 잘 되었는지 한번 보자. 전체 housing 데이터세트에서 소득 카테고리에 따른 비율을 볼 수 있다.
```
>>> housing["income_cat"].value_counts() / len(housing)

3.0    0.350581
2.0    0.318847
4.0    0.176308
5.0    0.114438
1.0    0.039826
Name: income_cat, dtype: float64
```
비슷한 코드로, 실험 데이터 세트에 소득 카테고리 비율을 측정해서 볼 수 있다. 이는 Jupyter notebook코드에 잘 구현되어있다. 아래의 그래프는 다양한 방법들과 계층화 샘플링을 전체 데이터 세트에 대한 소득 카테고리 비율과 비교해볼 수 있다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-10.png)

이제 `income_cat` 요소를 제거해서 데이터 세트의 원래 상태로 되돌아 갈 수 있도록 한다.
```
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```


# 데이터에 대한 이해를 얻기 위한 데이터 시각화 및 탐구
여태 우리가 다룰 데이터의 종류에 대한 일반적 이해를 얻기위해서 빠르게 한번 훑어보고 왔다. 이제는 좀 더 깊게 들어가보자.

먼저 실험 데이터 세트는 따로 만들어두고, 지금은 오직 학습 데이터만 탐구할 것이다. 만약 데이터 세트가 너무 크다면 탐사용 세트를 따로 만들어야 했지만, 우리는 데이터 셋 규모가 꽤 작기 때문에 전체 데이터를 들고가서 바로 작업해도 좋다. 처리를 하면서 혹시 모를 상황을 대비해서 학습 데이터 세트를 미리 복사해둔다.
```
housing = strat_train_set.copy()
```
### 지리적 데이터 시각화하기
(경도와 위도를 가진) 지리적 데이터이기때문에, 데이터를 시각화하기 위해 모든 구역에 대한 점을 찍는 것은 괜찮은 생각이다.
```
#위도와 경도에 따라 점을 찍고 그 결과를 그림으로 출력
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-11.png)

딱봐도 캘리포니아 처럼 생겼지만, 어떠한 특정 패턴을 발견하기는 어렵다. `alpha`라는 옵션에 0.1이라고 설정해주면 데이터 밀도가 높은 지역을 시각화 하기 좋다.
```
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-12.png)

좀 더 일반적으로, 우리의 뇌는 사진에서 패턴을 잘 알아챈다. 하지만 우리는 패턴이 좀 더 눈에 띠게 시각화 파라미터를 좀 더 가지고 놀아볼 것이다.

아래의 집 값 점표를 보자. 각각의 원의 반지름은 그 구역에 대한 인구 밀도를 나타내며(옵션 s), 색은 가격을 나타낸다(옵션 c). 그리고 Jet이라고 불리는 (옵션 cmap)미리 정의해둔 색표를 사용할 것인데 이는 파란색이 낮은 값이고 빨간색이 큰 값임을 의미한다.
```
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-13.png)

위의 이미지는 우리도 잘 알다시피, 인구 밀도와, 지형적 특정이 집 값과 매우 연관성이 높다는 것을 말해주고 있다. 이는 주군집을 찾는 클러스터링 알고리즘에 사용하기 유용할 것이고, 클러스터의 중심과의 가까움 정도를 측정하는 새로운 특징을 추가하기에도 유용할 것이다. 비록 북쪽 해안에 있는 집 값이 높은 편은 않지만 해안과의 거리를 나타내는 요소도 유용할 것이다.

## 상관관계 살펴보기
 
데이터 세트가 그렇게 크지 않기 때문에 `corr()`함수를 사용해서 각 모든 쌍들에 대한 표준상관계수(standard correlation coefficient)를 쉽게 계산할 수 있다.
```
corr_matrix = housing.corr()
```
그러면 이제 중간 집 값이 어떤 요소에 얼마나 관련되어 있는지 볼 수 있다.
```
>>> corr_matrix["median_house_value"].sort_values(ascending=False)

median_house_value    1.000000
median_income         0.687160
total_rooms           0.135097
housing_median_age    0.114110
households            0.064506
total_bedrooms        0.047689
population           -0.026920
longitude            -0.047432
latitude             -0.142724
Name: median_house_value, dtype: float64
```
상관 계수는 -1부터 1로 정의되는데 1에 가까울 수록 강한긍정, -1에 가까울수록 강한 부정(위도가 북쪽으로 갈수록 집 값이 낮아짐)을 나타내게 되며, 0에 가까운 숫자들은 선형적인 관계로 나타내기 힘들다는 것을 의미한다. 

요소간 상관관계를 확인하는 또한다른 방법은 Pandas의 `scatter_matrix`함수인데 이는 모든 숫자 요소들간의 관계를 점찍어서 보여주는 함수이다. 그림은 아래에 있으며 코드는 다음과 같다.
```
# from pandas.tools.plotting import scatter_matrix # Pandas 옛날 버전을 위한 코드
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-15.png)

최상단 좌측에서 최하단 우측으로 가는 대각선 상에 있는 데이터들은 히스토그램으로 나와있는데, 이는 자기자신에 대해 자기자신의 상관관계를 나타낸 것으로 즉 그냥 자기 자신을 나타내는 것이기에 자신의 값을 표현하는 히스토그램으로 표시되는 것이다.

중간 집 값을 예측하는데 가장 유망한 요소는 중간 소득값이며 이 둘의 관계를 표현한 그림을 좀 더 확대해서 보자
```
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
```
이 plot은 몇가지 특징을 보여준다. 먼저, 서로의 상관관계가 매우 강하다는 것이다. 위로 향하면서 밀도가 분산되어있지도 않다. 그리고 두번째로 우리가 전에 보았던 데이터 계층화를 총해 만든 것과 비슷해 보이는 $50,000의 선이 보인다. 하지만 $35,000과 $45,000에서도 수평선이 생성되어 있고, $28,000쯔음에도 선이 있는 것처럼 보인다. 데이터를 우리의 알고리즘에 학습시킬 때 이런 이상한 데이터들을 제거할 수 있으면 학습의 성능을 위해서라도 제거하는 것이 좋다.
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/02/2-16.png)

## 요소 조합 실험학기
이전 섹션에서 데이터에 대한 이해를 얻기위해 데이터를 탐구하는 몇가지 아이디어를 살펴보았다. 이렇게 학습에 사용할 데이터에서 기이한 점들을 제거할 수 있고, 요소간 흥미로운 관계또한 확인할 수 있다. 그리고 꼬리가 무거운 분산을 가진 요소도 있음을 알았을 것이기에, 이를 변환해서 사용할 수 있을지 모른다고 생각할 수 있다. 

그래서 기존에 존재하는 변수들을 조합해서 학습에 좀 더 의미가 있는 데이터로 변환할 수 있다. 예를 들어 전체 방수와 가정수를 연관지어 생각해보자. 각각의 독립적인 변수로보면 별로 도움이 되지 못하지만 한 가정당 방의 개수라는 변수를 만든다면 좀 더 도움이 될 것이다. 같은 원리로 방 갯수당 침실 갯수, 가정 인구수로 변환하여 사용하면 각각 독립적으로 사용하였을 때 보다는 좀 더 의미가 있다.
```
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```
이제 상관관계 행렬을 다시 한번 보자
```
>>> corr_matrix = housing.corr()
>>> corr_matrix["median_house_value"].sort_values(ascending=False)

median_house_value          1.000000
median_income               0.687160
rooms_per_household         0.146285
total_rooms                 0.135097
housing_median_age          0.114110
households                  0.064506
total_bedrooms              0.047689
population_per_household   -0.021985
population                 -0.026920
longitude                  -0.047432
latitude                   -0.142724
bedrooms_per_room          -0.259984
Name: median_house_value, dtype: float64
```
나쁘지 않다! 앞에서 요소들을 조합해서 만든 새로운 요소들에 대한 값이 좋게 나왔다. 그렇다고 무조건 이러한 과정을 거쳐가야한다는 것은 아니다. 이는 좋은 프로토타입을 얻기 위한 이해를 빠르게 얻기 위한 것이다. 하지만 이러한 과정은 반복적이다. 프로토타입을 만들고 좀 더 성능 향상을 위해 다시 이 단계로 내려와서 처리를 다시하는 경우도 존재하기 때문이다.

# 기계학습 알고리즘을 위한 데이터 준비
이제 기계학습 알고리즘에 넣을 데이터를 준비할 차례이다. 그냥 보고 따라하는 것보다는 함수를 직접 써봐가면서 해보는 것을 추천한다.
먼저 학습데이터를 원래상태로 되돌리자(`strat_train_set`을 다시 한번 더 사용하자). 그리고 목표값과 예측변수를 같은 Transformation에 무조건 같은 방법으로 적용할 필요가 없다(`drop`함수는 복사한 데이터를 만들어 내지만 `strat_train_set`에 영향을 끼치지는 않는다).
```
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
``` 

## 데이터셋 청소(Dataset Cleaning)
대부분의 기계학습 알고리즘은 누락된 특징들이 있으면 제대로 작동할 수 없기때문에 이를 방지하기 위한 몇가지 함수를 만들어야한다. 앞에서 `total_bedroom`에 누락된 값들이 있었기에 이를 고쳐볼 것이다. 이를 수행하는데 3가지 방법이 있다.
* `선택 사항 1`:누락된 값을 가진 구역(특징) 제거
* `선택 사항 2`:누락된 값을 가진 요소 제거
* `선택 사항 3`:다른 값으로 채워넣기

`DataFrame`에 있는 `dropna()`,`drop()`,`fillna()` 함수를 사용하여 쉽게 이를 수행해줄 수 있다.
```
# 어느 한칸이라도 비어있는 열을 가져와서 조회
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

#"total_bedrooms"에서 비어있는 칸이 있는 열을 드랍
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # 선택 사항 1

# "total_bedrooms" 열 자체를 삭제
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # 선택 사항 2

# 비어있는 칸들을 전부 중간 값으로 채워넣기
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # 선택 사항 3
sample_incomplete_rows
```
만약 선택사항 3번을 사용한다면, 학습 데이터 셋에 대한 중간값을 계산하여 학습데이터 상에 누락된 값이 있는 곳에 중간 값을 넣어줄 것이다. 그리고 잊지말고 연산한 중간값을 저장해라. 나중에 테스트 데이터 세트로 우리의 시스템에 대해 평가할 때 비어있는 값을 이 값으로 매꿀 것이기 때문이다. 

Scikit-Learn이 누락된 값을 손쉽게 처리해주는 `Inputer`라는 클래스를 가지고 있다. 아래에 이를 어떻게 사용할 수 있는지 나와있다. 먼저 `Imputor`라는 인스턴스를 만들어주어야한다. 그리고 처리할 값들에 어떤 값을 넣을지 설정해주어야 한다.
```
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
``` 
중간 값은 채울 값의 요소가 모두 숫자로 되어있어야 가능하기에, 문자열 데이터를 가지고 있는 요소`Ocean_proximity`를 제외하고 그 이외의 값들을 복사한다. 
```
housing_num = housing.drop('ocean_proximity', axis=1)
```
이제 `fit()`이라는 함수를 사용해서 학습 데이터를 fitting할 수 있을 것이다.
```
imputer.fit(housing_num)
```
`imputer`는 각 요소들에 대해 연산한 중간 값들을 `statistics_`라는 변수에 저장해둔다. `total_bedrooms`라는 변수만 누락된 값이 있지만, 실제 시스템으로 출시되었을 때에는 어떤 값이 누락될 지 모른다. 그래서 이와 같은 안전장치를 만들어두는 것이다.
```
>>> imputer.statistics_
array([ -118.51  ,    34.26  ,    29.    ,  2119.5   ,   433.    ,
        1164.    ,   408.    ,     3.5409])

>>> housing_num.median().values
array([ -118.51  ,    34.26  ,    29.    ,  2119.5   ,   433.    ,
        1164.    ,   408.    ,     3.5409])
```
이제 학습된 중간 값으로 학습 데이터 세트를 변환하는데 "학습된" imputer를 사용해볼 수 있을 것이다. 
```
X = imputer.transform(housing_num)
``` 
결과는 변환된 특징들을 가지고 있는 평평한 배열 형태로 결과가 추출될 것이다. 만약 Pandas DataFrame으로 다시 넣어보고 싶다면, 디음을 입력해보아라.
```
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values))
```
## Handling Text and Categorical Attributes

앞에서 우리는 text로 된 요소라서 우리가 중간 값을 계산하기 힘들기 때문에 카테고리적 요소인 `Ocean_proximity`를 따로 빼놓았다. 대부분의 기계학습 알고리즘은 대부분 숫자 데이터를 다루는 것을 선호하기 때문에, 텍스트 라벨을 숫자로 변환해주어야한다. 

scikit-learn에서 `LabelEncoder`라는 이 일을 처리해줄 변환 함수를 가지고 있다.
```
# 현재 scikit-learn에서 `LabelEncoder`라는 클래스를 더이상 제공하지 않는다.
# 그래도 책에는 수록이 되어있기에 참고할 수 있도록 코드를 업로드하여 두었다.
# 현재는 OneHotEncoder라는 클래스를 사용하며, 해당 클래스는 아래에 설명이 나와있다.
>>> from sklearn.preprocessing import LabelEncoder
>>> encoder = LabelEncoder()
>>> housing_cat = housing["ocean_proximity"]
>>> housing_cat_encoded = encoder.fit_transform(housing_cat)
>>> housing_cat_encoded
array([0, 0, 4, ..., 1, 0, 3])
```
이제 우리는 어떠한 기계학습 알고리즘에서도 사용할 수 있다. Class_ 요소를 사용해서 학습된 Encoder를 사용해서 어떻게 텍스트 라벨이 어떤 숫자로 변형이 되었는지 확인할 수 있다. 
```
>>> print(encoder.classes_)
['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN']
```
이제 One Hot Encoding에 대해서 알아보자. 일단 이 이름의 유래는 0을 cold, 1을 hot이라고 하여 One hot encoding이라는 기법이 된 것이고 알고리즘은 아주 간단하다. 여기서 위에서 클래스는 0~4의 숫자로 매겨지는데, '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'를 각각 0, 1, 2, 3, 4로 매기어놓고, 주어진 값에 따라 0,1로 각 클래스에 값을 넣어 진행하는 인코딩 방식으로, 예를 들어 주어진 텍스트가 'ISLAND'라면, 결과는 [0,0,1,0,0], 2라고 카테고리가 매겨질 것이다.

scikit-learn에서 one-hot vector로 집적 카테고리적 값으로 변환해주는 `OneHotEncoder`를 제공한다. `fit_transfrom()`이 이차원 배열의 입력을 받는데, `housing_cat_encoded`는 1차원 배열이기에 한번 처리를 거쳐서 입력을 넣어주어야한다. 
```
>>> from sklearn.preprocessing import OneHotEncoder

>>> encoder = OneHotEncoder()
>>> housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
>>> housing_cat_1hot
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
	with 16512 stored elements in Compressed Sparse Row format>
```
이 함수의 결과는 NumPy배열 대신, SciPy 희소 행렬을 사용한다. 수 천개의 카테고리를 가진 카테고리적 값들을 가지고 있을 때에 유용하다. One hot encoding을 마치고 나면 수천개의 column을 가진 행렬을 얻을 수 있을 것이며, 각 열에는 대부분이 0으로 되어 있을 것이고 해당하는 카테고리 하나에만 1이라는 값이 매겨저 있을 것이다. 그래서 아주 규모가 큰 데이터에 대해서 클래스를 표현하기 위해 대부분이 0으로 이루어져 있는 희소 행렬로 표현하는 것은 굉장히 공간 낭비가 심할 것이다. 따라서 0이 아닌 값만을 가지고 있는 행렬상 위치값을 저장하고 있는 희소 행렬을 사용해서 표현한다. 그래도 우리에게는 2D 행렬로 보여질 것이다. 그래도 혹시나 진짜 NumPy형식으로 행렬로 변환하고자 한다면 다음의 명령어를 쳐서 볼 수 있을 것이다.
```
>>> housing_cat_1hot.toarray()
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.]])

```
책에서는 `LabelBinarizer`또한 다루지만, 더이상 scikit-learn이 제공하지 않아서 생략하였다.

## 변환함수 커스터마이징

비록 Scikit-Learn이 매우 유용한 변환 함수들을 제공해주지만, 데이터 청소 함수나 특정 요소들을 합치는 함수같이 우리들만의 함수를 작성할 것이다. 우리가 이제 해야할 것은 fit(), transform(), fit_transform()이라는 함수를 작성해볼 것이다. 기본 클래스로 `TransformerMixin`를 추가할 것이고,

또한 `BaseEstimator`라는 함수를 추가하면, 자동으로 하이퍼파라미터를 조율하는데 유용한(get_params()와 set_params()) 두가지 추가 메소드를  얻을 것이다. 예를 들어, 여기에 앞에서 이야기하였던 값을 합친 요소들을 추가하는 작은 변환 함수가 있다.
```
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```
이 예시에서는 변환 함수가 가지고 있는 하이퍼파라미터는 하나를 가지고 있으며 `add_bedrooms_per_room`를 기본값으로 `True`로 설정해놓았다. 이러한 하이퍼파라미터들은 이런 속성을 추가하는 것이 기계학습에 도움이 되는지 아닌지를 쉽게 알게 해준다. 좀 더 일반적으로, 우리가 해당 데이터에 대해 100%확신이 없는 경우에도 이러한 데이터 전처리 단계를 지나가도록 하이퍼파라미터를 추가할 수도 있다. 이러한 데이터 전처리가 자동화될수록, 좀 더 많은 조합방법을 시도해볼 수 있을 것이며, (시간을 절약하면서) 좀 더 좋은 조합 방법을 찾아내는 경향이 있다. 

## 특징 값 크기 조정
우리의 데이터에 적용하기위해 우리에게 필요한 가장 중요한 변환방법은 바로 특징 값의 크기를 조정하는 것(*Feature scaling*)이다. 몇가지 예외로 기계학습 알고리즘은 숫자로 된 입력 속성들은 매우 다양한 크기 값을 가지고 있다. 이는 하우징 데이터의 경우이기도 하다. 평균 수입은 0부터 15까지 밖에 범위가 없는 반면에 전체 방의 수는 6부터 39320까지 범위가 찍여있다. 그리고 타겟 값을 재조정하는 것은 일반적으로 요구되지는 않는다. 

모든 속성값들이 같은 크기의 범위를 갖도록 하는 일반적인 두가지 방법이 있는데 최소-최대 스케일링(*min-max Scaling*)과 표준화(*standardization*)이다.

최소-최대 스케일링(많은 사람들이 이를 정규화(*normalization*)이라고도 한다)는 꽤나 간단하다. 값들을 재조정해서 0과 1의 범위 내로 만들어내는 것이다. 최소값으로 빼준 뒤에 그 값을 최대값에서 최소값을 뺀 값으로 나누어준다. Scikit-Learn은 이러한 연산을 수행해주는 `MinMaxScaler`라는 변환 함수를 제공해준다. 만약 특별한 이유로 0에서 1사이의 값으로 나오는 것을 원하지 않는다면 하이퍼 파라미터`feature_range`로 원하는 범위를 따로 설정을 해줄 수도 있다.

표준화는 앞의 방법과는 꽤 다르다. 먼저 평균값으로 빼고(그래서 표준화된 값이 항상 평균이 0이다), 분산값으로 나누어서 결과로 나오는 분포는 단위 분산이 된다. 최소-최대 스케일링과는 다르게, 표준화는 특정한 범위로 정해져있지 않은데, 이는 몇몇 알고리즘에서 문제를 보이기도 한다.(신경망이 종종 입력값은 0~1값으로 받는다) 하지만 표준화는 이상점에 대해서는 덜 영향을 받는다는 것이다. 예로들어 실수로 어떤 구역의 평균 수입이 100이라고 해보자. 최소-최대 스케일링은 0~15에서 0~0.15로 감소하게되는 반면에 표준화는 이에 영향을 받지 않는다는 것이다. Scikit-Learn은 표준화를 해주는 함수로 `Standardization`을 제공해준다.

## 변환 파이프라인
보다시피, 데이터 변환 작업은 올바른 순서를 가지고 진행해야한다. 다행히 Scikit-Learn이 이런 변환함수들의 연속을 다루는데 도움을 주는 `Pipeline`함수들 제공한다. 아래 숫자로 이루어진 속성값들을 작은 파이프 라인으로 구성한 것을 보여준다.
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```
`Pipeline` 구성함수는 연속되는 단계를 정의하는 이름/처리함수(estimator) 쌍의 리스트를 받는다. 거의 대부분 마지막 처리함수는 변환함수이여야만 한다. ("__" 언더바 두개가 들어가 있지 않는 이상) 좋아하는 이름 어떤 것으로든 해줄 수 있다.

pipeline의 `fit()`함수를 호출하면, 마지막 처리함수에 도달할때 까지, 각각의 호출에 대한 결과를 파라미터로 다음 호출로 보내주어 모든 변환 함수에 대해서 연속적으로 `fit_transform()`을 호출하게 한다.

파이브라인은 같은 함수들을 마지막 처리함수로 보여준다. 이번 예시에는 마지막 처리함수가 `StandardScaler`이며, 이는 변환함수안데, 그래서 파이프라인은 모든 변환을 데이터 차례차례로 적용하는 `transform()`함수를 가지고있다. (우리가 `fit()`다음에 `transform()`함수를 호출해는 대신에 사용했던 `fit_transform()`함수도 가지고 있다)

이제 파이프라인에 NumPy 배열에서 수치의 열들을 손수 추출해보는 것 대신에 직접적으로 Pandas의 DataFrame을 입력해보면 좋을 것이다. Scikit-Learn에는 Pandas의 DataFrame을 다룰 수 있는 것이 없지만, 직접 커스텀 변환함수를 작성할 수 있다.
```
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
```
`DataFrameSelector`는 바람직한 속성값들은 골라내고, 나머지를 버리고, 결과로 나온 DataFrame을 NumPy 배열로 변환시켜주는 방식으로 데이터를 변환해줄 것이다. 이것으로 우리는 Pandas의 DataFrame을 받는 파이프라인을 쉽게 작성할 수 있다. 근데 오직 숫자로 된 값들만 받는다. 파이프라인은 `DataFrameSelector`으로 시작해서 앞에서 말했던 전처리 단계들에 따라 오직 숫자로 된 속성값만 뽑아온다. 그래도 카테고리로 되어있는 속성값들에 대해서도 `DataFrameSelector`에서 카테고리로 되어있는 속성값들을 가져온 다음에 `LabelBinarizer`를 사용해서 간단하게 또다른 파이프라인을 작성해볼 수 있다. 
```
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])
```
하지만 앞의 두가지 종류의 파이프라인을 어떻게 단일 파이프라인으로 합칠 수 있을까? 답은 Scikit-Learn의 `FeartureUnion`클래스를 사용하는 것이다. 먼저 전체 파이프라인들을 transform_list로 입력해준다. `transform()`함수를 불러오면 병렬로 각각의 변환함수들의 `transform()`함수가 실행되고, 각각의 결과를 기다린 다음에 그 결과들을 합쳐서 결과로 리턴시켜준다. (당연히 `fit()`함수를 불러오면 각각의 변환함수들의 `fit()`함수들을 호출한다) 숫자로 된 속성값과 카테고리로 된 속성값들을 다루는 전체 파이프라인은 아마 다음과 같을 것이다.
```
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
```
그리고 전체 파이프라인을 한번 실행시켜보자.
```
>>> housing_prepared = full_pipeline.fit_transform(housing)
>>> housing_prepared
array([[-1.15604281,  0.77194962,  0.74333089, ...,  0.        ,
         0.        ,  0.        ],
       [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.        ,
         0.        ,  0.        ],
       [ 1.18684903, -1.34218285,  0.18664186, ...,  0.        ,
         0.        ,  1.        ],
       ..., 
       [ 1.58648943, -0.72478134, -1.56295222, ...,  0.        ,
         0.        ,  0.        ],
       [ 0.78221312, -0.85106801,  0.18664186, ...,  0.        ,
         0.        ,  0.        ],
       [-1.43579109,  0.99645926,  1.85670895, ...,  0.        ,
         1.        ,  0.        ]])
>>> housing_prepared.shape
(16512, 16)
```
# 모델 고르기 및 학습시키기
마침내! 우린 문제에 대한 틀을 잡았다. 우리는 데이터를 받아 한번 싹 보았고 학습 데이터와 테스트 데이터로 나누어 샘플링도 해보았고 학습하기 좋게 데이터를 처리하는 변환 파이프라인을 작성해보았고, 기계학습 알고리즘이 자동으로 주어진 데이터를 준비할 수 있게 해주었다. 이제 학습 모델을 골라보고 기계학습 모델을 학습시킬 준비가 되었다.
## 학습 시키기 및 학습 데이터세트에 대해 평가하기
좋은 소식은 이전의 단계 덕분에 우리가 생각했던 것보다 더 간단해질 것이라는 것이다. 이전 Chapter에서 해본 것처럼 선형 회귀 모델을 한번 학습시켜보자.
```
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```
이제 선형회귀모델이 작동할 것이다. 한번 학습데이터에서 몇가지 인스턴스를 시도해보자.
```
>>> some_data = housing.iloc[:5]
>>> some_labels = housing_labels.iloc[:5]
>>> some_data_prepared = full_pipeline.transform(some_data)
>>> print("Predictions:", lin_reg.predict(some_data_prepared))
Predictions: [ 210644.60459286  317768.80697211  210956.43331178   59218.98886849
  189747.55849879]
>>> print("Labels:", list(some_labels))
Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
```
비록 예측이 정확하지는 않지만(첫번째 예측값이 40%정도) 잘 작동한다. Scikit-Learn의 `mean_squared_error`를 사용해서 전채 학습 데이터 세트에 대해서 이 회귀 모델의 RMSE를 측정해보자
```
>>> from sklearn.metrics import mean_squared_error
>>> housing_predictions = lin_reg.predict(housing_prepared)
>>> lin_mse = mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse = np.sqrt(lin_mse)
>>> lin_rmse
68628.198198489219
```
좋다. 없는 것보다는 나은 것이지만 분명히 좋은 점수치는 아니다. 대부분 구역의 `median_hauding_values`는 $120,000에서 $265,000 사이의 범위를 가지기에 예측 에러치가 $68628이라는 것은 매우 불만족스러울 것이다. 이는 학습 모델이 학습 데이터에 대해서 제대로 학습을 하지 못했다는(Underfitting) 것이다. 이는 예측을 좋게 내려주지 못할 만큼 특징값들이 충분한 정보를 제공해지 못했거나 그 모델이 충분히 예측을 제대로 내리도록 충분히 강력하지 못했다는 것을 의미한다. 이전 Chapter에서 보았다시피, 제대로 학습을 하지 못화는 것에대한 해결책으로는 좀 더 강력한 학습 모델을 선택하고, 좀 더 나은 특징값들을 학습 알고리즘에 입력해주거나 학습 모델에 걸려있는 제한 사항들을 줄여주는 것이다. 이 학습 모델은 정형되되어있지 않기에, 마지막 옵션(걸려있는 제한 사항에 대한 내용)을 배제한다. 그렇다면 좀 더 측징값들을 추가해보자. (예:인구 밀도) 하지만 먼저 좀 더 복잡한 학습 모델을 선택해서 돌려보자.

`DecisionTreeRegressor`로 학습을 시켜보자. 이 모델은 강력한 모델이고, 데이터에서 복잡한 비선형 관계를 찾는 능력이 있다. (의사결정트리는 Chapter6에서 자세히 다룬다)
```
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```
이제 모델이 학습을 했으니, 학습 데이터 세트에 대해서 한번 평가를 해보자.
```
>>> housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse
0.0
```
에러치가 전혀 없다고 나온다. 그렇다면 이 모델이 정말 완벽한 모델이라고 할 수 있는 것인가? 이 모델은 오히려 데이터에 대해서 과잉학습(Overfitting)이 발생했다고 봐야한다. 어떻게 알 수 있는가? 전에 보았지만, 우리가 모델을 실행시키기전까지 테스트 데이터를 건들이지 않았다. 그래서 이제 학습데이터 세트를 학습으로 쓸 부분과 검증용으로 쓸 부분으로 나누어 사용해 볼 것이다.

## 교차검증법(Cross-Validation)을 사용한 더 나은 평가
의사결정트리를 평가하는 한가지 방법은 학습 데이터를 학습 데이터와 검증 데이터 세트로 좀 더 작게 쪼개는 `train_test_split`을 사용해서, 학습 데이터 세트로는 학습을 시키고 검증 데이터 세트를 사용해 그 학습된 결과에 대해서 평가를 하는 것이다. 번거롭지만 어려울 것도 없고, 꽤 잘 작동할 것이다.

더 좋은 방법은 Scikit-Learn의 교차검증법(*cross-validation*)을 사용하는 것이다. 다음의 코드는 *K-ford Cross-Validation*을 수행해주는 코드로, 학습 데티어 세트를 10개의 서브셋으로 나누고, 이때의 서브셋을 *fold*라고 한다. 그리고 의사결정트리 모델을 학습시키고 평가하는 것을 10번 수행한다. 매법 학습 데이터 세트를 10개 중 9개를 사용하고 그 나머지를 검증용 서브셋으로 사용하는데 매 번 그 검증 서브셋을 다르게 사용해야한다. 결과값은 평가 10회에 대한 평가점수를 가지고 있는 배열이 나올 것이다.
```
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```
```
Scikit-Learn의 교차검증법은 손실함수보다는 유틸리티 함수로 사용되는데, 점수를 내는 함수는 실제로
MSE의 반댓값을 가진다. 이는 왜 이전의 코드에서 제곱근을 계산하기 전에 `score`에 `-`를 붙였는 지를
말해준다. 
```
한번 결과를 확인해보자
```
>>> def display_scores(scores):
...     print("Scores:", scores)
...     print("Mean:", scores.mean())
...     print("Standard deviation:", scores.std())
... 
>>> display_scores(tree_rmse_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
​
display_scores(tree_rmse_scores)
Scores: [ 70232.0136482   66828.46839892  72444.08721003  70761.50186201
  71125.52697653  75581.29319857  70169.59286164  70055.37863456
  75370.49116773  71222.39081244]
Mean: 71379.0744771
Standard deviation: 2458.31882043
```
이제 의사결정트리가 앞의 선형 모델보다 더 좋지못하다는 것을 볼 수 있다. 실제로, 선형회귀 모델보다도 수행능력이 좋지못하다! 교차검증법은 우리가 우리의 모델의 수행 능력에 대한 평가 뿐만 아니라, 이 평가치가 얼마나 정확한지(표준편차)도 측정한다. 의사결정트리는 대략 79,379 ± 2458 정도의 점수치를 보인다. 만약 하나의 검증 데이터 세트만을 사용했다면 이러한 정보를 얻을 수 없을 것이다. 교차검증법은 모델을 여러번 학습 시켜야만 결과를 출력시키기에 항상 가능한 것이 아니다. 

이제 한번 선형 회귀 모델에 대해서도 같은 점수치를 측정해보자.
```
>>> lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
...                              scoring="neg_mean_squared_error", cv=10)
... 
>>> lin_rmse_scores = np.sqrt(-lin_scores)
>>> display_scores(lin_rmse_scores)
Scores: [ 66782.73843989  66960.118071    70347.95244419  74739.57052552
  68031.13388938  71193.84183426  64969.63056405  68281.61137997
  71552.91566558  67665.10082067]
Mean: 69052.4613635
Standard deviation: 2731.6740018
```
의사결정트리가 과잉학습을 해서 선형 회귀 모델 보다도 수행 능력이 나쁘다는 것을 알 수 있다.

마지막으로 `RandomForestRegressor`를 학습시켜보자. Chapter 7에서 다루게 될 랜덤 포레스트는 여러 개의 의사결정 트리를 랜덤 서브셋으로 학습을 시키고 각각의 모델에 대한 예측결과를 평균낸다. 좀 더 여러가지 종류의 모델로 모델을 구현하는 것을 앙상블 학습*Ensemble Learning*이라고 한다. 이는 종종 단일 기계학습 알고리즘 보다 더 좋은 성능을 낸다. 다른 모델과 겹치는코드는 생략한다. 자세한 코드는 이 레포지토리의 코드에 나와있다.
```
>>> from sklearn.ensemble import RandomForestRegressor
>>> forest_reg = RandomForestRegressor(random_state=42)
>>> forest_reg.fit(housing_prepared, housing_labels)
>>> [...] #생략
>>> forest_rmse
21941.911027380233
>>> display_scores(forest_rmse_scores)
Scores: [ 51650.94405471  48920.80645498  52979.16096752  54412.74042021
  50861.29381163  56488.55699727  51866.90120786  49752.24599537
  55399.50713191  53309.74548294]
Mean: 52564.1902524
Standard deviation: 2301.87380392
```
좀 더 좋아졌다! 랜덤 포레스트는 매우 유망있는 모델이다. 하지만 여전히 검증 데이터 세트보다 학습 데이터 세트에 대한 점수가 더 낮은데 이는 학습 모델이 학습 데이터 세트에 대해서 과잉학습을 한 것을 의미한다. 과잉학습에 대한 가능한 해결택은 학습 모델을 간소화하거나, 제한을 조금 준다거나(정형화), 학습데이터 세트를 더 많이 모으는 것이다. 하지만 랜덤포레스트에 대해서 더 깊에 다루기전에, 좀 더 기계학습의 다양한 종류의 다른 모델(예로 Support Vector Machine의 다양한 종류의 커널기법)을 먼저 보자. 목표는 2~5가지 정도의 사용할 만한 모델의 후보군을 만드는 것이다.
```
여러분들은 여태 경험해보았던 모든 학습 모델들을 저장할 필요가 있는데, 그래야 원할 때마다 쉽게 그 모델을
다시 불러올 수 있다. 교장검증법의 점수치나 실제 그 예측한 결과치 만큼이나 하이퍼 파라미터와 학습된 파라
미터들을 잘 저장해두어야한다. 이는 후에 모든 모델에 종류에 대해서 비교를 할 수 있게 해주고, 각 모델들이
만들어 낸 에러를 비교할 수 있게 해준다. 파이썬 "pickle" 라이브러리나 "sklearn.externals.joblib"
을 사용해서 Scikit-Learn 모델들을 쉽게 저장할 수 있다. "sklearn.externals.joblib"는 거대한
NumPy 배열을 좀 더 효율적으로 직렬화해준다.
      from sklearn.externals import joblib
      joblib.dump(my_model, "my_model.pkl") # DIFF
      # and later ...
      my_model_loaded = joblib.load("my_model.pkl") # DIFF
```

# 학습 모델 잘 조율하기
이제 사용할 만한 모델들의 후보군을 가지고 있다고 가정해보자. 이제 우리는 이들을 좀 더 잘 조율(fine-tune)할 필요가 있다. 몇가지 방법을 살펴보자
## 그리드 서치 (Grid Search)
조율하는 하나의 방법은 훌륭한 조합의 하이퍼 파라미터 값을 찾을 때까지 손수 직접 하이퍼 파라미터를 건들이는 것이다. 이는 매우 지루한 작업으로, 수많은 조합 방법을 일일히 보고 있을 시간이 없을 것이다.

대신에 Scikit-Learn의 `GridSearchCV`를 사용하는 것이다. 우리가 해야할 것은 어떤 하이퍼 파라미터를 탐색하기를 원하는 것인지, 어떤 값을 시도해 볼지이며, 교차검증법을 이용해서 가능한 모든 조합의 하이퍼파라미터 값을 평가하는 것이다. 예를들어 다음의 코드는 `RandomForestRegressor`에 대해서 최적조합의 하이퍼 파라미터 값을 찾아내는 코드이다.
```
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
```
```
하이퍼 파라미터가 어떤 값을 가져야할 지 모를때, 간단한 접근법으로는 연속적으로 10의 배수(*Consecutive
powers of 10*)를 사용해보는 것이다. (10, 100, ...) (아니면 "n_estimator"에서 보여지는 것처럼,
만약 좀 더 세밀한 탐색을 진행하고 싶다면, 10이 아닌 좀 더 작은수로 해볼수 도 있다.)
```
`param_grid`는 첫번째 `dict`에 명시되어있는 `n_estimators`과 `max_features`의 하이퍼 파라미터 값들의 조합 3 X 4 = 12개를 모두 평가 할 것이라고 Scikit-Learn에게 말한다.(지금 하이퍼 파라미터가 무엇을 의미하는지 몰라도 걱정할 필요가 없다. Chapter 7에서 자세하게 설명할 것이다) 그리고 두번째 `dict`에 있는 2 X 3 = 6 개 조합의 하이퍼 파라미터 조합에 대해서 모두 시도해 볼 것이다. 이번에 `bootstrap` 하이퍼 파라미터를 (이 하이퍼 파라미터의 기본 값인)True가 아닌 False라고 해둘 것이다. 

대체로, 그리드 서치는 12 + 6 = 18 개 조합의 `RandomForestRegressor` 하이퍼 파라이터 값들을 탐색해볼 것이고, 이는 각각 5번 정도 학습을 할 것이다.(5-fold Cross-Validation을 사용하기 때문) 다른말로 대체로, 18 X 5 = 90의 학습을 진행하는 것이다! 이는 꽤 시간이 오래걸리지만, 이것이 끝나면, 다음과 같은 최적조합의 파라미터 값을 얻게될 것이다. 
```
>>> grid_search.best_params_
{'max_features': 8, 'n_estimators': 30}
``` 
```
8과 30은 주어진 값들 중에서 모두 가장 높은 값들이기 때문에, 계속 개선될 때까지 가능하면 좀 더 
높은 값을 주고 다시 서치를 해보는 것이 좋다.
```
또한 최고의 최적 처리함수를 직접적으로 다루어 보는 것도 좋을 것이다
```
>>> grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=42,
           verbose=0, warm_start=False)
```
```
"GridSearchCV"가 만약 (기본값으로)"refit=true"로 초기화된다면, 교차검증법으로 최고의 처리함수를
 찾기만 한다면 전체 학습 데이터에 대해서 재학습 할 것이다. 데이터를 더 입력하는 것이 성능할 더 좋게 하기
때문에 이는 더 좋은 생각이다.
```
그리고 당연히 평가 점수도 나오게 된다
```
>>> cvres = grid_search.cv_results_
... for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
...     print(np.sqrt(-mean_score), params)
    print(np.sqrt(-mean_score), params)
63647.854446 {'max_features': 2, 'n_estimators': 3}
55611.5015988 {'max_features': 2, 'n_estimators': 10}
53370.0640736 {'max_features': 2, 'n_estimators': 30}
60959.1388585 {'max_features': 4, 'n_estimators': 3}
52740.5841667 {'max_features': 4, 'n_estimators': 10}
50374.1421461 {'max_features': 4, 'n_estimators': 30}
58661.2866462 {'max_features': 6, 'n_estimators': 3}
52009.9739798 {'max_features': 6, 'n_estimators': 10}
50154.1177737 {'max_features': 6, 'n_estimators': 30}
57865.3616801 {'max_features': 8, 'n_estimators': 3}
51730.0755087 {'max_features': 8, 'n_estimators': 10}
49694.8514333 {'max_features': 8, 'n_estimators': 30}
62874.4073931 {'max_features': 2, 'n_estimators': 3, 'bootstrap': False}
54643.4998083 {'max_features': 2, 'n_estimators': 10, 'bootstrap': False}
59437.8922859 {'max_features': 3, 'n_estimators': 3, 'bootstrap': False}
52735.3582936 {'max_features': 3, 'n_estimators': 10, 'bootstrap': False}
57490.0168279 {'max_features': 4, 'n_estimators': 3, 'bootstrap': False}
51008.2615672 {'max_features': 4, 'n_estimators': 10, 'bootstrap': False}
```
이 예시에서, `max_feartures` 하이퍼 파라미터로 8, `n_estimators` 하이퍼 파라이터로, 30을 설정하는 것이 최적을 솔루션임을 얻게될 것이다. 이 조합에 대한 RMSE 점수는 49,694임을 알 수 있고, 이는 기존에 52,564를 보여주는 기본 하이퍼 파라미터를 사용해서 얻은 점수보다 조금 더 나은 수준이다. 축하한다. 우리는 성공적으로 우리의 최적의 모델을 잘 조욜해냈다.
```

```
## 확률적 탐색 (Randomized Search)
그리드 서치 접근법은 아전의 예시처럼 상대적으로 적은 조합법을 찾을 때는 괜찮지만, 하이퍼 파라미터 탐색 공간이 널을 때, `RandomizedSearchCV`를 사용하는 것이 더 선호될 때도 있다. `Grid Search`와 비슷하게 쓰이기는 하지만, 모든 가능한 조합법을 시도해보는 대신에, 매 반복횟수마다 각각의 하이퍼 파라미터에 대해서 주어진 임의의 수에 대한 조합의 값을 평가하게된다. 이러한 접근법은 두가지의 좋은 장점을 가지고 있다.
* 1000번 반복에 대해서 임의로 값을 찾는다면, 이러한 접근법은 오직 하이퍼 파리미터 당 몇가지 값들로만 찾는 그리드 서치를 하는 대신에 하이퍼 파라미터에 1000번의 각기 다른 값들을 입력해보면서 탐색하게 될 것이다
* 반복횟수를 설정함으로써 우리가 하이퍼 파리미터에 할당하고 싶은 연산 자원에 대해서 더 세세하게 다룰 수 있다.

## 앙상블 기법 (Ensemble Methods)
우리의 시스템을 잘 조율시키기 위한 도다른 방법은 최고로 잘 수행해주는 모델을 합쳐보는 것이다. 그 그룹(앙상블)은 종종 개별적 모델보다 좀 더 수행능력이 좋다. 이 내용은 Chapter 7에서 다룰 예정이다.

## 최고의 모델과 그 모델의 에러를 분석하자
종종 최고의 모델에 대해서 조사해보는 것으로 문제에 대한 괜찮은 통찰력을 얻는다. 예로 `RandomForestRegressor`는 정확한 예측을 내기 위해 각각의 속성값에 대한 상대적 중요도를 나타낼 수 있다. 
```
>>> feature_importances = grid_search.best_estimator_.feature_importances_
>>> feature_importances
array([  7.33442355e-02,   6.29090705e-02,   4.11437985e-02,
         1.46726854e-02,   1.41064835e-02,   1.48742809e-02,
         1.42575993e-02,   3.66158981e-01,   5.64191792e-02,
         1.08792957e-01,   5.33510773e-02,   1.03114883e-02,
         1.64780994e-01,   6.02803867e-05,   1.96041560e-03,
         2.85647464e-03])
```
속성 값의 이름과 그 중요도 점수를 같이 표시를 해보자
```
>>> extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
>>> cat_one_hot_attribs = list(cat_encoder.categories_[0])
>>> attributes = num_attribs + extra_attribs + cat_one_hot_attribs
>>> sorted(zip(feature_importances, attributes), reverse=True)
[(0.36615898061813418, 'median_income'),
 (0.16478099356159051, 'INLAND'),
 (0.10879295677551573, 'pop_per_hhold'),
 (0.073344235516012421, 'longitude'),
 (0.062909070482620302, 'latitude'),
 (0.056419179181954007, 'rooms_per_hhold'),
 (0.053351077347675809, 'bedrooms_per_room'),
 (0.041143798478729635, 'housing_median_age'),
 (0.014874280890402767, 'population'),
 (0.014672685420543237, 'total_rooms'),
 (0.014257599323407807, 'households'),
 (0.014106483453584102, 'total_bedrooms'),
 (0.010311488326303787, '<1H OCEAN'),
 (0.0028564746373201579, 'NEAR OCEAN'),
 (0.0019604155994780701, 'NEAR BAY'),
 (6.0280386727365991e-05, 'ISLAND')]
```

## 테스트 데이터 세트로 우리의 시스템을 평가해보자

# 시스템 출시와 감시 및 유지
완벽하다. 이제 출시를 해도 좋다고 승인을 받았다. 이제 제품에 들어갈 우리의 솔루션을 준비해야한다. 특히 제품의 입력 데이터를 우리의 시스템에 연결해주고, 테스트 결과를 작성해보아야한다.

우리는 또한 정해진 간격 마다 우리의 시스템을 실시간으로 수행을 확인해주고, 문제가 발생하면 알려주는 감시 코드를 작성할 필요가 있다. 이는 갑작스러운 고장뿐만 아니라 성능 저하를 잡아내주기 때문에 아주 중요하다. 그리고 이는 꽤나 일반적인 일인데, 정기적으로 새로운 데이터에 대해서 학습 모델을 학습시킴에도 불구하고 시간이 지나면서 학습에 관여된 데이터로 인해 학습모델이 점점 망가지는 경향이 있기 때문이다. 

우리의 시스템에 대한 수행능력을 평가하는 것은 시스템의 예측값들을 샘플링하고 평가하는 것이 필요할 것이다. 이는 일반적으로 사람이 평가하게 된다. 이런 분석가들은 아마 그 분야의 전문가들이나 크라우드소싱 플랫폼(예를들어 아마존의 mechanical turk나 CrowdFlower 같은 곳)의 사람들이 될 지 모른다. 어떤 방법이든지 우리의 시스템에 사람이 직접 평가하는 파이프라인도 필요하다.

우리는 또 시스템에 입력되는 데이터의 질에 대해서도 확인해야만 한다. 때때로 수행능력이 조금씩 감소하기도 하는데 질이 떨어지는 신호 때문이다. (예를들어 기계가 제대로 움직이지 않는 것을 감지하는 센서가 이상한 값을 보낸다거나 다른 팀의 결과가 너무 오래되었다든지 등등) 하지만 성능이 어느정도 저하되면 알려줄 것이다. 만약 우리의 시스템에 들어가는 데이터를 감시한다면 이를 초기에 잡아낼 수 있을 것이다. 입력을 감시하는 것은 실시간 학습 시스템에서 아주아주 중요하다.

마지막으로, 우리는 일반적으로 신선한 데이터를 사용해서 정기적으로 모델을 학습시키고 싶을 것이다. 가능하다면 이런 프로세스도 자동으로 만들어주어야한다. 자동화를 해놓지 않는다면, 모델을 오직 매 6달에 한번(잘해야) 업데이트를 할 것이고 시스템의 성능은 요동치면서 저하하게될 것이다. 시스템이 실시간 학습 시스텝이라면, 정해진 기간마다 그 모델의 상태에 대한 스냅샷을 저장해두자. 그래야 이전의 작업상태로 쉽게 되돌릴 수 있다.

# 도전해보자
다행히도 이 Chapter는 기계학습 프로젝트라는 것이 무엇인지에 대한 느낌을 잘 보여주고, 우리가 훌륭한 시스템을 학습시키는데 사용하는 몇가지 툴을 보았다. 보다시피 대부분의 일은 데이터를 전처리하고, 감시 도구를 구축하고, 사람이 평가해주는 파이프라인을 세팅하고, 보편적인 학습 모델을 자동으로 학습 시키는 것이다.
기계학습 알고리즘도 당연히 중요하다. 하지만 어떤 진보한 알고리즘을 보느라 전체 시간을 소모해서 전체 프로세스를 만질 시간이 불충분해지는 것 보다는 전체 프로세스에 대해 편안해지고 2~3가지 알고리즘을 잘 알고 있는 것을 더 선호된다. 

그래서 만약 아직 해보지 못했다면 노트북을 꺼내서 분석해보고 싶은 데이터세트를 골라보자. 그래서 처음부터 끝까지 전체 프로세스를 한번 경험해보는 것도 좋을 것이다. 시작하기 좋은 사이트로 [Kaggle](http://kaggle.com/)이라는 공모전 웹사이트가 있다. 여기서 여러분들은 데이터 세트를 직접 가지고 놀 수 있으며 목표를 향해 도전해볼 수도 있고, 사람들과 경헙을 나누어 볼 수도 있을 것이다.

**[뒤로 돌아가기](https://github.com/Hahnnz/handson_ml-Kor/)**
