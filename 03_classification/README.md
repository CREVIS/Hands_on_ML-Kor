Chapter 3. 분류
=========
Chapter 1에서 우리는 지도학습의 가장 흔한 일은 회귀(Regression, 값을 예측)그리고 분류(Classification, 클래스를 예측)이다. Chapter 2에서는 선형회귀, 의사결정트리, 랜덤포레스트(후에 자세하게 설명할 것임) 알고리즘들로 집 값을 예측하는 회귀 기법을 사용해보았다. 이번에는 분류 시스템에 집중해보자.

# MNIST
이 Chapter에서는 우리는 MNIST데이터 세트를 사용할 것이다. MNIST 데이터 세트는 미국 통계청의 직원들과 고등학생들이 손으로 직접 쓴 숫자 이미지 7만장을 가지고 있다. 이 데이터 세트는 너무많이 연구가 되어 기계학습계의 "Hello world"와 같은 존재이다. 새로운 기계학습 알고리즘이 나오면 이 데이터 세트를 사용해서 얼마나 잘 수행되는지 증명하기도 하며, 언제든지 누군가가 기계학습을 배울때는 꼭 MNIST 데이터 세트를 거쳐간다.

Scikit-Learn이 인기있는 데이터 세트를 다운로드하는데 도와줄 많은 함수들을 가지고 있다. MNIST도 그 중 하나이다. MNIST 데이터 세트를 다운로드 허기 위해 다음의 코드를 실행시켜보자.
```
>>> from sklearn.datasets import fetch_mldata
>>> mnist = fetch_mldata('MNIST original')
>>> mnist
{'COL_NAMES': ['label', 'data'],
 'DESCR': 'mldata.org dataset: mnist-original',
 'data': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ..., 
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
 'target': array([ 0.,  0.,  0., ...,  9.,  9.,  9.])}
```
Scikit-learn이 일반적으로 다음과 같은 점들을 가진 사전적 구조로 데이터세트를 호출해온다.
* 데이터 세트를 표현하는 _DESCR_ 키
* 하나의 세로당 특징 하나, 하나의 가로당 인스턴스 하나로 이루어진 배열을 가진 _data_ key
* 레이블 값을 가진 배열을 포함한 *target*key

이제 배열을 살펴보자.
```
>>> X, y = mnist["data"], mnist["target"]
>>> X.shape
(70000, 784)
>>> y.shape
(70000,)
```
70000개의 이미지가 있으며 784개의 특징 값들을 가진 이미지들이다. 이는 일단 이미지가 28X28 픽셀 크기로 이루어져 있으며, 각각의 특징들은 색의 진함 정도를 0~255의 값으로 표현한다. 이제 데이터 세트에서 한 숫자를 뽑아보자. 우리가 할 것은 인스턴스의 특징벡터를 골라서 이를 28X28크기의 배열로 사이즈를 재조정하고, Matplotlib의 `imshow()`함수를 사용해서 확인을 해보자
```
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")

save_fig("some_digit_plot")
plt.show()
```

![code3-1](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-1.png)

5처럼 보이기는 한다. 실제로 이제 레이블은 어떻게 되어있는지 확인해보자.
```
>>> y[36000]
5.0
```

아래의 그림은 분류의 복잡성을 느껴볼 수 있도록 MNIST 데이터 세트에 들어있는 이미지 몇가지를 더 보여준다.

![3-1](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-1.png)

계속 진행하기전에, 우리는 항상 실험 데이터 세트를 만들어서 따로 빼두어야한다. 근데 MNIST데이터 세트는 실제로 학습데이터 앞에서 60000장, 실험 데이터 세트를 뒤에서 10000장을 이미 나누어놓았다. 
```
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```
이제 학습데이터 세트를 섞어보자. 이는 Cross-validation를 하면서 각각의 fold들이 어느정도 골고루 나누어 가져 서로가 비슷하도록 하는 것이다. 좀 더 나아가 일부 학습알고리즘은 학습 인스턴스의 순서에 민감하기도 할 뿐더러, 제대로 작동되지 않을 때도 있다. 데이터 세트를 섞는 것은 이런일이 일어나기 위한 것을 방지하고자하는 것이다.
```
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

# 이항분류 학습하기
이제 하나의 숫자만을 구별하고자 한다고 할 때 이런 문제를 어떻게 풀까? 숫자 5를 예로 들면, 5-분류기라고도 할 수있다. 이는 오직 5인지 아닌지로 두가지의 클래스 사이를 구분하는 능력을 가지는 좋은 *이항분류*의 예시가 될 것이다. 이제 이 분류 시스템을 위한 타겟 벡터를 만들어보자
```
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```
이제 분류 모델을 골라서 학습을 시켜보자. Scikit-Learn이 제공하는 `SGDClassifier`라는 클래스를 사용해서 확률적 경사 하강법(Stochastic Gradient Descent:SGD)이라면 시작하기에 좋을 것이다. 이 분류모델은 커다란 데이터 세트를 다루는 능력이 있다는 장점이 있다. 하지만 이는 일부분이다. 왜냐하면 SGD는 독립적으로 학습 인스턴스를 한번에 하나씩 다룬다.(이는 Online Learning, 실시간 학습에서 잘 어울리는 것이다.) 전체 데이터 세트로 SGD Classifier를 사용해 학습을 시켜보자
```
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```
```
SGD Classifier는 학습을 하는 동안에 랜덤 함수를 사용한다. (그래서 "stochastic" 확률적이라는 이름이다.)
만약 이 때 사용되었던 임의특성을 유지하고 싶다면, random_state 파라미터를 추가해야한다.
```
이제 숫자 5의 이미지를 찾아내는데 사용할 수 있다.
```
>>> sgd_clf.predict([some_digit])
array([ True], dtype=bool)
```
이 분류 모델은 숫자 5로 예측되는 이미지를 5라고 예측할 것이다. 이제 모델을 평가하는 방법에 대해서 생각해보자

# 수행도 측정

분류 모델을 평가하는 것은 종종 회귀 모델보다 더 교묘하고 힘들기 때문에 이 Chapter에서 상당한 부분이 이 토픽에 대해서 댜루는 부분이다. 가능한 많은 함수들이 있기에, 커피 한잔 마시고 새로운 개념을 배울 준비를 하자.

## Cross-Validation을 사용해서 정확도 측정하기

학습 모델을 평가하는 좋은 방법은 Cross-validation을 사용하는 것인데, Chapter 2에서 한 것처럼하면 된다.

### Cross-Validation 구현
가끔 Scikit-Learn에서 제공하는 기존 함수들 것보다는 좀 더 Cross-validation 과정 전체를 컨트롤할 필요가 있다. 이러한 경우, Cross-validation을 직접 구현할 필요가 있다. 실제로 꽤나 간단하다. 다음의 코드들은 Scikit-Learn에서 제공하는 `Cross_val_score()`함수와 비교해도 거의 비슷하며, 결과 출력도 똑같이 해낸다.
```
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```
이제 K-fold Cross-Validation을 사용해서 SGD 분류 모델을 평가해주는 `cross_val_score()`라는 함수를 사용할 수 있다. K-fold Cross-Validation는 학습 데이터 세트를 K개 만큼의 folds(여기서는 3개)로 나누어 놓은 것을 뜻하며, 학습에 사용한 데이터 세트 fold들에 대해서 학습에 사용하지 않은 테스트 데이터를 사용해 학습모델을 평가하고 예측을 한다. (Chapter2를 보아라.)

```
>>> from sklearn.model_selection import cross_val_score
>>> cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([ 0.9502 ,  0.96565,  0.96495])
```
모든 Cross-Validation folds에 대해 95% 이상의 정확도를 보인다. 결과가 아주 좋아보인다. 하지만 너무 좋아하진 말자. 보기엔 좋아보이더라도 어쩌면 5는 제대로 구분 못하면서 5가 아닌 것들만 우구장창 잘 찍어내는 것일 수도 있다.
```
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
```
이 모델의 정확도를 제대로 알아보자.
```
>>> never_5_clf = Never5Classifier()
>>> cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([ 0.909  ,  0.90715,  0.9128 ])
```
맞다. 90%가 조금 넘는 정도이다. 이는 아주 간단히 정확도만 믿어서는 안된다고 말하고 있다. 숫자 5라는 이미지는 전체 데이터 세트 중에서도 약 10%를 차지하는데 숫자 5라고 제대로 예측하지 못하고 5가 아닌 다른 숫자 이미지들을 모두 5가 아니라고 맞추면 어쨌거나 90%는 넘어가게된다. 

그래서 어떤 학습된 모델을 평가하고자 할 때에는 정확도만 믿어서는 옳지않다. 

## 혼동 행렬
분류 모델의 성능을 평가하는데 좀 더 나은 방법은 *혼동 행렬*을 사용하는 것이다. 즉 A클래스 인 것은 B클래스라고 예측한 예시의 숫자들을 세는 것이다. 숫자 5의 이미지를 3이라고 혼동하여 예측한 횟수를 알기 위해서 5번째 행의 3번째 열을 확인할 것이다.

이를 연산하기 위해선 미리 예측한 값을 가지고 정답 레이블과 비교하는 것이다. 우리는 테스트 데이터 세트로 예측을 했지만 이를 일단 지금은 건들지말자. 대신에, `cross_val_predict()`를 사용해보자.
```
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
``` 
`cross-val_score()`함수처럼, `cross_val_predict()`함수도 K-folds Cross-Validation을 수행하지만, 평가 점수를 출력하는 대신에, 각 folds에 대한 결과를 출력한다. 이는 학습 데이터상의 각각의 예시에 대해 깨끗한 결과를 얻을 수 있다.(여기서 "깨끗한"의 의미는 학습하는동안 절대 보지못한 데이터를 사용한 것이다) 이제 `confusion_matrix()`함수를 사용해서 혼동 행렬을 만들 준비가 되었다. 정답 레이블 클래스를 (y_train_5), 예측한 결과를 (y_train_pred)라고 하자.
```
>>> from sklearn.metrics import confusion_matrix
>>> confusion_matrix(y_train_5, y_train_pred)
array([[53272,  1307],
       [ 1077,  4344]])
```
혼동 행렬에서 각각의 행들은 실제 클래스를 의미하며, 반면에 각각의 열은 예측한 클래스를 나타내는 것이다. 이 행렬의 1행은 5가 아닌 것들(Negative Class) 중에서 5가 아니라고 맞게 맞춘 것(true negative)을 뜻하며 53,272개가 이에 해당한다. 그리고 5가 아닌데 5라고 한 것(false positive)이 1307개가 존재한다. 그리고 5인데 5가 아니라고 한(false negative)가 1077개 존재하며, 5인 이미지를 5라고 맞게 잘 예측한 것(true positive)는 4344개가 존재한다. 최고의 분류모델은 5가 아닌 것들(Negative Class) 중에서 5가 아니라고 맞게 맞춘 것(true negative)과 5인 이미지를 5라고 맞게 잘 예측한 것(true positive)만이 있는 것이다. 다음 코드를 입력해보면 모두 답을 맞추었다고 할 때에 대한 결과를 보기위한 코드이다.
```
>>> confusion_matrix(y_train_5, y_train_perfect_predictions)
array([[54579,     0],
       [    0,  5421]])
```
혼동 행렬은 많은 정보를 준다. 하지만 우리는 좀 더 간결한 평가법으로 된 것을 원한다. 그래서 잘 맞춘 것들에 대한 정확도를 보는 재밌는 방법을 분류 모델의 **precision**이라고 한다. 5라고 예측한 것들 중에서 정말 맞춘 것들을 카운트해보는 것이다.

###### Equation3-1. Precision
![](https://render.githubusercontent.com/render/math?math=%5Ctext%7Bprecision%7D%20%3D%20%5Ccfrac%7BTP%7D%7BTP%20%2B%20FP%7D&mode=inline)

완벽한 **precision**를 얻는 간단한 방법은 단일 Positive 예측이 전부 맞아 떨어지게하면 된다. 하지만 이는 매주 좋지 못한 방법이다. 그래서 보통 **Recall**이라는 평가법이 같이 따라온다. 이는 간혹 **Sensitivity**, 혹은 **True Positive Rate(TPR)**이라고 하며 분류 모델이 맞게 예측한 Positive 예시의 비율이다.

###### Equation3-2. Recall
![](https://render.githubusercontent.com/render/math?math=%5Ctext%7Brecall%7D%20%3D%20%5Ccfrac%7BTP%7D%7BTP%20%2B%20FN%7D&mode=inline)

## **Precision**과 **Recall**
Scikit-Learn에서 Precision과 Recall을 포함한 여러가지 분류 모델 평가법을 제공하고있다. 
```
>>> from sklearn.metrics import precision_score, recall_score
>>> precision_score(y_train_5, y_train_pred)
0.76871350203503808
>>> recall_score(y_train_5, y_train_pred)
0.80132816823464303
```
이제 우리의 5-탐색기는 이전에 앞에서 본 정확도 만큼이나 대단해보이지 않는다. 오직 77%만이 이미지가 5라고 보인다고 하며, 80%가 5라고 찾아내는 것이다. 이는 종종 전형적으로 두가지 분류모델의 성능을 비교하고자 할 때, F1 score라는 단일 평가법에 Precision과 Recall이 조합된 것을 사용한다. 일반적인 평균 값은 모든 값들에 대해 동등하게 다루는데, 조화평균은 작은 값에 좀 더 무게를 준다. 결국에는 분류모델은 Recall과 Precision이 높아야 높은 F1 점수를 얻을 수 있다.

###### Equation3-3 F1 score
![](https://render.githubusercontent.com/render/math?math=F_1%20%3D%20%5Ccfrac%7B2%7D%7B%5Ccfrac%7B1%7D%7B%5Ctext%7Bprecision%7D%7D%20%2B%20%5Ccfrac%7B1%7D%7B%5Ctext%7Brecall%7D%7D%7D%20%3D%202%20%5Ctimes%20%5Ccfrac%7B%5Ctext%7Bprecision%7D%5C%2C%20%5Ctimes%20%5C%2C%20%5Ctext%7Brecall%7D%7D%7B%5Ctext%7Bprecision%7D%5C%2C%20%2B%20%5C%2C%20%5Ctext%7Brecall%7D%7D%20%3D%20%5Ccfrac%7BTP%7D%7BTP%20%2B%20%5Ccfrac%7BFN%20%2B%20FP%7D%7B2%7D%7D&mode=inline)

F1 점수를 계산하려면 `f1_score()`라는 함수를 호출하기만 하면 된다.

```
>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)
0.78468208092485547
```

F1 점수가 비슷한 Precision과 Recall 값을 갖는 분류 모델을 좋아하지만, 이는 항상 우리가 원하는 것은 아니다. 우리는 상황에 따라 Precision과 Recall을 각각을 따로 좀 더 신경써서 다루어야할 때도 있다. 예를들어 아이들의 안전을 위한 비디오 감지시스템을 만든다면, 우리는 아마 많은 좋은 상황의 비디오를 거부할 것이다. (이는 낮은 Recall을 의미한다.) 그렇지만 안전하기를 원한다. (이는 높은 Precision을 의미한다). 즉, 높은 Recall값을 대신해서, 낮은 Recall값과 높은 Precision 값으로 정말 나쁜 상황들을 보여줄 수 있는 상품을 원할 것이다. 반면에, 우리가 가게 좀도둑을 감시하는 분류 모델을 학습시킨다고 가정하자. Recall값이 99%를 가지면서, 우리의 분류모델은 오직 30%만의 Precision 값을 가진다면, 이는 아마 좋은 시스템이다. (당연하다. 감시요원도 조금 실수하지만 거의 대부분 모든 가게 좀도둑을 잡아낸다.)

불행하게도 우리는 이 둘의 값을 동시에 다 올릴수는 없다. Recall값이 높아지면 Precision값이 줄어들고, Recall값이 낮아지면 Precision 값이 높아진다. 이를 Precision과 Recall의 Tradeoff라고 한다.


## **Precision**과 **Recall**의 Tradeoff
이 Tradeoff를 이해하기 위해서, 어떻게 SGD 분류모델이 분류 결정을 내리는지 살펴보자. 각각의 경우에 대해서, *의사결정 함수*에 기반하여 연산을 하고, 만약 점수가 정해놓은 값보다 더 좋다면, 그 경우를 Positive클래스로 할당하게된다. 아래의 그림에서 보면 오른쪽으로는 가장 높은 점수가 배치되어있고, 왼쪽으로는 제일 낮은 점수들이 배치되어있다. 만약 *의사결정 경계선*이 중간 화살표(5 사이에)위치해 있다면, 경계선 기준 오른쪽 4개의 true positive(실제 5)와, 1개의 false Positive(실제로 6)인 것은 발견할 수 있을 것이다. 그러므로 정해놓은 경계선에 따라, precision은 (4/5) 80%지만, Recall은 67% (4/6)이다. 이제 만약 우리가 경계선을 올린다면(오른쪽 화살표), false positive 6은 True negative가 되고, 그러므로, Precision(여기서는 100%)은 증가하지만, 하나의 true positive가 false negative로 바뀌기 때문에, Recall은 50%까지 저하된다. 역으로, 경계선을 왼쪽으로 낮추게 된다면, Recall은 증가하지만, Precision은 줄어들 것이다. 

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-3.png)

Scikit-Learn은 우리에게 직접적으로 경계선(Threshold)를 설정하라고 하지는 않지만, 예측을 하는데 사용하는 의사결정 점수로 접근은 할 수 있다. 분류 모델의 `predict()`함수 대신, `decision_function()`을 호출하는데, 이는 각각의 경우에 대한 점수를 리턴해주고, 우리가 원하는 경계점에 따라서, 예측을 하게된다.
```
>>> y_scores = sgd_clf.decision_function([some_digit])
>>> y_scores
array([ 161855.74572176])
>>> threshold = 0
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([ True], dtype=bool)
```
`SGDClassifier`는 0으로 경계점을 설정해 사용하고, 이전 코드`predict()`와 같은 결과를 도출해낼 것이다. 한번 경계점을 올려보자.
```
>>> threshold = 200000
>>> y_some_digit_pred = (y_scores > threshold)
>>> y_some_digit_pred
array([False], dtype=bool)
```
이렇게 경계점을 올리면 Recall이 감소한다는 것을 확인할 수 있다. 이미지는 실제로 5라고 나타나지만 경계점이 0일 때나 5라고 예측하지 200,000으로 경계점을 올렸더니 예측이 잘 되지 않는다.

그러면 어떻게 경계점을 정할 수 있을까? 우린 처음에 `cross_val_predict()`를 사용해서 학습 데이터세트에 대해 모든 인스턴스의 점수를 받아야한다. 그리고 우리는 이번에 예측 대신에 의사 결정 점수를 반환하는 것을 원한다. 다음의 코드를 실행시켜보자.
```
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```
이제 이 점수를 가지고, 우리는 `precision_recall_curve()`함수를 사용해서 모든 가능한 경계점에 대한 Recall과 Precision을 연산하도록 한다. 
```
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```
마침내, 우리는 Matplotlib을 사용해서 경계점들에 대한 Precision과 Recall 함수를 plot할 수 있게 되었다. 아래 코드와 그림을 참고하라.
```
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot")
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-4.png)

```
아마 위의 그림에서 Recall 곡선보다 Precision 곡선이 더 울퉁불퉁한지 궁금할 수 있다.
왜인지 이해하기 위해선, 이 그림 위의 그림을 참고하라. 중앙에 경계점이 있을 때부터 오른쪽으로
숫자 하나씩 경계점을 옮긴다고 하자. precision은 4/5(80%)에서 3/4(75%)로 내려간다.
반면에, Recall은 경계점이 증가해가도 그냥 쭉 매끄럽게 내려간다. 이가 왜 Recall 곡선이
스무스한지 설명해준다.
```
이제 우리는 간단하게 우리의 일을 위해 최적의 Precision/Recall Tradeoff를 받아 경계값을 쉽게 정할 수 있을 것이다. 또 다른 좋은 Precision/Recall을 선택하는 방법은 Recall에 대한 Precision을 직접적으로 찍어보는 것이다. 아래 그 그림이 있다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-5.png)

이 그래프를 통해, 우리는 Precision이 80%에서 아주 가파르게 감소하는 것을 확인할 수 있다. 우리는 아마 Precision이 떨어지기 전인 60%를 선택할 확률이 높다. 하지만 이 사항은 우리가 하는 프로젝트에 따라 다를 것이다.

이제 우리의 목표가 Precision 90%라고 해보자. 위의 첫번째 Precision & Recall Tradeoff 그래프를 확인해보면 경계점으로 70,000을 사용해야할 것이다. (지금은 학습 데이터 세트로)예측을 내리기 위해, `predict()`함수 대신, 다음의 코드를 실행시켜보자.
```
y_train_pred_90 = (y_scores > 70000)
```
이제 예측한 것들에 대한 Precision과 Recall을 보자.
```
>>> precision_score(y_train_5, y_train_pred_90)
0.86592051164915484
>>> recall_score(y_train_5, y_train_pred_90)
0.69931746910164172
```
좋다. 우리는 90%에 가까운 Precision을 얻었다. 보다시피 사실상 우리가 원하는 어떠한 Precision으로 분류 모델을 만들기가 꽤나 쉽다. 그냥 높게 경계값을 설정해주면 되는 거다. 그렇지만 Precision이 높은 분류모델은 Recall이 너무 낮다면 그렇게 아주 유용한 것은 아니다.


## ROC 곡선
수신자조작특성(*receiver operating characteristic:ROC*) 곡선은 이항 분류에서 자주 사용되는 또다른 도구이다. 이 곡선은 앞에서 봤던 Precision/Recall 곡선과 비슷하지만 이는 precision과 recall을 사용해서 그래프를 그리지만, ROC곡선은 *false positive rate*(FPR)에 대해서 *true positivie rate*(TPR:Recall의 또다른 이름)를 plot한 것이다. FPR은 positive라고 잘못예측한 negative 비율을 말하며. 이는 `1 - true negative rate`의 값과 동일하다. TNR은 특이성(specificity)이라고 한다. 그러므로 ROC 곡선은 (1-sensitivity)에 대해서sencitivity(Recall)를 plot한 것이다. 

ROC 곡선을 그리기 위해서는 `roc_curve()`를 사용해서 다양한 경계값들에 대한 TPR과 FPR을 연산해야한다.
```
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```
그리고나서 Matplotlib을 사용해서 TPR에 대하여 FPR을 plot한다. 아래의 코드가 아래의 ROC 곡선을 만들어줄 것이다.
```
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
save_fig("roc_curve_plot")
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-6.png)

여기서 또다시 Tradeoff가 존재한다. recall(TPR)이 높아질수록, 분류 모델은 더 많은 FPR를 만들어낸다. 점선은 순수하게 랜덤 분류모델의 ROC 곡선을 나타낸다. 좋은 분류모델은 이 점선에서 가능한 제일 멀리 있어야하며 왼쪽 꼭대기로 그래프가 치우쳐지는 경향이 있다. 

분류 모델들을 비교하는 방법은 *Area Under the Cureve*(AUC)를 측정하는 것이다. 완벽한 분류모델은ㅇ ROC AUC가 1이 되어야한다. 반면에 좋지 않은 것들은 random 선인 점선에 가까운 0.5를 가질 것이다. Scikit-Learn에서 ROC AUC를 연산해주는 함수를 제공한다.
```
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```

이제 `RandomForestClassifier`로 학습을 시킨 뒤에 SGD 분류모델과 비교할 수 있도록 ROC 곡선과 ROC AUC를 연산하자. 먼저 학습 데이터 세트의 각 인스턴스들에 대하여 점수를 얻는다. 작동 방식 때문에(Chapter 7을 보라), `RandomForestClassifier` 클래스는 `decision_function()`이라는 메소드가 없다. 대신에 `predict_proba()`메소드가 있다. `predict_proba()`는 주어진 클래스로 주어진 값이 가지는 값으로 표현되며 열은 클래스를, 행은 인스턴스 값을 가진 배열이 반환된다.(예: 이 이미지가 5일 확률은 70%이다.)

```
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
```
ROC 곡선을 만들기 위해선, 확률보다는 점수가 필요하다. Positive 클래스에 대한 확률을 점수로 사용할 수 있도록 하는 코드가 아래에 있다.
```
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
```
이제 ROC 곡선을 그릴 준비가 되었다. 아래의 코드를 사용해서 그려보자.
```
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot")
plt.show()
```

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/3-7.png)

위의 그림을 보다시피, `RandomForestClassifier`가 `SGDClassifier`보다 좀 더 왼쪽 끝으로 가까운 곡선형태를 보이며 좋은결과를 보여주고 있으며 ROC AUC 점수도 더 좋다.
```
>>> roc_auc_score(y_train_5, y_scores_forest)
0.99312433660038291
```
Precision과 Recall점수도 한번 측정해보자 : 98.5% Precision과 82.8% Recall값을 가지고 있는 것을 알 수 있다. 나쁘지 않다!
```
#Precision Score
>>> y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
>>> precision_score(y_train_5, y_train_pred_forest)
0.98529734474434938

#Recall Score
>>> recall_score(y_train_5, y_train_pred_forest)
0.82826046854823832
```
이제 이항분류모델을 어떻게 학습시키는지 배웠고, Cross-validation을 사용해 분류모델을 평가하고 우리의 것에 잘 맞는 적절한 측정법을 고를수 있으며, 우리 요구에 잘 맞는 Precision/Recall Tradeoff를 선택할 수 있으며, 다양한 학습 모델을 비교하기 위해 ROC 곡선을 그리고, ROC AUC 점수를 얻을 수 있다. 이제 5 하나 예측하는 것보다 좀 더 많은 것을 예측하는 모델을 만들어보자.

# 다중 클래스 분류
이항 분류는 클래스가 2개만 있는 반면에, *다중 클래스 분류모델*(흔히 다항 분류 모델 *multinomial classifier*라고도 한다)은 2개 이상의 많은 클래스를 구분해낼 수 있다.

랜덤 포레스트 분류 모델이나 나이브 베이즈 분류모델 같은 몇몇 알고리즘들은 직접적으로 다중 클래스를 다루는 능력이 있다. 그 외 다른 것들 예를들면 Support Vector machine이나 선형회귀모델 같은 것들은 엄밀히 말하면 이항 분류이다. 그렇지만 다중 이항 분류를 사용해서 다중 클래스 분류를 수행해주는 전략을 사용할 수 있다.

예를들어 MNIST데이터 셋을 숫자 0~9로 10개의 클래스를 분류해야하는데 1-분류기, 2-분류기를 각각 모두 각 클래스에 대한 분류기를 이항분류로 만들어주어 하나의 이미지에 대해서 열 개의 분류기를 돌려보고 해당 클래스의 Positive의 제일 높은 점수를 가진 것을 답으로 사용하는 방법이 있다. 이를 일대다(one versus all: OvA)전략이라고 한다.

또 다른 방법으로는 모든 클래스에 대해 비교하는 것이다. 다시말하면 0과 1, 0과2, 0,3이런식으로 두개로 분류하는 방식을 하면서 클래스를 분류하는 것이다. 이를 일대일 (one versus one : OvO) 방식이다. 각 두개의 클래스로 구분만 하면 되기때문에 모든 클래스를 다 학습할 필요가 없다는 장점이 있다. 

몇몇 (Support Vector machine같은)알고리즘들은 학습 데이터 세트의 크기에 영향을 많이 받는데, 커다란 데이세트에 대해서 몇가지의 클래스를 동시에 올리는 것보다 작게 데이터 세트를 쪼개서 학습시키는 것이 더 빠르기 때문에 OvO가 선호된다. 하지만 대부분의 다중 이항분류의 경우에선 OvA가 선호된다.

Scikit-Learn은 다중 클래스 분류를 이항 분류법으로 사용할 때를 감지해서, 자동으로 OvA를 실행시킨다.(SVM같은 경우는 자동으로, OvO로 설정된다.) `SGDClassfier`로 한번 해보자
```
>>> sgd_clf.fit(X_train, y_train)
>>> sgd_clf.predict([some_digit])
array([ 5.])
```
아주 쉽다! 이 코드는 타켓 클래스 (y_train_5)로 설정해 5대다 분류 대신, 원래의 타겟 클래스 0~9까지(y_train)로 학습데이터에 대해서 SGD 분류 모델을 학습시킨다. 내막에는 사실 Scikit-Learn이 10개의 이항 분류기를 만들어내서 각각에 대한 이미지 점수를 받아와서 제일 점수가 높은 것을 출력해주는 것이다.

이제 클래스 값 하나에 대한 점수만 받아오는게 아닌 모든 클래스에 대해서 점수를 받아와보자. 
```
>>> some_digit_scores = sgd_clf.decision_function([some_digit])
>>> some_digit_scores
array([[-311402.62954431, -363517.28355739, -446449.5306454 ,
        -183226.61023518, -414337.15339485,  161855.74572176,
        -452576.39616343, -471957.14962573, -518542.33997148,
        -536774.63961222]])
```
가장 높은 점수는 5로 출력된다.
```
>>> np.argmax(some_digit_scores)
5
>>> sgd_clf.classes_
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
>>> sgd_clf.classes_[5]
5.0
```
분류 모델이 학습되고 나면, `class_`라는 속성에 타겟 클래스 리스트로 저장되며 값의 순서대로 정렬된다. 이 경우에선, 각각 클래스가 숫자 순서대로 정렬이 될 것이다. 만약 Scikit-Learn을 OvO나 OvA로 바꾸어서 진행시키고 싶다면, `OneVsOneClassifier` 혹은 `OneVsRestClassifier`를 사용할 수 있다. 예로, 아래의 코드는 OvO전략을 이용해서 다중 분류 모델을 만든다.
```
>>> from sklearn.multiclass import OneVsOneClassifier
>>> ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
>>> ovo_clf.fit(X_train, y_train)
>>> ovo_clf.predict([some_digit])
array([ 5.])
>>> len(ovo_clf.estimators_)
45
```
`RandomForestClassifier`로 학습 시키는 것도 아주 간단하다.
```
>>> forest_clf.fit(X_train, y_train)
>>> forest_clf.predict([some_digit])
array([ 5.])
```
이번에 Scikit-Learn에서는 OvA나 OvO전략을 사용할 필요가 없다. 랜덤 포레스트 분류 모델이 직접적으로 다중 클래스로 인스턴스를 분류하기 때문이다. `predict_proba()`를 사용하여 분류모델이 각각의 클래스에 대해 각각의 인스턴스에 할당한 확률 리스트를 얻을 수 있다.
```
>>> forest_clf.predict_proba([some_digit])
array([[ 0.1,  0. ,  0. ,  0.1,  0. ,  0.8,  0. ,  0. ,  0. ,  0. ]])
```
5번째에 있는 인덱스의 값이 0.8, 즉 80%로 가장 높게 나타나며 이는 주어진 이미지가 5라고 하는 것이다.
이제 우리의 분류 모델 성능을 평가해보고 싶을 것이다. 보통, 우리는 Cross-validation을 사용한다. `cross_val_score()`함수를 사용해서 `SGDClassifier`의 정확도를 측정해보자
```
>>> cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
array([ 0.84063187,  0.84899245,  0.86652998])
```
모든 테스트 fold에 대해 약 84%가 넘는 결과를 보인다. 무작위 분류 모델을 사용했다면 정확도는 약 10%정도였을 것이고 이는 그렇게 아주 나쁜 결과는 아니지만 우리는 더 잘할 수 있다. 예를들어 (Chapter2에서 다루었던) 입력값의 사이즈를 키우면 정확도는 90%이상 증가한다.
```
>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
>>> cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
array([ 0.91011798,  0.90874544,  0.906636  ])
```  

# 에러분석
당연히 이것이 실제 프로젝트였다면, 우린 기계학습 프로젝트 체크 리스트에 있는 단계를 따라야했을 것이다. 이전 Chapter에서 우리가 했던 것처럼, 데이터 준비, 여러가지 모델 테스트, GridSearchCV를 사용한 잘 학습된 하이퍼 파라미터 값 및 좋은 학습모델 찾기, 이 전체과정 가능한 자동화하기였다. 이제 우리는 유망한 학습 모델을 찾았고, 이를 향상시킬 방법을 찾고있다고 가정해보자. 이를 위한 방법은 이 학습 모델이 만드는 에러의 형태를 분석하는 것이다. 

먼저 혼동 행렬을 보자. `cross_val_predict()`함수를 사용해서 예측 결과를 먼저 내고, 그다음에 `confusion_matrix()`를 전에서 한 것처럼 호출해보면 된다.
```
>>> y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
>>> conf_mx = confusion_matrix(y_train, y_train_pred)
>>> conf_mx
array([[5725,    3,   24,    9,   10,   49,   50,   10,   39,    4],
       [   2, 6493,   43,   25,    7,   40,    5,   10,  109,    8],
       [  51,   41, 5321,  104,   89,   26,   87,   60,  166,   13],
       [  47,   46,  141, 5342,    1,  231,   40,   50,  141,   92],
       [  19,   29,   41,   10, 5366,    9,   56,   37,   86,  189],
       [  73,   45,   36,  193,   64, 4582,  111,   30,  193,   94],
       [  29,   34,   44,    2,   42,   85, 5627,   10,   45,    0],
       [  25,   24,   74,   32,   54,   12,    6, 5787,   15,  236],
       [  52,  161,   73,  156,   10,  163,   61,   25, 5027,  123],
       [  43,   35,   26,   92,  178,   28,    2,  223,   82, 5240]])
```
숫자가 아주 많다. 이를 한번 matplotlib에 있는`matshow()`를 사용하여 혼동 행렬을 이미지로 바꾸어보자
```
plt.matshow(conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_plot", tight_layout=False)
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-2.png)

이 혼동 행렬은 꽤 좋아보인다. 대각선으로 아주 높은 값들을 가지고 있는데 이는 분류가 올바르게 되었다는 것을 의미한다. 5가 다른 숫자들 보다 조금 더 어두워보인다. 이는 5의 이미지 데이터 수가 상대적으로 부족해서 그런 것일수도 있고, 다른 숫자들에 비해 5의 분류가 잘 되지 못한 것일 수도 있다. 사실 이는 이 두 경우 모두에 해당된다고 볼 수 있다.

에러 값에 대한 plot에 집중해보자. 먼저 각각 상응하는 클래스에 의미지의 숫자로 혼동행렬 안에 각각의 값을 나눌 필요가 있다. 그래서 우리가 에러의 절댓값대신 에러율과 비교할 수 있다. 
```
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```
이제 그림상의 대각선 값들은 0으로 초기화 해주고 결과를 출력해보자.
```
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-3.png)

이제 우리는 분명하게 분류 모델이 만들어내는 에러의 종류를 볼 수 있다. 행은 실제 클래스를 의미하고, 열은 예측된 클래스임을 기억하자. 행 8번째와 9번째를 보면 비교적 밝은 것을 확인할 수 있는데 이는 수많은 이미지들이 잘못 분류되었다는 것이다. 역으로 첫번째 행 같은 경우는 어두운 부분이 많은데 이는 분류가 아주 잘되었다는 것을 믜미한다. 위의 그림에서 보여지는 값을 완벽한 대칭을 이루지는 않는다.

놀라운 혼동 행렬은 우리의 분류 모델을 향상 시키도록 통찰력을 줄 수도 있다. 이 plot을 보면 8하고 9를 분류는 3하고 5의 분류처럼 향상시키는데 많은 시간을 요구할 것이다. 예를들어 데이터를 더 모아 학습할 데이터량을 키우거나, 이를 서로 분별해줄 수 있는 특징 값들 새로이 발견해 사용해야한다. 아니면 이미지 전처리를 하여 이미지 패턴을 더 강조시키는 방법도 있을 것이다.

개별적 에러 출력은 우리의 학습모델이 무엇을 하는지, 왜 실패하는지에 대하여 통찰력을 얻는데 아주 좋다. 하지만 좀 더 어렵고 시간이 많이든다. 예를들어, Matplotlib의 `imshow()`함수를 사용하여 3과 5의 이미지를 찍어보자. 
```
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
save_fig("error_analysis_digits_plot")
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-4.png)

왼쪽의 있는 이미지는 3이라고 분류된 것들이고, 오른쪽에 있는 이미지들은 5라고 분류된 것들이다. 왼쪽 아래 이미지와 오른쪽 위의 이미지는 잘못 분류된 것이다. 그리고 사람들이 숫자를 너무 나쁘게 써서 분류가 잘 안된 것들도 있다. 하지만 잘못 분류된 사진들을 보면 분명한 에러는 우리에게 있어보이지만, 왜 분류 모델이 실수를 하는지 이해하기 어렵다. 이유는 우리가 간단한 SGD모델을 사용했기 때문인데, 이는 선형 모델이기 때문이다. 이 알고리즘이 한 일은 각 픽셀에 대해 가중치를 준 것이고, 새로운 이미지가 들어오면 각각의 클래스에 대한 점수를 얻기위해 가중치가 적용된 픽셀 강도를 더하는 것이다. 그런데 3과 5는 오직 몇개의 픽셀로만 분류가 되는 것이라 분류모델이 쉽게 혼동할만하다. 

3과 5의 다른점이라고는 상위 부분에 조인트가 꺾이는 방법인데 이를 잘못써서 불분명하게 꺾으면 잘못 예측할 가능성이 있다는 것이다. 즉 이는 학습모델이 이미지 이동이나 회전에 대해 아주 민감하다는 것을 의미핟ㄴ다. 그래서 3과 5를 혼동하는 것을 줄여주는 방법은 너무 회전되지 않고 잘 중앙에 배치가 되게끔 이미지 전처리를 해야하는 것이다. 이는 아마 다른 에러를 줄이는 데에도 많은 도움이 될 것이다.

# 다중 레이블 분류
이제까지 우리는 하나의 인스턴스를 하나의 클래스로 분류해보았다. 일부의 경우에서는 우린 각 인스턴스에 대해 분류모델이 클래스를 여러개 적용해서 결과를 출력하도록 하고 싶을지 모른다. 예로 얼굴 인식을 하고자 하는데 만약 한 사진에 여러명이 있는 경우라면 어떻게 해야할까? 철수, 영희, 민수가 있다고 해보자. 어떤 사진에 철수와 민수만 나와있다고 한다면, 결과는 [1,0,1]이 되어야 할 것이다. 이렇게 0과 1로 다중 레이블 결과를 출력하는 분류 시스템을 *다중 레이블 분류(Multilabel classification)*시스템이라고 한다.

우리는 아직 얼굴 인식을 진행하지는 않을 것이다. 좀 더 간단한 예시를 보자. 
```
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```
위의 코드는 각각의 숫자 이미지에 대해서 두 개의 레이블이 붙은 `y_multilabel` 배열을 만든다. 먼저 숫자가 7이상의 값인지에 대해서 레이블을 매긴 것이고 두번째는 숫자가 홀수인지 짝수인지를 나타내도록 레이블을 매긴것이다. 다음 줄은 다중 레이블 분류를 해줄 수 있는 `KNeighborsClassifier` 라는 인스턴스를 만들어 주고 다중 타겟 배열을 사용해 모델을 학습시킨다. 우리는 이제 예측을 해볼 수 있다. 그러면 결과로 두개의 클래스가 나오는 것을 확인할 수 있다.
```
>>> knn_clf.predict([some_digit])
array([[False,  True]], dtype=bool)
```
결과가 잘 나왔다! 숫자 5는 실제로 크지 않으며(7>, False) 홀수(True)이다. 

다중 레이블 분류를 평가하는데는 많은 방법이 있으며, 프로젝트에 따라서 적당한 측정법을 골라야한다. 예를들어 한가지 접근법이 각각의 개별적 레이블에 대해서 F1 점수를 측정하는 것이다. (혹은 앞에서 다루었던 다른 어떤 이항 분류 모델 측정법을 사용할 수 있다) 그리고 나서 평균 점수를 계산한다. 아래의 코드는 모든 레이블에 대해서 F1 평균 점수를 연산하는 것이다.
```
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```
이는 모든 레이블이 똑같이 중요해야만한다. 특히, 만약 영희, 민수의 사진보다 철수의 사진을 더 많이 가지고 있다면, 우리는 철수의 사진에 대해서 분류 모델의 점수에 가중치를 더 높게 줄 것이다. 하나의 선택 사항으로는 각각의 레이블에 그 레이블의 *support*를 해주는 가중치를 동등하게 주는 것이다. 이를 해주기 위해서는 `arverage = "weighted"`라고 위에 실행될 코드에 수정해주면 된다.
 
# 다중 결과 출력 분류
우리가 마지막으로 다루어볼 분류의 마지막 타입은 *다중 클래스 다중 결과 출력 (Multioutput-Multiclass)분류*라고 불리는 것이다. 이는 각각의 레이블이 다중 클래스가 될 수 있는 다중 레이블 분류의 일반화이다. (즉 가능한 값을 두개 이상 가지는 것이다)

이를 설명하기 위해서 이미지에서 노이즈를 제거하는 시스템을 만들어보자. 노이즈를 가진 숫자 이미지를 입력으로 받아서, MNIST데이터 처럼 픽셀 강도의 배열로 나타나지는 노이즈가 제거된 깨끗한 이미지를 출력할 것이다. 분류 모델의 출력은 다중 레이블로 출력될 것이며 각각의 레이블을 여러가지의 값을 가질 수 있다.(0~255픽셀 강도 범위 내) 그래서 다중 결과 출력 분류 시스템의 예시이다.
```
이러한 예시에서는 분류와 회귀 사이의 선은 때때로 모호하다. 틀림없이 픽셀의 강도를 예측하는 것은 좀 더 
분류보다는 회귀와 유사하다. 좀 더 나아가 다중 결과 출력 시스템은 분류 업무에만 한정되어 있는 것이 아니다.
우리는 두 개 모두의 레이블과 값의 레이블을 포함한 인스턴스당 다중레이블를 결과로 출력하는 시스템을 만들 수
도 있다.
```
MNIST 학습 데이터 셋에서 이미지를 하나 가져와서 NumPy의 `randint()` 함수를 사용해서 픽셀 강도의 노이즈를 추가해보자. 타겟 이미지는 원본 이미지가 될 것이다.

```
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
```
이제 테스트셋에서 이미지를 가져와보자.
```
some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-5.png)

왼쪽에는 노이즈가 있는 입력 이미지이고, 오른쪽에는 깨끗한 타겟 이미지가 있다. 이제 학습을 시키고 이미지를 깨끗하게 만들어보자
```
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/03/Code3-6.png)

목표 이미지와 충분히 비슷해졌다! 이것이 우리 분류에대한 공부 여정의 결과이다. 우리는 분류에 좋은 측정법을 어땋게 골라야하는지 알고 있으며, 적절한 Precision/Recall tradeoff를 고를 수 있고, 분류 모델을 비교할 수 있으며, 좀 더 아낭가 다양한 분야에 좋은 분류 시스템을 만들 수 있을 것이다.

**[뒤로 돌아가기](https://github.com/Hahnnz/handson_ml-Kor/)**
