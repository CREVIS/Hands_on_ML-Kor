=============**미완성**=============
======



Chapter 4. 학습 모델
===========
우리는 기계학습모델에 대하여 다루어보았고 이에대한 학습 알고리즘은 거의 마치 기능은 알지만 작동 원리를 이해할 수 없는 복잡한 기계 장치, 블랙박스 같다.만약 이전 Chapter들의 연습문제를 풀어보았다면 그 안의 함수 내막을 제대로 알지 못하더라도 기계학습 프로젝트를 수행할 수 있었던 점에서 놀랐을 것이다. 우리는 여태까지 회귀 모델을 최적화해보았고 숫자 이미지 분류 모델 알고리즘을 발전 시켜 보았으며, 아무런 기반지식 없이 스팸 분류 모델도 구현해보았다. 실제로 어떻게 작동되는지도 모르는 채 말이다. 실제로 많은 상황에서 우리는 세부 구현사항까지 알 필요는 없다.

그렇지만 어떤 함수가 어떻게 동작하는지에 대하여 충분히 이해하고 있는 것은 우리를 사용에 올바르고 적절한 학습 모델로, 그리고 우리의 일에 잘 맞는 훌륭한 하이퍼 파라미터 세팅을 빠르게 나아갈 수 있게 해준다. 내부 함수가 어떻게 되어있는지 이해하는 것은 또한 디버깅 문제에도 도움이 되며, 좀 더 효과적인 에러 분석을 수행하는데 도움을 준다. 마지막으로 이 Chapter에서 얘기할 대부분의 내용들은 신경망(Neural Network)에 대한 학습 및 구현에 대한 이해가 필수적이다. (이 책의 제 2부에서 다룬다)

이 Chapter에서는 우리는 여기서 가장 간단한 모델인 선형회귀모델을 살펴보는 것부터 시작할 것이고, 우리는 이를 학습시키는 두 가지의 서로 매우 다른 방법에 대해서 이야기를 해볼 것이다.

* 학습 데이터 세트에 모델이 최적의 fitting을 가지는 학습 모델의 파라미터를 직접적으로 연산하는 **직접적인 "폐쇄적 형태"의 수학 방정식 사용하기**
   * EX: 학습 데이터 세트에 대해서 손실 함수 값을 최소화하는 모델 파라미터 값
* 학습 데이터 세트에 대해 손실 함수를 최적화하기 위해 점진적으로 모델 파라미터를 조절하여, 결국에는 위의 방법과 파라미터 세팅이 같아지게 되는 **경사 하강법 (Gradient Descent:GD)라고 불리는 반복적으로 최적화를 수행하는 접근법 사용하기** 우리는 나중에 이책의 제 2부에서 신경망에 대해 공부할 때 나오는 몇가지 다양한 경사하강법인 Batch GD, Mini-batch GD, Stochastic GD를 살펴볼 것이다.

그 다음에는 비선형 데이터 세트를 fitting할 수 있는 좀 더 복잡한 모델인 다항 회귀 모델을 살펴볼 것이다. 이 모델은 선형 회귀 모델보다 더 많은 파라미터를 가지고 있기에, 학습 데이터를 과하게 학습하기 쉬워서(Overfitting) 우리는 학습 곡선(Learning Curve)를 사용해서 어떻게 Overfitting이 발생하는 경우를 찾아내는 지 살펴볼 것이며, 그리고 나서 우리는 학습 데이터 세트를 과하게 학습하게 되는 위험을 줄일 몇가지의 Regularization 방법을 살펴볼 것이다.
```
이 Chapter에서는 선형대수와 미적분학의 기본개념을 사용하여 수학적 방정식을 꽤 많이 사용할 것이다. 이러한
방정식들을 이해하기 위해서는 벡터와 행렬이 무엇인지 알 필요가 있고 이들을 변환하는 방법과 내적이 무엇인지,
역행렬은 무엇이고, 편도함수가 무엇인지에 대해 알고있어야한다. 만약 이러한 개념들과 친숙하지 않다면, 이
레포지토리에서 제공하는 온라인 상으로 추가적으로 제공되는 자료로 선형대수와 미적분학을 소개해주는 Jupyter
Notebook 튜토리얼을 보고 공부를 해야한다. 만약 수학에 알레르기가 있는 사람이라면 여기서 나오는 방정식들을
가뿐히 넘어가도 좋다.
``` 

# 선형 회귀 모델 (Linear Regression)
Chapter 1에서, 우리는 간단한 회귀 학습 모델![](https://render.githubusercontent.com/render/math?math=%5Ctext%7Blife_satisfaction%7D%20%3D%20%5Ctheta_0%20%2B%20%5Ctheta_1%20%5Ctimes%20%5Ctext%7BGDP_per_capita%7D&mode=inline) 을 본 적이 있다.

이 모델은 입력 받는 특징으로 `GDP_per_capita`를 받는 선형 함수이다. θ1과 θ2는 모델의 파라미터이다. 

좀 더 일반적으로, 선형 학습 모델은 입력받는 특징에대하여 가중치의 합을 연산하여 예측결과를 낸다. 더해서 아래의 4-1에서 나타나는 상수를 *bias_term*, 편향도라고 하며, 이를 *intercept term*, 절편이라고도 한다. 

###### Equation 4-1. 선형 회귀 모델 예측 함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-1.png)

* **ŷ** : 예측된 값
* **n** : 입력받는 특징의 수
* **xi** : i번째의 특징 값
* **θj** : j번째의 모델 파라미터

이는 아래의 보여지는 방정식처럼 벡터형태로 좀 더 간결하게 쓰여질 수 있다.

###### Equation 4-2. 선형 회귀 모델 예측 함수 **[벡터형태]**
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-2.png)

* **θ** : 모델 파라미터 벡터
* **θ^T** : 모델 파라미터 벡터의 전치 행렬
* **x** : 인스턴스의 특징 벡터 값
* **θT*x** : θT와 x의 내적 값
* **hθ** : 모델 파라미터 θ를 사용한 연산하는 추측 함수

좋다. 이것이 바로 선형 회귀 모델인데, 이제 어떻게 학습을 시킬까? 모델을 학습시켜 모델의 파라미터를 정할 수 있게 재현해서 학습 데이터에 대해 모델의 최적 fitting을 찾아보자. 이를 위하여, 먼저 학습 모델이 학습 데이터에 대해 얼마나 잘 학습 되었는지 평가할 수 있어야한다. chapter2에서, 우리는 회귀 학습 모델의 가장 흔한 모델 평가방법으로 평균 제곱근 오차기법(Root Mean Square Error: RMSE)([Equation2-1](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-2.-End-to-end-Machine-Learning-project#equation-2-1-root-mean-square-errorrmse%ED%8F%89%EA%B7%A0-%EC%A0%9C%EA%B3%B1%EA%B7%BC-%EC%98%A4%EC%B0%A8))을 본 적이 있을 것이다. 그러므로 선형 회귀 모델을 학습 시키기 위해서 RMSE를 최소화 해줄 수 있는 θ값을 찾아야만 한다. 특히 RMSE보다는 MSE(Mean Square Error)를 최소화 하는 것이 더 간단하며, 같은 결과를 출력한다. 

학습 데이터 세트 X에 대하여 선형 회귀 모델의 예측 값 hθ의 MSE는 아래의 방정식으로 계산이 이루어진다.

###### Equation 4-3. 선형 회귀 학습 모델에 대한 MSE 손실 함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-3.png)

이러한 개념의 대부분은 [Chapter2에서 소개](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-2.-End-to-end-Machine-Learning-project#%EC%82%AC%EC%9A%A9%ED%95%A0-%EC%88%98%ED%96%89-%EC%B8%A1%EC%A0%95%EC%B9%98%EB%A5%BC-%EC%A0%95%ED%95%98%EC%9E%90)가 되었다. 다른 점이라고는 평범한 h대신에 hθ가 사용되었는데 이는 모델 파라미터가 벡터 값 θ로 바뀌었기에 이와같이 표현한 것이다. 표기법을 간소화하기 위해서, 우린 MSE(X,hθ)대신에 MSE(θ)로 바꾸어 사용한다.

# 정규 방정식 (normal equation)
손실 함수 값을 최소화하는 θ값을 찾기위해, 폐쇄적 형태의 솔루션, 즉 다른말로, 바로 답을 줄 수 있는 수학 방정식을 사용할 것이며, 이를 정규 방정식이라고 한다.

###### Equation 4-4. 정규 방정식
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-4.png)

* **θ̂** : 손실 함수 값을 최소화하는 θ 값
* **y** : y^(1)부터 y^(i)까지를 포함한 타겟 값의 벡터 값

아래의 그림은 이 방정식을 테스트하기 위해 선형으로 보이는 데이터를 만들어본 것이다.
```
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-1.png)

이제 정형 방정식을 사용하여 θ̂를 계산해보자. 우리는 역행렬을 연산하기 위해서 NumPy의 선형대수 모듈`(np.linalg)`에 있는 `inv()`라는 함수를 사용해볼 것이다. 그리고 `dot()`이라는 함수를 이용해서 행렬 내적을 연산하자.
```
X_b = np.c_[np.ones((100, 1)), X]  # 각각의 인스턴스에 x0 = 1 추가
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```
데이터를 생성해낼 때 자주 사용하는 함수는 `y = 4 + 3x_1 + 가우시안 잡음`이다. 이제 공식이 어떤 값이 좋다고 하는지 보자.
```
>>> theta_best
array([[ 4.21509616],
       [ 2.77011339]])
```
우린 θ_0=4.21509616과 θ_1=2.77011339 대신에 θ_0=4와 θ_1=3가 나오기를 원했다. 충분히 비슷하지만, 노이즈가 기존 함수의 정확한 파라미터 값을 찾는 것이 불가능하게 했다. 

이제 θ̂를 사용해서 예측값을 연산해보자.
```
>>> X_new = np.array([[0], [2]])
>>> X_new_b = np.c_[np.ones((2, 1)), X_new]  # 각각의 인스턴스에 x0 = 1 추가
>>> y_predict = X_new_b.dot(theta_best)
>>> y_predict
array([[ 4.21509616],
       [ 9.75532293]])
```
이 모델의 예측값을 그려보자.
```
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-2.png)

Scikit-Learn을 사용해서도 위와 같은 결과를 얻을 수 있다.
```
>>> from sklearn.linear_model import LinearRegression
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([ 4.21509616]), array([[ 2.77011339]]))
>>> lin_reg.predict(X_new)
array([[ 4.21509616],
       [ 9.75532293]])
```

# 연산 복잡도
정형 방정식은 역행렬 X^T⋅X를 연산하는데, 이는 n X n 행렬이라는 것이다. 이런 뒤집힌 행렬의 *연산 복잡도*는 전형적으로 최소 약 `O(n^2.4)`에서, 최대 `O(n^3)`까지 나타난다. (어떻게 구현을 하였는지에 따라 다름) 
```
정형 방정식은 특징의 수가 너무 많을 경우 엄청 느려진다.
```
장점은 이 방정식이 학습 데이터의 인스턴스 수에 대해서는 선형이라는 것이다. (O(m)임) 

또한 우리의 선형 회귀 모델을 (정형 방정식이나 여느 다른 알고리즘을 사용해서) 학습시키고 나면, 예측 연산은 아주 빠르다는 것이다. 예측하고 싶은 인스턴스의 수와 특징의 수 모두에 대해서 선형적 연산 복잡도를 보인다.

이제 우리는 선형 회귀 모델을 학습시키는 또다른 방법이지만 방식이 아주 다른 방법을 살펴볼 것이다. 특징의 수가 너무 많은 경우나 주어진 메모리 자원으로 학습을 시키기 힘들때 아주 좋다.

# 경사 하강법(Gradient Descent)
경사 하강법은 넓은 범위의 문제에 최적의 솔루션을 찾아주는 능력을 가진 매우 포괄적인 알고리즘이다. 경사 하강법은 손실 함수 값을 최소화하기 위해서 반복적으로 파라미터 값을 조절해보는 것이다. 

예로 들면 우리가 산 속에 조난을 당했는데 안개까지 껴 있다고 해보자. 가장 이 계곡을 빠르게 내려갈 수 있는 전략은 바로 제일 가파르고 제일 지상과 가까워 보이는 경사로 내려가는 것이다. 이것이 바로 경사 하강법이다. 벡터 θ 파라미터에 대해 손실 함수 값의 경사도를 측정핵서 그 방향으로 경사도를 낮추면서 내려오는 것이다. 경사값(기울기)가 0이 되는 지점이 바로 지상인 최소값에 도달한 것이다.

구체적으로, 랜덤 값으로 θ값을 채워넣으면서 시작을 한다.(이를 랜덤 초기화라고 한다.) (MSE 같은) 손실 함수를 매 스탭마다 줄여나가면서 한번에 작게 한걸음씩 이 값이 최소값에 도달할 때까지 점점 개선될 것이다. 

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-3.png)

경사 하강법에서 중요한 파라미터는 스탭, 즉 학습률(Learning rate)라고 하는 파라미터(한번에 얼마나 움직일 것인지)의 크기이다. 학습률이 너무 작다면 알고리즘의 반복횟수는 늘어날 수 밖에 없으며 이는 많은 시간을 소모한다는 것이다. (아래의 그림 참고)

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-4.png)

반면에 학습률이 너무 높으면, 최소점에 도달하기 어렵고 심지어 너무 높은 탓에 오히려 값이 증가하는 경향도 있다. (아래의 그림 참고)

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-5.png)

마지막으로 모든 손실 함수에 잘 맞아 떨어지는 정형적인 경우는 없다. 어떤 경우는 손실 함수 그래프에 능선이 여러게 존재하거나 고원이 하나 있는 형태로 이루어져 있을 수도 있다. 즉 그 손실 함수에서 최고로 낮은 곳에 도달하는 일은 아주 어렵다. 이 점은 전역 최솟값(Global minimum)라고 하며 아래서 보여지는 것 처럼 능선 사이의 최저점을 지역 최소점(local minimum)이라고 한다. 고원 형태지점에서 시작하게 될 경우 최저점에 도달하는데 시간이 꽤 걸릴 것이고, 지역 최저점에 빠져버린다면 전역 최저점에 도달할 수 없다. (아래 그림 참고)

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-6.png)

다행히도 선형 회귀 모델에 대한 MSE 손실 함수는 볼록한 형태의 함수다. 어느 지점을 잡더라도 내려갈 수 있는 곳은 단 한군데 뿐이며, 이는 전역 최저값이 하나뿐이라는 것이다. 그리고 또한 절대 갑자기 변하는 기울기가 아닌 연속 함수이다. 즉 학습률을 나쁘게 잡지만 않는다면 어쨌거나 손실 함수의 전역 최저점에 도달할 수밖에 없다.   

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-7.png)

왼쪽에 있는 그림은 바로 쭉쭉 내려갈 수 있어서 빠르게 전역 최저점을 찾을 수 있다. 하지만 왼쪽은 수직의 형태를 가진 손실 함수로 시간은 오래 걸리지만 어쨌든 전역 최저점을 찾아간다. 
```
경사 하강법을 이용할 때에는 반드시 모든 특징값들이 비슷한 크기여야만 한다. 아니면 전역 최저점에
찾아가는 시간이 너무 오래걸린다. (Scikit-Learn의 StandardScaler 클래스를 사용하자)
```
위의 다이어그램은 모델을 학습시킨다는 것은 손실 함수 값을 최소화해주는 모델 파라미터들의 조합을 찾는 과정을 뜻한다는 것을 알려준다. 모델 파라미터 공간을 탐색하는 것인데, 이는 모델이 파라미터수를 많이 가질수록, 공간이 가지는 차원의 수가 증가하는 것을 알아두자. 

## 배치 경사 하강법 (Batch Gradient Descent)
경사 하강법을 구현하기 위해서는, 각 모델의 파라미터 값 θj에 대한 손실 함수 값의 기울기를 연산해야한다. 다른말로 θj를 조금씩 바꿀때마다 손실 함수 값이 얼마나 변하는지 알아야한다는 것이다. 이를 편도함수(*partial derivative*)라고 한다. 이는 "내가 지금 동쪽으로 간다면 그 방향의 기울기는 얼마나되나요?"라고 묻는 것과 같은 것이다. 아래의 공식 4-5는 ![](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_j%7D%20%5Ctext%7BMSE%7D%28%5Cmathbf%7B%5Ctheta%7D%29&mode=inline)라고 쓰여지며, 파라미터의 θj에 대해서 손실함수의 편도함수를 계산한다.
###### Equation 4-5. 손실함수의 편도함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-5.png)

개별적으로 이러한 편도함수를 연산하지않고, 우리는 한번에 공식 4-6으로 이를 전부 연산할 수 있다. ∇θMSE(θ)이라고 쓰여지는 기울기 벡터는 (각 모델 파라미터에 대한)손실 함수의 모든 편도함수값을 가지고 있다.

###### Equation 4-6. 손실함수의 기울기 벡터
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-6.png)

```
이 공식은 각 경사하강법 스탭에 전체 학습 데이터 세트 X에 대한 연산을 수반하고 있다. 이가 왜 이
알고리즘이 "배치 경사 하강법"이라고 하는지 알 수 있게 해준다. 매 스탭 마다 학습데이터 전체를
사용하기 때문인데, 결국 이는 학습데이터가 엄청 커지면 느려진다는 것을 의미한다. (하지만 우리는 
곧 좀 더 빠른 경사 하강법을 보게될 것이다.) 
```
윗방향을 가르키고 있는 기울기 벡터값을 얻고나면, 그 반대 방향으로 내려간다. 이는 θ에서 ∇θMSE를 빼는 것을 의미한다. 여기에 학습률 η이 작용되는 곳이다. 한 스탭으로 얼마나 내려갈지 정하기 위해 η으로 기울기 벡터를 곱한다.

###### Equation 4-7. 경사 하강법 스탭
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-7.png)

이제 이 알고리즘을 구현해보자
```
eta = 0.1 #학습률
n_iterations = 1000 #반복 횟수
m = 100 #입력받는 특징수
theta = np.random.randn(2,1) # 파라미터 값

#정해준 반복횟수 만큼 배치경사하강법 수행하기
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```
그렇게 어렵지 않다! 이제 한번 θ값을 확인해보자.
```
>>> theta
array([[ 4.21509616],
       [ 2.77011339]])
```
정형 방정식에서 찾았던 값과 완전 일치한다! 경사하강법이 제대로 작동했다! 하지만 전혀 다른 학습률`eta`를 사용한다면 어떨까? 아래의 그림은 3가지의 서로 다른 값으 학습률로 처음 10번의 스탭에 대한 결과를 출력한 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-8.png)

왼쪽에 있는 그림은 학습률이 너무 낮아서 제대로 도달하지 못했다. 결국엔 도달하겠지만 시간이 오래 걸릴것이다. 가운데 것은 아주 적당한데 10번밖에 수행하지 않았는데도 벌써 충분히 가까워졌다. 하지만 오른쪽 그림은 학습률이 너무 높은탓에, 이미 너무 높은 곳에 있으며, 스탭이 진행될 때마다 더욱 증가할 것이다.

그렇다면 이제 반복 횟수는 얼마나 설정해야하는가 물어보는 사람도 있을 것이다. 너무 적다면 도달도 못하고 끝날 것이고, 너무 많다면 시간낭비를 할 수도 있다. 좋은 방법은 충분히 많이 반복 횟수를 설정하고 학습을 시작하고 중간에 학습을 멈추고 해당 파라미터를 저장하는 것이다. 아니면 혹은 매 회 정해진 횟수마다 파라미터 값을 저장하는 방법도 있다.

## 확률적 경사 하강법 (Stochastic gradient descent:SGD)
배치 경사 하강법의 가장 큰 문제는 학습을 진행하면서 매 스탭마다 전제 학습 데이터 세트에 대해서 기울기를 연산한다는 사실인데, 이는 데이터 세트가 엄청 크면 학습이 매우 느려질 수도 있다. 확률적 경사 하강법은 매 스탭마다 랜덤으로 학습데이터세트에서 인스턴스를 뽑아서 그 뽑은 단일 인스턴스에 대해서만 기울기를 연산한다. 분명히 이는 매 반복횟수마다 적은 데이터만으로 수행하기 때문에 알고리즘은 더 빨라질 것이다. 각 반복 횟수 마다 인스턴스를 연산하는데 필요한 메모리만 요구하기 때문에 거대한 학습 데이터 세트도 충분히 제한된 환경에서도 빠르게 연산이 가능하다. (SGD가 초과 메모리 알고리즘(Out-of-core)이다)

반면에, 확률적이기 때문에 배치 경사 하강법보다는 덜 규칙적이다. 천천히 스무스하게 최저점에 도달하지 않고 위 아래로 요동치면서 도달을 하게된다. 최저점에 충분히 수렴하겠지만, 한번 주위로 요동치기 시작하면 진정할줄 모른다. 그래서 학습을 멈추어보면 파라미터 값은 좋겠지만 최적값은 아닐 것이다. 

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-9.png)

손실 함수가 매우 불규칙적(곡선이 심한)이라면 이는 실제로 알고리즘이 지역적 최저점을 점프해서 넘어가는데 도움이 되기에 확률적 경사 하강법이 배치 경사 하강법이 하는 것보다는 전역 최저점을 확률이 더 높다. 

그러므로 임의성은 지역 최저점에 빠지는 것을 탈출할 수 있는 좋은 특성이지만, 한 지점에 정착을 하지 못한다는 단점이 있다. 하나의 해법으로는 반복 횟수가 증가할 때마다 학습률을 낮추는 것이다. 처음에는 높게 설정해주어 지역적 최저점을 넘어다닐수있도록 해주고, 점점 덜 움직이게하여 전역 최저점에 안착할 수 있도록해준다. 이런 과정을 "모의 가열 냉각, simulated annealing"이라고 하는데, 철강산업에서 철강을 냉각시는 것에서 비유해온 것이다. 그리고 매 반복 횟수마다 학습률을 설정해주는 함수를 "학습 스케줄"이라고 한다. 

아래의 코드는 간단한 학습 스케줄을 사용해서 확률적 경사 하강법을 구현한 것이다.
```
# 학습 스케줄을 사용해서 확률적 경사 하강법 구현
n_epochs = 50
t0, t1 = 5, 50  # 학습 스케줄 파라미터

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # 랜덤값으로 초기화

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # 책에는 없음
            y_predict = X_new_b.dot(theta)           # 책에는 없음
            style = "b-" if i > 0 else "r--"         # 책에는 없음
            plt.plot(X_new, y_predict, style)        # 책에는 없음
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 # 책에는 없음

plt.plot(X, y, "b.")                                 # 책에는 없음
plt.xlabel("$x_1$", fontsize=18)                     # 책에는 없음
plt.ylabel("$y$", rotation=0, fontsize=18)           # 책에는 없음
plt.axis([0, 2, 0, 15])                              # 책에는 없음
save_fig("sgd_plot")                                 # 책에는 없음
plt.show()                                           # 책에는 없음
``` 
우리가 m번 반복하라고 설정해놓은 것으로 이를 한바퀴 다 돌면 한 주기(epoch)가 돌았다고 한다. 배치 경사 하강법은 전체 학습 세트를 1000번 돌려야했지만, 여기서는 50번만 돌려도 충분히 꽤 좋은 솔루션으로 도달한다.
```
>>> theta
array([[ 4.21509616],
       [ 2.77011339]])
```
아래의 그림은 최초 학습 10번 스탭을 밟은 것에 대한 결과를 보여준다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-10.png)

학습을 진행하면서, 랜덤으로 인스턴스를 뽑아서 학습을 하기 때문에 학습에 전혀 참여하지 못하는 인스턴스도 있을 수 있다. 만약 매 주기때마다 학습에 가급적이면 모든 인스턴스가 참여할 수 있도록하기 위해서는 전체 학습 데이터를 섞어보는 것이다. 그리고 매 인스턴스를 사용할 때마다 섞는 것이다. 하지만 이는 일반적으로 좀 더 느리게 수렴하도록 한다.

Scikit-Learn으로 SGD를 사용해 선형 회귀 모델을 실행하기 위해서, 우리는 `SGDRegressor`이라는 클래스를 사용하는데, 기본 설정 손실 함수로, 제곱 오차 손실 함수를 최적화하는 것으로 설정되어있다. 아래의 코드는 50회 주기를 돌면서 기본적으로 학습 스케줄은 사용하면서 학습률로 0.1로 시작해, 학습하도록 구현되어있으며, 어떠한 정형화 함수도 쓰지 ㅇ낳는다.
```
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
```
다시 한번, 우리는 정규 방정식이 리턴하는 값에 충분히 가까워지는 솔루션을 볼 것이다.
```
>>> sgd_reg.intercept_, sgd_reg.coef_
(array([ 4.16782089]), array([ 2.72603052]))
```
## 소규모 배치 경사 하강법 (Mini-batch Gradient Descent)
마지막 경사 하강법 알고리즘은 "소규모 배치 경사 하강법"이라는 것이다. 이는 전체 학습 데이터 세트로 기울기를 연산하거나, 하나만 랜덤으로 뽑아서 연산하는 대신에, 각각의 스탭에서 배치와 확률적 경사 하강법을 같이 사용하는 것으로 이해하면 꽤 간단히 이해할 수 있다. 소규모 배치 경사 하강법은 "소규모 배치"(mini-batch)라고 하는 랜덤으로 인스턴스를 몇개 뽑아서 세트로 만든 뒤에 이에 대한 기울기를 연산하는 방법이다.

파라미터 공긴 내에서 알고리즘의 개선된 점은 SGD의 것보다 덜 불규칙하다는 점이며 SGD보다는 좀 더 가까이 수렴한다. 하지만 이는 좋지않은 지역적 최저점을 탈출하기 더 힘들어진다는 점이라는 것이다. 앞에서 보았던 두가지의 알고리즘의 중간 쯔음이라고 보면 좋을 것이다. 아래의 그림은 우리가 다루었던 세가지의 경사 하강법이 학습을 하면서 움직이는 경로를 그림으로 그려서 보여준 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-11.png)

앞에서 다루었던 선형회귀 알고리즘과 비교해보자

###### Table 4-1. 선형 회귀 학습 모델에 대한 알고리즘 비교
 **Algorithm** | **Large M** | **Out-of-core Support** | **Large N** | **Hyperparameters** | **Scaling Required** | **Scikit-Learn**
 :-------: | :-----: | :-----------------: | :-----: | :-------------: | :--------------: | :----------:
 Normal Eqaution | Fast | NO | Slow | 0 | No | LinearRegression
 Batch GD | Slow | No | Fast | 2 | Yes | n/a
 Stochastic GD | Fast | Yes | Fast | ≥2 | Yes | SGDRegressor
 Mini-batch GD | Fast | Yes | Fast | ≥2 | Yes | n/a
```
학습이 끝난 후에는 큰 차이점은 없다. 모든 알고리즘이 비슷한 모델이 될 것이고
정확하게 같은 방법으로 예측해서 같은 결과를 출력해낸다.
```

# 다항 회귀 모델(Polynomial Regression)
우리의 데이터가 실제로 간단한 직선 보다 더 복잡하다고 한다면 어떨까? 놀랍게도, 우리는 선형 모델로 비선형성 데이터를 학습시키는데 사용할 수 있다. 이를 하는데 간단한 방법은 새로운 특징값으로 각각의 특징의 파워를 추가해주는 것이다. 그리고나서 확장된 특징 세트에 대해서 선형 모델을 학습시키는 것이다. 이러한 기술을 흔히 *다항 회귀 모델*이라고 한다.

예시를 한번 보자. 먼저 간단한 이차함수로 비선형성 데이터를 만들어보자
```
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-12.png)

누가봐도 직선으로는 적절하게 데이터를 학습시킬수 없을 것이다. 그래서 Scikit-Learn의 `PolynomialFeatures`의 클래스를 사용해 학습 데이터 세트를 변환하고, (2차식으로 만들기 위해서) 새로운 특징값으로 학습 데이터 세트에 각각의 특징에 제곱을 추가한다. 
```
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
>>> X[0]
array([-0.75275929])
>>> X_poly[0]
array([-0.75275929,  0.56664654])
```
X_poly는 기존 특징값에 그 특징값의 제곱값을 더한 것이다. 우리는 이제 `LinearRegression`학습 모델을 사용해서 학습을 시킬 수 있다.
```
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([ 1.78134581]), array([[ 0.93366893,  0.56456263]]))
```
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-13.png)

나쁘지 않다. 원래 실제 함수는 ![](https://render.githubusercontent.com/render/math?math=y%20%3D%200.5%20x_1%5E2%20%2B%201.0%20x_1%20%2B%202.0%20%2B%20%5Ctext%7BGaussian%20noise%7D&mode=inline)인데, 학습 모델은 ![](https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D%20%3D%200.56%20x_1%5E2%20%2B%200.93%20x_1%20%2B%201.78&mode=inline)라고 예측하였다.

다중 특징값들이 존재한다면, 다항 회귀 모델은 선형 회귀 모델이 하지 못하는 특징값들 사이에 관계를 찾아내는 능력이 있다. 이는 `PolynomialFeatures`가 주어진 제곱수까지 모든 특징값들의 조합을 추가하기 때문에 가능한 것이다.
 
`PolynomialFeatures`(degree=d)는 n개의 특징을 가지고 있는 배열을 ![](https://render.githubusercontent.com/render/math?math=%5Cdfrac%7B%28n%2Bd%29%21%7D%7Bd%21%5C%2Cn%21%7D&mode=inline)개의 특징을 가지고 있는 배열로 변환해준다.

# 학습 곡선 (Learning Curves)
만약 높은 차수의 다항 회귀 모델을 사용하고 있다면, 평평한 선형 회귀 모델 보다 좀 더 학습 데이터를 잘 학습하는 경향이 있다는 것을 알게될 것이다. 예를들어, 아래의 그림은 순수한 선형 모델과 이차방정식 모델, 그리고 300차수의 다항식 모델에 대한 결과를 보여준다. 

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-14.png)

당연히 높은 차수의 다항 회귀 모델은 일부 학습 데이터에서 과하게 학습된(Overfitting) 반면에 선형회귀모델은 덜 학습된(Underfitting) 것을 볼 수 있다. 이차방정식 모델이 최적으로 학습이 되었다. 그렇지만 일반적으로 함수가 생성한 데이터는 모를 것이다. 그럼 우리의 학습 모델은 얼마나 복잡한 것일까? 어떻게 함수가 과하게 혹은 덜 학습되었다고 볼 수 있을까?

Chapter2에서 우리는 모델의 일반화 능력을 평가하기 위해서 Cross validation을 사용해왔다. 만약 Cross validation측정법으로 모델이 학습 데이터에 대해서는 잘 수행하지만 일반화 능력이 좋지 않다면, 우리의 모델은 과하게 학습된 (Overfitting)것이다. 둘 다 성능이 나쁘다면 덜 학습된(Underfitting) 것으로 볼 수 있다. 

또 다른 방법은 학습 곡선을 보는 것이다. 학습 데이터 셋 사이즈의 함수로 설정된 검증 데이터 세트와 학습데이터 세트에 대해서 모델에 대한 수행 능력을 그래프로 찍어내는 것이다. 그래프를 생성하기 위해서, 학습 데이터 세트의 다양한 사이즈의 서브셋으로 모델을 몇 번 학습을 시켜보자. 아래의 코드는 주어진 학습 데이터에 대해서 모델의 학습 그래프를 그리는 함수를 정의한 것이다.
```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # 책에는 없음
    plt.xlabel("Training set size", fontsize=14) # 책에는 없음
    plt.ylabel("RMSE", fontsize=14)              # 책에는 없음
```
평평한 선형 회귀 모델에 대한 학습 곡선을 보자.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-15.png)

이는 몇가지 설명이 필요하다. 먼저 학습 데이터에 대한 수행도를 보자. 학습 데이터세트에 하나나 두개의 인스턴스만 있을 때 학습 모델은 완벽하게 이를 학습할 수 있는데, 이는 왜 곡선이 0부터 시작하는지를 말해준다. 하지만 학습데이터에 새로운 인스턴스가 추가될 때 마다, 학습 모델이 학습 데이터에 대해서 완벽하게 학습을 하는 것은 불가능한데, 데이터에 노이즈가 발생하고, 전혀 선형적이지 못하기 때문이다. 그래서 학습 데이터 세트에 대한 에러가 고원의 형태로 도달하게되는데, 여기서는 학습 데이터가 더이상 평균 에러치를 더좋게 혹은 나쁘게 하지 못한다. 그렇다고 수가 적은 학습 인스턴스로 모델로 학습을 시키게되면 일반화를 잘 수행하지 못한다. 이는 왜 초기에 Validation 곡선의 에러가 아주 크게 나오는지를 알려준다. 학습을 진행하면서 점점 낮아지게되지먼 어느 지점에 도달하면 평평해져 고원형태를 띠게되고 더이상 수행도가 좋아지지 않으며 다른 곡선과 점점 가까워진다.

이러한 학습 곡선은 덜 학습된 (Underfitting) 모델의 전형적인 예시이며, 두개의 곡선이 고원형태를 띠고있다.  이들은 가까우면서도 꽤나 높다.
```
만약 학습 모델이 학습 데이터에 대해서 Underfitting을 발생시킨다면 학습 데이터를 늘리는 것은
더이상 도움이 안될 수도 있다. 한번 더 복잡한 모델을 사용해보자.
```
이제 같은 데이터에 대해서 10차수의 다항식 모델을 살펴보자. 
```
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])           # 책에는 없음
save_fig("learning_curves_plot")  # 책에는 없음
plt.show()                        # 책에는 없음
```
학습 곡선은 이전 것들보다 좋아보이지만, 두가지의 분명한 다른 점이 있다.
* 학습 데이터에 대한 에러가 선형 회귀 모델보다 더 낮다.
* 두 곡선 사이에는 갭이 존재한다. 이는 검증 데이터에 대해서 보다는 학습 데이터에 대해서가 어느정도 더 수행을 잘한다는 것을 의미하는데, 이는 과잉 학습의 전조이다. 하지만, 좀 더 커다란 학습 데이터 세트를 사용한다면 두개의 곡선은 좀 더 가까워 질 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-16.png)
```
과하게 학습된 (Overfitting)모델을 개선할 수 있는 방법은 검증 에러값이 학습 에러값에 
가까워질 때까지 학습 데이터를 더 추가해서 학습을 시키는 것이다. 
```
### Bias/Variance Tradeoff
기계학습과 통계학에서 중요한 최종적인 결과는 모델의 일반화 에러가 3가지의 매우 다양한 에러의 합으로 표현된다는 사실이다.
* 편향치(Bias) 

    이 일반화 에러의 일부는 잘못된 가정(assuption)으로 발생하는데 예로 데이터는 실제로 비선형적인데 선형이라고 가정하는 것이다. 높이 편향된 모델은 학습 데이터 세트에 대해서 학습이 잘 안될수도 있다.
* 변화량(Variance) 

    일반화 에러의 일부는 모델이 학습 데이터세트의 작은 변화에도 지나치게 민감한것이다. 모델이 자유도를 많이 가지게되면(높은 차원의 다항식 모델) 변화량도 높게 가지게 되며 이는 과잉학습으로 이어진다. 
* 더이상 줄일수 없는 에러(Irreducible error) 

    이는 데이터 자체에 노이즈에서 발생하는 것으로, 이 에러를 없애는 방법은 오직 데이터를 청소하는 것뿐이다. 

모델의 복잡도가 증가하는 것은 전형적으로 변화량이 증가하고 편향치가 낮아진다. 역으로 복잡도를 줄이는 것은 편향치가 증가하고 변화량이 줄어든다. 

# 정형화된 선형 모델 (Regularized Linear Models)
Chapter 1과 2에서 배웠던 것처럼, 과하게 학습되는 것(Overfitting)을 줄이는 방법은 모델을 정형화하는 것이다. 자유도(degree of freedom)를 적게 가질수록, 데이터를 과하게 학습하기가 어렵다. 예를들어 다항 회귀 모델을 정형화하는 쉬운 방법은 타항식의 차수를 줄이는 것이다.

## 리지 회귀 모델 (Ridge Regression)
리지 회귀 모델은 정형화된 선형 회귀 모델이다. 정형화 식 ![](https://render.githubusercontent.com/render/math?math=%5Calpha%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Ctheta_i%5E2%7D&mode=inline)를 손실함수에 추가한다. 이는 오직 학습을 하는 과정에만 더해지는 것이다. 학습이 되고나면 비정형화된 성능 측정으로 모델 성능을 파라미터의 성능을 파악하고싶을 것이다. 
```
학습에서 사용되는 성능 측정과 학습에서 사용하는 손실함수가 다른것은 아주 일반적인 것으로, 정형화를 떠나서
다를 수도 있는 이유는 좋은 학습 손실함수는 최적화에 친근한 도함수를 가지고 있는 반면에, 테스트에서 쓰이는 
성능 특정은 가능한 최종 목표에 가깝게 가야하기 때문이다. 이에대한 좋은 예시로 (곧 다루는)log loss라는
손실함수로 모델을 학습 시키지만 평가 지표로는 Precision/Recall을 사용한다느 것이다.
```
하이퍼 파라미터 α는 얼마나 모델을 정형화 할 것인지 컨트롤해준다. α=0이면 리지 회귀 모델은 일반적인 선형 회귀모델이 되는 것이다. 만약 α가 너무 크다면, 모든 가중치 값들은 결국엔 0으로 가게될 것이고, 데이터 평균을 경험하는 평평한 선이 될 것이다. 아래의 수학 공식은 리지 회귀모델 손실 함수를 나타내는 것이다.
###### Equation 4-8. 리지 회귀 모델 손실 함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-8.png)

편향치 θ0는 정형화되지 않는다. **w**를 θ1부터 θn까지 특징값의 가중치 벡터로 정의하면, 정형화 식은 1/2(||**w**||2)^2가 될 것이며, || . ||2는 가중치 벡터의 정규값을 나타낸다. 경사 하강법에 대해서는 α**w**를 MSE 기울기 함수에 더하게된다. 
```
리지 회귀 모델을 사용하기 전에 (StandardScaler를 사용하여)데이터의 크기를 조절하는 것이 중요한데, 이는 
입력 특징 값의 크기에 민감하기 때문이다. 모든 정형화가 적용되는 모델에는 공통적인 부분이다.
```
아래의 그림은 다양한 α값을 사용하여 일부 선형 데이터에 대하여 학습된 리지 회귀 모델을 보여준다. 왼쪽에는 선형 예측으로 유도한 평평한 리지 회귀 모델이 사용된 것이고, 오른쪽에는, 데이터가 `PolynomialFeature(degree=10)`를 사용하여 확장시키고, `StandardScaler`로 스케일을 조정한 뒤에, 이렇게 처리된 특징들로 리지 모델에 적용시킨다. 이것이 리지 정형화로 만든 다항 회귀 모델이다.

선형 모델에서는 경사 하강법을 사용하거나 폐쇄형태의 공식을 사용해서 리지 회귀를 수행할 수 있다. 장점과 단점은 같으며 아래의 공식예 폐쇄적인 형태의 솔루션을 보여주고 있다. 

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-17.png)

###### Equation 4-9. 리지 회귀 모델 폐쇄적 형태 솔루션
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-9.png)

여기 폐쇄적 형태의 솔루션을 사용하여 Scikit-Learn으로 리지 회귀 모델을 작성한 코드가 있다. 
```
>>> from sklearn.linear_model import Ridge
>>> ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
>>> ridge_reg.fit(X, y)
>>> ridge_reg.predict([[1.5]])
array([[ 1.55071465]])
```
그리고 확률적 경사 하강법도 사용해보자
```
>>> sgd_reg = SGDRegressor(penalty="l2", random_state=42)
>>> sgd_reg.fit(X, y.ravel())
>>> sgd_reg.predict([[1.5]])
array([ 1.13500145])
```
`Penalty`라는 하이퍼 파라미터는 사용하려는 정형화 식의 종류를 설정해준다. "l2"를 쓰게되면 SGD에 가중치 벡터의 정규값의 제공을 반으로 나눈 것과 같이 손실함수에 졍형화 식을 추가하라고 하는 것이다. 

## 라소 회귀 모델 (Lasso Regression)
최소 절대 능선 및 선택 연산(Least Absolute Shrinkage and Selection Opeation) 회귀 모델, 흔히 라소(Lasso) 회귀 모델은 선형 모델의 또다른 정형화된 버전이다. 리지 회귀모델 처럼, 손실함수에 정형화 식을 추가하지만, 가중치 벡터의 l1 정규값을 사용한다. 아래 공식을 참고하자.

###### Equation 4-10. 라소 회귀 모델 손실 함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-10.png)

아래의 그림은 이전의 그래프와 같은 것을 보여주고 있지만, 좀 더 작은 α값을 사용한 라소 모델로 리지 모델을 바꾸어본 것이다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-18.png)

라소 회귀 모델의 중요한 특성은 덜 중요한 특징의 가중치를 (0으로 만들어)환벽하게 제거하는 것인데, 예를들어 위의 그림에서 α=1e-7의 선은 비선형적으로보이지만 거희 선형으로보인다.

![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/4-19.png)
```
라소 손실 함수에 대해서, BGD의 경로는 푹 패인 곳으로 요동치며 가는 경향이있다. 이는 경사도가
θ2 = 0에서 갑자기 변하기 때문이다. 정말ㅇ로 전력 최저점에 수혐하기 위헤서는 학습률을 점차적으로
줄여나갈 필요가 있다.
```
라소 손실 함수는 θi=0인 지점에서는 미분이 불가능하다. 하지만 경사 하강법은 θi=0일때를 대신해서브 기울기 값으로 벡터 g^15를 사용하면 여전히 잘 작동할 것이다.아래의 공식은 서브 기울기 벡터 공식이며, 라소 손실 함수로 경사 하강법을 사용할 수 있다.  

###### Equation 4-11. 라소 회귀모델 서브 기울기 벡터
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/04/Eq4-11.png)

다음은 Scikit-Learn의 `Lasso`글래스를 사용한 예시이다. `SGDRegressor(Penalty="l1")`대신에 사용할 수 있다.
```
>>> from sklearn.linear_model import Lasso
>>> lasso_reg = Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
>>> lasso_reg.predict([[1.5]])
array([ 1.53788174])
```

## 탄력있는 망(elastic net)




**[뒤로 돌아가기](https://github.com/Hahnnz/handson_ml-Kor/)**
