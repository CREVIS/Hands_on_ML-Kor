hapter 5. 서포트 벡터 머신
=====
*Support Vector Machine*(SVM)은 선형이나 비선형 분류, 회귀 그리고 심지어 이상점 탐지를 수행하는 능력을 가진 매우 강력하고 다재다능한 기계학습모델이다. 기계학습에서 가장 인기있는 모델중 하나이며, 기계학습에 관심있는 사람이라면 이 알고리즘을 경험해볼 필요가 있다. SVM은 특히 분류 모델에 아주 적합하지만 데이터 세트가 충분히 작거나 중간치 정도만 되어야한다.

이번 Chapter에서는 SVM에 대한 기본 개념과 어떻게 사용하고 어떻게 작동을 하는지 설명할 것이다. 

# 선형 SVM 분류

SVM의 기반이 되는 아이디어는 그림으로 아주 잘 설명되어있다. 아래의 그림을 보면 Chapter4의 끝에서 소개했던 iris 데이터 세트의 일부를 보여준다. 두개의 클래스는 선형적인 직선으로 분명하게 나뉘어진다. 왼쪽의 그림은 3가지의 가능한 선형 분류 모델의 의사 결정선이다. 점선으로 표현된 의사 결정선을 가지고 있는 학습 모델은 두개의 클래스를 제대로 나누어주지 못해서 안좋은 모델이다. 다른 두개의 모델은 학습데이터 세트에 대해서는 완벽하게 잘 작동하고 있지만, 이 모델들은 데이터와 의사결정선이 서로 너무 가까워서 새로운 데이터에 대해서는 제대로 분류를 하지 못할 것이다. 반면에, 오른쪽 그림의 실선으로 표시된 것은 SVM의 것으로, 이 두개의 실선은 두개의 클래스를 구분할 뿐만 아니라 각각의 선들이 각 클래스의 데이터에 최대한 가깝게 배치가 되어있다. 이제 SVM 분류 모델을 클래스 사이에 두개의 병렬로 그어진 선들을 최대한 서로 멀리 그어지도록 하는 것으로 생각해볼 수 있다. 이를 (*large margin classification*)이라고 한다.
###### 그림 5-1. Large Margin Classification
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-1.png)

학습데이터로 좀 더 멀리 떨어져있는 것을 넣는 것은 의사결정선에 전혀 영향을 미치지 못한다. 이는 전적으로 거리의 끝에 위치해있는 것에 의해 결정되기 때문이다. 이러한 인스턴스를 Support Vector라고 부른다. (위의 그림에서 )
```
SVM은 아래의 그림에 나와있는 것처럼 특징값의 크기값에 매우 민감하다. 왼쪽의 그림에는 수직선의 값들이 
아래의 수평선의 값들보다 아주 크다. 이러면 가장 넓은 가능한 거리는 수평선으로 가까워 질 것이다.
Feature Scaling을 하고 나면, (Scikit-Learn의 `StandardScaler`를 사용하자), 의사 결정선은
오른쪽 그림에서 보여지는 것처럼 좀 더 보기 좋아질 것이다.
```
###### 그림 5-2. Sensitivity to feature scales
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-2.png)
## Soft Margin Classification
만약 모든 인스턴스가 올바른 측면으로 거리를 멀어지게 하도록, 즉 두개의 클래스에 대해서 각 클래스에 속한 인스턴스들이 구분이 완벽히 되도록 강요한다면, 이를 엄격한 차이 분류(*Hard Margin Classification*)이라고 한다. 이에 대한 두가지의 이슈가 있다. 첫번째는 데이터가 선형적으로 나뉘어질 수 있어야한다는 점과, 두번째로 이는 꽤 이상점(Outlier)에 민감하다는 것이다. 아래의 그림은 iris데이터 세트에 하나의 이상점을 넣어본 것이다. 왼쪽은 엄격하게 차이를 내는 것(Hard margin) 자체가 불가능하다. 그리고 오른쪽은 맨 앞에서 봤던 첫번째 그림과 아주 다른 모양의 의사결정선으로 도출될 것이다. 이러한 선은 적절하게 일반화를 하지 못할 것이다.
###### 그림 5-3. Hard margin sensitivity classification
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-3.png)

이러한 이슈를 피하기 위해 좀 더 유연한 학습모델이 선호된다. 목적은 가능한 두 선사이의 거리를 멀게 유지해주고 마진 위반(*Margin violation*)를 제한하는(입력된 인스턴스가 결정선 사이에 있거나 잘못된 클래스에 있는 것을 말한다.) 좋은 균형점을 찾는 것이다. 

Scikit-Learn의 SVM은 우리가 C라는 파라미터를 사용하여 그 균형을 조절할 수 있다. c값을 작게할수록 좀 더 거리가 넓어지지만 *Margin Violation*을 일으킬 경향이 있다. 아래의 그림은 비선형적으로 나누어진 데이터 세트에 대하여 두개의 Soft Margin SVM의 거리와 결정선을 보여준다. c값을 크게할수록 마진 위반은 줄어들지만 선 사이의 거리가 좁아지게되어 일반화를 잘 못할 것이다. 오른쪽은 낮은 값의 c값을 사용하였고, 이는 선 사이의 거리가(마진이) 좀 더 넓어지지만 많은 인스턴스들이 그 선 사이 상에 있을 것이다. 하지만 두 번째의 분류 모델이 좀 더 일반화를 잘하는데, 사실 이 학습 데이터 세트에 대해서 예측 에러가 적게 발생하는데 이는 대부분의 margin violation이 실제로 의사 결정성의 맞는 쪽에 있기 때문이다.
###### 그림 5-4 Fewer margin violation versus large margin
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-4.png)
```
SVM 모델이 과잉학습을 하게되면, c값을 줄임으로써 이를 정형화해볼 수 있다.
```
다음의 Scikit-Learn 코드는 iris 데이터 세트를 불러와서 특징값들을 스케일링하고 그 후에 virginica flower를 탐지하는 선형 SVM모델을 사용하여 학습을 진행합니다. (`LinearSVM`에 c값은 1로, 그리고 곧 설명할 `hinge loss`라는 함수를 사용한다.) 이 코드에 대한 결과는 위의 그림으로 표현된다.
```
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

svm_clf.fit(X, y)
```
그리고나서, 늘 하던대로, 학습된 모델을 사용해 예측값을 만들어 보자
```
>>> svm.clf.predict([[5.5,1.7]])
array([1.])
```
```
로지스틱 회귀 분류모델과는 다르게, SVM 분류 모델은 각각의 클래스에 대한 확률로 결과를 주지 않는다.
```
# 비선형 SVM 분류
비록 선형 SVM분류 모델이 수많은 경우에서 효과적이고 놀랍게 잘 작동하지만 많은 데이터 세트들은 선형적으로 나누기가 쉽지않다. 이러한 비선형적인 데이터 세트를 다루는 접근 방법은 (Chapter4에서 했던 것처럼) 특징들을 여러개 사용해 다항으로 만드는 것처럼 좀 더 특징값들을 추가하는 것이다. 일부의 경우에서는, 선형적으로 나눌 수 있는 데이터 세트로 결과를 낼 수 있다. 아래의 왼쪽 그림을 보자. 이는 특징값 x1을 하나 더한 것을 보여준다. 이 데이터 세트는 보다시피 선형적으로 나누어지지않는다. 하지만, x2=(x1)^2을 두번째 특징값으로 입력한다면, 다음의 이차원 데이터세트 결과는 완벽히 선형으로 나누어질 수 있다.
###### 그림 5-5. Adding feature to make a data linearly separable
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-5.png)

이 아이디어를 Scikit-Learn으로 구현하기 위해서는, `StandardScaler`와 `LinearSVC`라는 함수로 구현된 `PolinomialFeature` 변환함수를 포함한 pipline을 만들어야한다. 이제 moons 데이터 세트로 이를 실행시켜보자. 
```
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
```
###### 그림 5-6. Linear SVM classifier using ploynomial feature
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-6.png)
## 다항식 커널 (Polynomial Kernel)
다항 특징값들을 추가하는 것은 구현하기 간단하고, (SVM뿐만 아니라)모든 기계학습 알고리즘에 대해서 아주 잘 작동하지만, 다항식의 차수가 낮은 모델로는 매우 복잡한 데이터 세트를 다루기에는 적합하지 않다. 

다행히도, SVM을 사용할 때 (잠깐 설명되었던) 커널 기법(kernel trick)이라고 불리는 거희 기적과도 같은 수학공식을 적용할 수 있다. 이는 마치 심지어 아주 높은 차원의 다항 모델로도 실제로는 다항 특징값을 추가하지는 않았지만 많은 다향 특징값들을 추가한 것 처럼 같은 것과 같은 결과를 얻을 수 있게해준다. 그래서 특징값을 추가하지 않기 때문에, 특징값 조합 수의 폭발적 증가는 발생하지 않는다. 이러한 기법은 `SVC`클래스에 의해 구현되어있으며 한번 moons 데이터 세트로 확인해보자.
```
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
```
이 코드는 3차다항식 커널을 사용하여 SVM을 학습시키는 코드이다. 이는 아래의 그림의 왼쪽에 그려져 있으며, 오른쪽은 10차다항식 커널을 사용하여 SVM을 학습시킨 것에 대한 결과이다. 분명히 학습 모델이 과잉학습을 하면, 우리는 다항식의 차수를 줄일것이다. 역으로, 학습이 덜되었다면, 차수를 올릴려고 할 것이다. `coef0`라는 하이퍼 파라미터는 얼마나 모델이 낮은 차수 다항식 대비 높은 차수의 다항식에 의해 영향을 받도록 할 것인지 정해주는 하이퍼 파리미터이다. 
###### 그림 5-7. SVM classifier with a polynomial kernel
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-7.png)

## 유사특징(similarity feature) 추가하기
비 선형 문제를 다루는 또다른 기술은 바로 유사도 함수를 사용해 연산되는 특징들을 추가하는 것이다. 유사도 함수는 각각의 인스턴스들이 특정 랜드마크와 얼마나 비슷한지를 측정한다. 예로 앞에서 다루었던 1차원 데이터세트를 가지고 2개의 랜드마크 x1=-2, x2=1을 추가하자. (아래 그림 참고) 그리고 유사도 함수를 가우시안 방사성 기저 함수(Gaussian Radial Basis Function: RBF)를 γ(감마)값 3으로 정의한다.
###### Equation 5-1 가우시안 RBF
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-1.png)

함수는 종모양 형태로 0(설정한 랜드마크와 거리가 멀은 것)에서 1(설정한 랜드마크와 거리가 가까운 것)까지 고르게 퍼져있다. 이제 새로운 인스턴스에 대해서 연산해보자. 새로운 인스턴스로 x1=-1이 들어오면, 첫번째 랜드마크인 x1=-2와는 거리가 1정도 차이나며, 두번째 랜드마크인 x1=1은 거리가 2만큼 떨어져있다. 그러므로 새로운 특징값 x2는 exp(-0.3 X 1^2) = 0.74만큼, x3는 exp(-0.3 X 2^2)=0.30을 가지게 된다. 아래의 그림은 (기존의 특징값을 버린)변형된 데이터 세트를 그린 것으로 보다시피 이제 선형적으로 구분이 가능하다
###### 그림 5-8. Similarity features using the Gaussian RBF
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-8.png)

랜드마크를 어떻게 설정해야하는지 궁금해할 수도 있다. 가장 간단한 방법은 데이터세트에 있는 모든 변수들의 위치에 랜드마크를 설정해보는 것이다. 이는 수많은 차원을 만들어 낼 것이고, 그러므로 변형된 데이터셋이 좀 더 선형적으로 구분될 가능성을 높여준다. 단점은 m개의 인스턴스와 n개의 특징값들로 이루어진 학습 데이터 세트를 m개의 인스턴스와 m개의 특징값으로 변형시킨다.(기존에 존재하던 특징값들을 버리면서) 만약 학습데이터세트가 크다면 엄청난 숫자의 특징값들을 보게될 것이다.

## 가우시안 방사성 기저 함수 커널기법 (Gaussian RBF Kernel)

다항으로 특징값들을 조합하는 것처럼, 유사특징값은 어느 기계학습알고리즘에도 아주 유용하지만, 데이터 세트가 엄청 크다면, 가능한 추가될 수 있는 값들을 연산하는데 엄청 오래 걸릴것이다. 하지만 커널 기법이 다시한번 이 SVM에 마법을 선사해줄 것이다. 실제로 어떠한 특징값을 추가하지 않고도 마치 수많은 유사특징을 사용한 것같이 비슷한 결과를 얻을 것이다. 이제 한번 가우시안 RBF 커널기법을 `SVC`클래스를 사용하여 구현해보자
```
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)
```
이 모델은 아래 그림의 아래에서 왼쪽에 존재하는 그림으로 표현되어있다. 다른 그림들은 다양한 `감마값(γ)`과 C라는 하이퍼 파라미터들을 통하여 학습된 것을 보여준다. 감마값이 오를 수록 종모양의 형태가 날렵해지며 각각의 인스턴스의 영향력 범위가 감소한다. 이는 의사결정선이 좀 더 불규칙해지고, 각각 인스턴스 주위에서 이리저리 요동쳐대고 있을 것이다. 역으로 감마값이 작아지면 종모양이 좀 더 넓게 변할 것이고. 각각의 인스턴스들의 영향력 범위가 넓어져서 의사결정선이 스무스하게 유지될 것이다. 그래서 γ는 정형화(Regularization)의 하이퍼 파라미터 처럼 움직이는데 만약 학습 모델이 과잉학습을 하게 된다면 이 값을 줄여야 할 것이고, 학습이 덜되었다면 그 값을 올려야할 것이다. (C 하이퍼 파라미터 또한 마찬가지이다.)
###### 그림 5-9. SVM classifiers using an RBF kernel
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-9.png)

다른 커널기법들이 존재하지만, 거의 사용이되지 않는다. 예를들어 DNA 배열이나 문서의 텍스트를 분류하는데에 사용되는 *String Kernel*기법이라는 것도 존재하긴 한다.

## 연산 복잡도
`LinearSVC`클래스는 *liblinear*라이브러리에 기반하는 것이며, 이는 이미 선형 SVM에 [최적화된 알고리즘](http://goo.gl/R635CH)이 구현되어있는 것이다. 커널 기법을 지원하지않지만 특징값의 수와 학습 인스턴스의 수로 거의 선형적인 크기를 지니고 있어 학습 시간 복잡도는 O(mXn)이다. 

만약 높은 Precision을 요구하게된다면 알고리즘은 더 시간이 오래걸릴 것이다. 이는 용인(tolerance) 하이퍼 파라미터(Scikit-Learn에서는 `tol`이라고 부른다)로 컨트롤된다. 대부분의 분류 작업에서는 기본값으로 용인해도 좋다고 설정되어 있다.

`SVC`클래스는 *libsvm*라이브러리를 기반으로 하고 있으며, [커널 기법을 지원해주는 알고리즘](http://goo.gl/a8HkE3)이 구현되어있다. 학습 시간 복잡도는 보통 O(m^2Xn) 에서 O(m^3Xn)까지이다. 불행하게도 이는 학습데이터세트의 크기 크면 클수록 그 수에 몇배나 되는 시간이 걸리는 것이다. 이 알고리즘은 복잡한 문제에 좋지만 학습 데이터 세트가 작거나 중간치 정도여야만 한다. 하지만 특징의 수를 잘 스케일링하는데 특히 희소특징(특징값이 대부분이 0인 것)에 대해서 잘 수행한다. 

Class | Time Complexity | Out-of-coure Support | Scaling required | Kernel trick
:---: | :-------------: | :------------------: | :--------------: | :----------:
LinearSVC | O(mXn) | No | Yes | No
SGDClassifier | O(mXn) | Yes | Yes | No
SVC | O(m^2Xn) 에서 O(m^3Xn) | No | Yes | Yes

# SVM 회귀 모델
앞에서 언급했다시피, SVM알고리즘은 꽤나 다재다능하다. 선형/비선형 분류뿐만 아니라 선형/비선형 회귀도 지원해주기 때문이다. 이 기법은 목적함수가 정반대이다. 두개의 클래스 사이의 거리를 충분히 멀리 유지하고 마진 위반(Margin Violation)을 예방하는 것 대신에, SVM 회귀 모델은 마진 위반을 예방하면서 가능한 인스턴스들이 모두 마진 위에 있게끔 학습시키는 것이다. 거리의 넓이는 하이퍼 파리미터 ϵ에 의해 컨트롤된다. 아래의 그림은 ϵ=1.5와 ϵ=0.5로 랜덤 선형 데이터로 학습시킨 두가지 선형 SVM 회귀 모델을 보여준다.
###### 그림 5-10. SVM Regrssion
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-10.png)

마진에 들어가는 학습 인스턴스를 추가하는것은 모델의 예측 능력에 영향을 주지 못한다. 그러므로 모델은 *ϵ에 민감한* 모델이라고 할 수 있다.

Scikit-Learn에서 `LinearSVR`이라는 클래스가 선형 SVM 회귀를 수행해준다. 아래의 코드는 위의 그림 왼쪽을 보여주는 모델을 만들어주는 코드이다. (학습하고 있는 데이터는 이미 스케일링이 되어 있는 것이다.)
```
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)
```
비선형 회귀 업무를 다루기 위해서 커널화된 SVM 모델을 사용할 수 있다. 예를 들어, 아래의 그림은 이차항의 커널을 사용한 랜덤 이차원 학습 데이터에 대한 SVM 회귀 모델을 보여준다. 왼쪽 그림에는 작은 정형화(큰C값)가 존재하며 왼쪽에는 좀 더 많은 정형화(작은C값)가 존재한다. 
###### 그림 5-11. SVM regression using a 2nd-degree polynomial kernel
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-11.png)

다음의 코드는 Scikit-Learn의 `SVR`클래스를 사용하여 위 그림의 왼쪽에 있는 그림으로 보여지는 모델을 만들어낸다. `SVR`클래스는 `SVC` 클래스에 동등한 회귀를 사용하는 것이며 `LinearSVR`클래스는 (`LinearSVC`처럼) 학습 데이터의 사이즈를 선형적으로 스케일링하는 반면에 `SVR`클래스는 (`SVC`처럼) 학습 데잍 ㅓ세트가 커지면 너무 오래걸린다는 단점이 있다.
```
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```
```
SVM은 또한 이상점(Outlier) 탐지하는데 사용하기도 하는데 이는 Scikit-Learn의 문서에서 좀 더 자세하게 확인할 수 있다.
```

# 내막 속에는 (Under the Hood)
이 섹션에서는 선형 SVM부터 시작해서 어떻게 SVM이 예측을 하고, 어떻게 학습 알고리즘이 작동하는지에 대하여 설명할 것이다. 만약 기계학습을 이제 막 시작했다면 이 Chapter의 끝에 있는 연습문제를 먼저 풀어보고와서 다시 돌아와 이 Chapter를 보는 것을 추천한다.

먼저 개념에 관한 것으로 Chapter 4에서 우리는 편향치 θ0와 입력 특징값 θ1부터 θn까지로 사용하고, 편향치 입력 x0=1을 더하는 식으로 하나의 벡터 θ만을 보편적으로 사용하였다. 이번 Chapter에선 다른 형태의 용어를 사용할 것인데 SVM에서 다룰때 자주 사용되는 용어이다. **w**라고 하는 가중치 벡터와 **b**라고 하는 편향치값을 사용할 것이다. 입력 특징값 벡터에는 편중치 특징값이 추가되지 않는다.

## 의사결정함수와 예측
선형 SVM모델은 의사 결정함수 `W^T⋅X+ b` = W1⋅X1 + W2⋅X2 + ... + Wn⋅Xn + b를 연산하여 새로운 인스턴스 X에 대하여 클래스를 예측한다. 만약 결과가 Positive라면 ŷ는 Positive(1)클래스로 아니면 Negative(0)클래스로 예측값을 준다. 아애의 공식을 참고하라.

###### Equation 5-2. 선형 SVM 분류모델 예측함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-2.png)

아래의 그림은 [그림5-4](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#%EA%B7%B8%EB%A6%BC-5-4-fewer-margin-violation-versus-large-margin)의 오른쪽 그림에 상응하는 모델에 대한 의사 결정 함수를 보여준다. 이는 2차원 평면인데 데이터 세트가 2개의 특징값(꽃잎의 넓이와 길이)을 가지고있기 때문이다. 의사결정선은 의사결정함수값이 0인 점의 집합으로 이는 두개의 평면이 상호 교차되는 것이며 직선으로 표현되어 있다. 
###### 그림 5-12. iris 데이터 세트에 대한 의사결정선
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-12.png)

대쉬선은 의사결정 함수값이 1이나 -1이 나오는 지점들에 대한 집합으로 마진값을 생성하면서 의사결정선 사이와 거리가 동등하게 병렬로 두개의 선이 놓여있다. 선형 SVM을 학습시킨다는 것은 마진위반을 피하거나(Hard margin-엄격한 마진) 제한하면서도(Soft Margin-유연한 마진), 가능한 넓게 마진을 만들어 주는 **W**값과 **b**값을 찾는 것이다. 

## 학습 목적함수 (Training Objective)
의사결정선의 기울기에 대해서 생각을 해보자. 가중치 벡터의 Norm값 ||**W**||과 동일하다. 만약 2로 기울기를 나눈다면 의사결정함수값이 ±1이 나오는 지점은 의사결정선에 2배 정도 멀어질 것이다. 다시말해서 기울기를 2로 나눈다는 것은 마진에 2를 곱하는 것과 같은 것이다. 아마 이는 아래의 그림에 2차원으로 시각화하기 더 쉬울 것이다. 가중치 벡터값 **W**가 작아질수록, 마진값은 커져간다.
###### 그림 5-13. Large Margin에서 점점 줄어들고 있는 가중치 벡터 결과
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/5-13.png)

그래서 큰 값의 마진을 가지는 ||**W**||를 최소화하는 목표이다. 하지만 마진 위반도 피하게 하고 싶다면(Hard Margin) 모든 Positive 학습 인스턴스들에 대하여 1 이상으로 큰 값이 되고 모든 Negative 학습 인스턴스에 대해서는 -1이하가 될 수 있는 의사결정 함수값이 필요하다. 만약 (y^(i)=1라고 하여)t^(i)=-1로 Negative 클래스를 정의하고, (y^(i)=1라고 하여) t^(i)=1로 Positive 클래스를 정의하면 우리는 이를 모든 인스턴스에 대해서 `t^(i)(W^T⋅X^(i)+b) ≥ 1`로 표현할 수 있다.

그러므로 엄격한 마진 선형 SVM 분류 모델의 목적함수는 아래의 공식으로 제약식 최적화(Constrained Optimization)라고 한다. 
###### Equation 5-3. 엄격한 마진 선형 SVM 분류모델 목적
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-3.png)
```
1/2⋅W^T⋅W를 최소화하는 것으로, 이는 ||W||를 최소화하는 것보다는 1/2⋅||W||^2를 최소화한다. 두개는 서로
결과적으로는 같은 값을 가지고 있지만 ||W||는 W=0인 점에서 미분할 수 없어서 1/2⋅||W||^2를 사용하여 
미분하기 용이하게 해준다. 
```
유연한 마진 목적에 대해서 각각의 인스턴스에 대해서 Slack 변수 ζ^(i)≥0를 도입할 필요가 있다. ζ^(i)는 얼마나 i번째의 인스턴스가 마진을 위반하는지 측정하는 것으로, 이제 두가지의 목적에 대해서 고민해보아야한다. 마진 위반을 가능한 작게 만들어 주도록 Slack 변수를 만들거나 마진이 가능한 많이 커질 수 있도록 1/2⋅W^T⋅W를 가능한 작게 만드는 것 둘 중에 무엇을 선택할 지 골라야한다. 이는 C 하이퍼 파라미터가 관여되는 곳으로 앞의 두가지 목적 함수에 대하여 Tradeoff가 발생된다. 이는 아래의 공식으로 제약식 최적화 문제를 준다.

###### Equation 5-4. 유연한 마진 선형 SVM 분류모델 목적
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-4.png)

## 이차 계획법 (Quadratic Programming)
엄격한 마진과 유연한 마진 문제는 모두 선형 제한사항들에 대한 볼록한(Convex) 이차 최적화 문제이다. 이러한 문제를 흔히 이차 계획법(Quadratic Programming:QP)문제라고 한다. 이 책의 범위 외에 있는 다양한 기법들을 사용하여 바로 사용이 가능한 수많은 해결책들이 존재한다. 일반적인 문제 수학 공식이 다음과 같이 아래에 주어져있다. 
###### Equation 5-5. 이차 계획법 문제
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-5.png)

**A⋅P ≤ b** 수식은 실제로 nc 제한사항들로 정의되어있다. i=0,1,...,nc에 대하여 **P^T⋅a^(i)≤b^(i)**로 정의되어 있으며, a^(i)는 A의 i번째 row의 요소에 대한 벡터값을 의미하며, b^(i)는 b의 i번째 요소를 의미한다.

다음과 같은 방법으로 QP파라미터를 설정하면 엄격한 마진 선형 SVM 분류 모델의 목적을 달성하는 것을 쉽게 보장할 수 있다.
* np = n+1로, n은 특징값의 수 (+1은 편향치에 대한 것이다.)
* nc = m으로, m은 학습 인스턴스의 수를 말한다.
* **H**는 (편향치를 무시하기 위해서) 맨위 오른쪽의 cell에 있는 0을 제외한 np X nc 항등행렬이다. 
* ** f = 0 **, 모든 값이 0으로 된 np차원 벡터값
* ** b = 1 **, 모든 값이 1로 된 nc차원 벡터값
* a^(i) = -t^(i)x^(i)로, x^(i)는 추가 편향 특징값 x0=1이 더해진 x^(i)와 같다.

그래서 엄격한 마진 선형 SVM분류 모델을 학습시키는 방법은 이전의 파라미터들을 패스하고 바로 사용이 가능한 QP Solver를 사용하는 것이다. 결과 벡터값 p는 i=0,1,...,m에 대하여 편향치 b=p0와 특징 벡터값 wi = pi 를 포함하고있다. 비슷하게 유연한 마진 문제를 해결하기 위해서도 QP Solver를 사용할 수 있다. (이 Chapter의 맨 끝에 있는 연습문제를 보자)

하지만 커널 기법을 사용하기 위해서 다른 형태의 제약식 최적화 문제를 살펴볼 것이다.

## 쌍대문제(Dual Problem)
쌍대 문제는 선형계획(Linear Programming)문제 또는 비선형계획(Non-Linear Programming)문제에서는 원래의 문제와 대응되는 표리 관계에 있는 다른 문제를 칭하는 것으로 선형 계획법(LP)의 최소(최대) 문제로부터 유도한 최대(최소) 문제를 주(主)문제에 대하여 이루는 용어이다. 예를 들면, 주 문제가 표준형으로 제약식은 ai1x1＋…＋ainxn＝bi(i＝1, 2, …, m), xj≥0(j＝1, 2, …, n), 최소로 해야 할 목적 함수를 y＝c1x1＋…＋cnxn라 하면 쌍대 문제는 제약식을 aijw1＋…＋amjwm≤cj(j＝1, …, n)로 하는 목적 함수 z＝b1w1＋…＋ bmwm의 최대화 문제로 된다.

쌍대 문제에 대한 솔루션은 전형적으로 primal 문제의 솔루션에 최소 한계점을 준다. 하지만 몇가지 조건 하에 primal 솔루션것과 같은 솔루션을 가질 수도 있다. 다행히 SVM 문제는 이러한 조건에 충족하기 때문에 우리는 쌍대문제나 Primal 문제 둘 중에 하나를 선택해서 풀면 되고 두 가지 모두 똑같은 솔루션을 가지고 있다. 아래의 공식은 선형 SVM 목적함수의 쌍대 문제 형태를 보여준다. 
###### Equation 5-6. 선형 SVM의 쌍대 문제 목적함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-6.png)

(QP Sovler를 사용해서) 이 공식을 최소화 해주는 벡터값 α̂를 찾기만 하면, 아래의 공식을 사용해서 primal 문제를 최소화해줄 수 있는 ŵ값과 b̂값을 연산할 수 있다.
###### Equation 5-7. 쌍대 솔루션에서 Primal 솔루션으로
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-7.png)

쌍대 문제는 학습 인스턴스의 수가 특징값의 수보다 더 적을 때 primal보다 더 빠르게 해결해준다. 더 중요한 것은 primal에서 사용할 수 없었던 커널 기법을 사용할 수 있게 해준다.
## 커널화된 SVM 
(moons 데이터셋 같은)2차원 데이터 세트를 이차다항식 변환에 적용하고 변환된 학습 데이터 세트에 대해서 선형 SVM 분류 모델을 학습시켜보자. 아래의 공식은 우리가 적용하기를 원하는 이차 다항식 맵핑함수 ϕ를 보여준다.
###### Equation 5-8. 이차수의 다항맵핑(Second-degree polynomial mapping)
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-8.png)

변환된 벡터는 2차원이 아닌 3차원으로 변환된 것이다. 이제 한 쌍의 벡터값, a와 b에 대하여 이차 다항식 맵핑함수를 사용하고 두 변환된 벡터에 대해서 내적을 하면 어떻게 되는지 확인해보자. (아래 공식 참고)

###### Equation 5-9. 커널 기법을 적용한 이차수의 다항맵핑
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-9.png)

어떤가? 변환된 벡터의 내적은 기존 벡터값의 내적한 것에 제곱한 것`ϕ(a)^T⋅ϕ(b)=(a^T⋅b)^2`과 같은 결과를 보인다. 

가장 중요한 키 포인트는 만약 모든 인스턴스에 대해서 변환함수 ϕ를 적용한다면 쌍대문제([공식 5-6](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#equation-5-6-%EC%84%A0%ED%98%95-svm%EC%9D%98-%EC%8C%8D%EB%8C%80-%EB%AC%B8%EC%A0%9C-%EB%AA%A9%EC%A0%81%ED%95%A8%EC%88%98))은 내적값 ϕ(X^(i))^T⋅ϕ(x^(j))를 포함하고 있다. 하지만 만약 ϕ가 [공식 5-8](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#equation-5-8-%EC%9D%B4%EC%B0%A8%EC%88%98%EC%9D%98-%EB%8B%A4%ED%95%AD%EB%A7%B5%ED%95%91second-degree-polynomial-mapping)에서 정의된 이차 다항식 맵핑함수라면, 
![](https://render.githubusercontent.com/render/math?math=%28%7B%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D%7D%5ET%20%5Ccdot%20%5Cmathbf%7Bx%7D%5E%7B%28j%29%7D%29%5E2&mode=inline)로 간단히 변환된 벡터에 대한 내적값을 대체해줄수 있다. 그래서 실제로 학습 인스턴스를 전혀 변환해줄 필요가 없다. 그냥 [공식 5-6](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#equation-5-6-%EC%84%A0%ED%98%95-svm%EC%9D%98-%EC%8C%8D%EB%8C%80-%EB%AC%B8%EC%A0%9C-%EB%AA%A9%EC%A0%81%ED%95%A8%EC%88%98)에서 이것을 제곱한 값으로 내적 값을 대체해주면 된다. 이 결과는 마치 학습 데이터 셋을 실제로 변환하는데 고생하고 이를 선형 SVM 알고리즘으로 학습시킨 것과 결과가 완전 똑같다. 하지만 이런 기법은 전체 프로세스에 연산 횟수를 더 증가시킨다. 이것이 커널 기법의 정수이다. 

함수 K(a,b) = (a^T,b)^2은 이차 다항식 커널이라고 부른다. 기계학습에서 커널이라는 것은 변환 함수 ϕ를 연산할 필요없이 오직 기존 벡터값 a와 b를 사용해서 ϕ(a)^T⋅ϕ(b)를 연산하는 능력을 가진 함수이다. 아래의 공식은 몇가지의 가장 흔하게 사용되는 커널의 리스트를 보여준다.
###### Equation 5-10. 일반적 커널들
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-10.png)
```
머서의 정리(Mercer's Theorem)

머서의 정리에 따르면, 만약 K(a,b)는 *머서의 조건* (K는 받는 인자들의 값이 연속적이고, 
대칭적(K(a,b)=K(b,a))이여야 한다.)이라고 하는 몇가시 수학적 조건에 준수한다면, K(a,b)=ϕ(a)^T⋅ϕ(b)
처럼 가능한 좀 더 높은 차원으로 a와 b를 또 다른 공간에 맵핑해주는 함수 ϕ가 존재한다. 그래서 K를 커널처럼
사용할 수 있는데 우리가 ϕ가 무엇인지 모르더라도 ϕ가 존재한다는 것을 알기 때문이다. 가우시안 RBF커널같은
경우, ϕ가 실제로 끝이 없는 차원 공간에 각각의 학습 인스턴스를 맵핑하기 때문에 그래서 실제로 맵핑이 어떻게
진행되는 지 모르더라도 괜찮은 것이다.

일부 자주 사용되는 (시그모이드 커널 같은)커널들은 머서의 조건을 충족시켜주는 것은 아니지만 실제에서도 그래도 잘
일반화를 해준다.
```
그래도 아직 부족한 부분이 많다. [공식 5-7](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#equation-5-7-%EC%8C%8D%EB%8C%80-%EB%AC%B8%EC%A0%9C-%EB%B6%80%ED%84%B0-primal-solution%EA%B9%8C%EC%A7%80)은 선형 SVM 분류 모델의 경우에서 쌍대 솔루션에서 primal 솔루션으로 어떻게 가는지 보여주고 있지만, 만약 커널 기법을 적용하고 싶다면 결국 ϕ(x^(i))를 포함하는 공식으로 도달할 것이다. 사실, ŵ값은 ϕ(x^(i))와 같은 차원수를 가져야만 하고, 이는 아주 커질 것이고 어쩌면 무한대에 가까울 수도 있기에 이를 연산할 수는 없다. 하지만 ŵ값을 모르더라도 예측결과를 낼 수 있을까? 좋은 소식은 새로운 인스턴스 x^(n)에 대해서 [공식 5-7](https://github.com/Hahnnz/handson_ml-Kor/wiki/Chapter-5.-Support-Vector-Machine#equation-5-7-%EC%8C%8D%EB%8C%80-%EB%AC%B8%EC%A0%9C-%EB%B6%80%ED%84%B0-primal-solution%EA%B9%8C%EC%A7%80)에서 의사결정함수로 ŵ에 대한 공식을 꽂을수 있다는 것이고, 우리는 입력 벡터 사이에 오직 내적만 있는 공식을 얻게되는 것이다. 이는 다시한번 커널 기법을 사용할 수 있게끔 해준다.
###### Equation 5-11. 커널화된 SVM으로 예측하기
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-11.png)

서포트 벡터에 대해서만 α^(i)≠0이기 때문에 예측값을 내는것은 오직 서포트 벡터에만 새로운 입력 벡터 x^(n)의 내적연산이 수반되며, 학습 인스턴스에 대해서는 전혀 수행을 하지 않는다. 당연히 같은 기법을 사용해서 편향식 b̂값을 연산해주어야 한다.  
###### Equation 5-12. 커널 기법을 사용한 편향식 연산
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-12.png)

머리가 아프기 시작하는게 정상이다. 불행하게도 이게 커널 트릭의 최대의 단점이라고 볼 수 있다.
## 실시간 SVM (Online SVM)
이 Chapter를 마무리 짓기 전에, 실시간 SVM 분류 모델을 살펴보자. (우리가 맨 앞에서 다루었던 실시간 학습, Online Learning과 비슷한 것이다) 선형 SVM 분류모델에 대해서 한가지 방법은 아래의 손실함수를 최소화해주는 (`SGDClassifier`를 사용해서) 경사 하강법을 사용하는 것이다. 이는 Primal 문제에서 도출된 것이다. 불행히도 QP에 기반으로 한 방법보다 더 느리게 수렴한다는 것이다.
###### Equation 5-13. 선형 SVM 분류모델 손실 함수
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/Eq5-13.png)

손실함수의 첫번째 총합은 학습 모델이 큰 마진 값을 가지도록 하는 작은 가중치 벡터값 w를 가지게 해준다. 두번째 총합은 전체 마진 위반의 총합을 연산한다. 인스턴스의 마진 위반은 올바른 쪽 클래스에 있거나 마진 사이를 벗어나 있으면 0값을 가지고 마진 사이 안쪽에서 올바른 클래스쪽에 있는 거리값과 비례하게 연산된다. 식을 최소화하는 것은 모델이 마진 위반을 작게, 적게 가지도록 해준다.
```
Hinge Loss

함수 * max *(0,1-t)를 *Hinge Loss*함수라고 한다. t ≥ 1 이상이면 0이다. 이 함수의 도함수(기울기)는 
t <1이면 -1, t>1이면 0이다. t가 1인 지점에서는 미분을 할 수 없으며, 라소 회귀(Lasso Regression)에서
다루었던 것처럼 t=1인 지점의 2차도함수를 구하여 계속 사용할 수 있다. 
```
###### Hinge Loss Graph
![](https://github.com/Hahnnz/handson_ml-Kor/blob/master/Book_images/05/HingeLoss.png)

["Incremental and Decremental SVM Learning"](http://goo.gl/JEqVui)나 ["Fast Kernel Classifier with Online and Active Learning"](https://goo.gl/hsoUHA)를 사용해서 실시간 커널화된 SVM을 구현하는 것도 가능하다. 하지만 이는 Matlab과 C++로 구현이 되어있다. 거대한 크기의 비선형 문제에 대해서는 우리는 아마 신경망(Neural Network를 사용해서 구현하는 것을 고려할수도 있다. (제 2장에서 다룰 예정이다)

**[뒤로 돌아가기](https://github.com/Hahnnz/handson_ml-Kor/)**
