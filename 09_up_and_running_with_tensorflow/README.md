=============**미완성**=============
======
Chapter 9. 텐서플로우 설치 및 실행
=====
텐서플로우(*Tensorflow*)는 수많은 연산 기능을 제공해주는 오픈소스 소프트웨어로, 특히나 거대한 크기의 기계학습에 대해서 잘 학습시켜주며 파라미터를 잘 조율해준다. 텐서플로우의 기본 원리는 아주 간단하다. 아래의 그림과 같은 수행할 연산을 그래프로 파이썬을 통해 정의한다음에, 텐서플로우가 최적화된 C++코드를 통해서 이 그래프를 받아서 효율적으로 실행시켜준다. 
###### 그림 9-1. 간단한 연산 그래프
![](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/Book_images/09/9-1.png)

가장 중요한 것은, 아래의 그래프에서 보여주는 것처럼, 그래프의 연산과정을 일부분으로 쪼개어 다중 CPU나 GPU를 사용해서 병렬로 연산이 가능하다는 점이다. 텐서플로우는 또한 연산을 분산시키는 것도 가능해서 수백개의 서버를 통해 연산을 나누어 수행함으로써 커다란 학습 데이터 세트에 대해서 거대한 신경망을 학습시킬 수 있다.(Chapter 12에서 다룰 예정이다) 텐서플로우는 각각의 수백만의 특징값들을 가진 몇십억의 인스턴스로 구성된 학습 데이터 세트에 대해서 수백만개의 파라미터를 가진 신경망을 학습시킬 수 있다. 이것은 그렇게 놀라운 일이 아닌데, 텐서플로우는 구글의 브레인팀에서 개발된 것이며, 이것으로 구글의 거대한 서비스, 예를들면, 구글 클라우드 스피치, 구글 사진, 구글 검색등을 제공했기 때문이다.
###### 그림 9-2. 다중 CPU/GPU/Server를 통한 병렬 연산수행
![](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/Book_images/09/9-2.png)

2015년도 11월에 텐서플로우가 처음 오픈소스로 공개되었을때, 그 전에도 딥러닝을 수행하기 위한 오픈소스소프트웨어는 이미 많이 있었다.(도표 9-1을 참고하자) 그리고 이미 텐서플로우에서 제공하는 것들을 다른 라이브러리들에서도 제공하고 있었다. 그럼에도 불구하고, 텐서플로우의 깔끔한 디자인과 확장성 그리고 유연성, 좋은 자료들이 텐서플로우를 기존의 제공되던 라이브러리들을 재치고 정상에 설 수 있었다. 다음은 텐서플로우의 장점이다.
* Windows, Linux, macOS에서뿐만 아니라, iOS나 Android같은 휴대기기에서도 작동한다.
* Scikit-Learn에 견줄만한 TF.Learn(`Tensorflow.contrib.learn`)이라고 하는 아주 간편한 파이썬 API를 제공한다. 앞으로 볼 것이지만, 코드 몇줄만으로도 매우 다양한 종류의 신경망들을 사용할 수 있다. 이는 이전에 Scikit Flow(skflow)라고 불리던 독립적인 프로젝트였다. 
* 신경망 구현, 학습, 평가과정을 간편화한 TF-slim(`tensorflow.contrib.slim`)이라고 하는 또 다른 간편한 API를 제공해준다.
* [케라스(Keras)](http://keras.io)나 [Pretty Tensor](https://github.com/google/prettytensor/)같은 일부 다른 높은 레벨의 API는 텐서플로우을 백그라운드로 사용해 구현되어있다.
* 텐서플로우의 메인 API인 Python API는 우리가 구상할 수 있는 어떠한 신경망 구조도 포함하여 모든 종류의 연산을 구현하기 위한 유연성이 아주 좋다.
* 신경망을 구현하기 위해 필요한 기계학습 알고리즘을 C++로 구현하여 아주 효율적이며, 우리가 필요한 연산을 C++을 통해서 구현해 추가를 해줄수도 있다.
* 손실함수를 최소화하는 파라미터를 찾기위한 최신의 최적화 노드를 일부 제공한다. 텐서플로우는 우리가 정의한 그래디언트를 연산해주는 함수를 자동으로 관리해주기 때문에, 사용하기가 아주 쉽다. 이를 자동미분기(*Automatic differentiation* , *autodiff*)이라고 한다. 
* 학습 곡선 등등을 보여주는 연산 그래프를 인터넷 브라우저 상으로 볼 수 있는 텐서보드(*Tensorboard*)라고 하는 훌륭한 시각화 도구를 제공한다.
* 구글은 [텐서플로우 그래프](https://cloud.google.com/ml)를 클라우드 서비스로 제공하기도 한다.
* 마지막에서 말하지만 아주 중요한 것으로, 텐서플로우에는 열정적이면서 도움을 주는 개발자들의 헌신적인 시간들과 이를 개선시키기 위해 기여를 해준 여러 커뮤니티들이 있다. 이는 Github상에서 가장 인기있는 오픈소스 프로젝트 중 하나이며, 더욱 더 대단한 프로젝트들이 텐서플로우를 기반으로 하여 구현되어져있다. (예로, Resource페이지인 https://www.tensorflow.org/ 나 https://github.com/jtoy/awesome-tensorflow 를 참고하라) 기술적인 질문들을 묻고자한다면, http://stackoverflow.com/ 을 이용하고, "Tensorflow"라는 태그를 달아서 질문을 해보자. Github를 통해서도 질문을 할 수 있다. 좀 더 일반적인 논의사항들에 대해서는 [구글 그룹](http://goo.gl/N7kRF9)에 참여하는 것이 좋다.

이번 Chapter에서는 텐서플로우의 기초에 대해서 다룰 것이며, 설치부터 시작해 구현, 실행, 저장, 그리고 간단하게 연산 그래프를 시각화하는 방법도 배워볼 것이다. 다음 Chapter부터 나올 신경망을 구현하기 위해서는 이번 Chapter에서 소개되는 기초를 정복하는 것이 아주 중요하다. 
###### 도표 9-1. 딥러닝 라이브러리 오픈소스 (완벽한 것은 아님)
 라이브러리 | API | 플랫폼 | 제작자 | 출시년도   
 :-----: | :-: | :--: | :---: | :----: 
 Caffe | Python, C++, Matlab | Linux,macOS,Windows | Y.Jia, UC Berkeley (BVLC) | 2013
 DeepLearning4j | Java, Scala, Clojure | Linux,macOS,Windows,Android | A.Gibson, J.Patterson | 2014
 H2O | Python, R | Linux,macOS,Windows | H2O.ai | 2014
 MXNet | Python, C++, Others | Linux,macOS,Windows,iOS,Android | DMLC | 2015
 Tensorflow | Python, C++ | Linux,macOS,Windows,iOS,Android | Google | 2015
 Theano | Python | Linux,macOS,iOS | University of Montreal | 2010
 Torch | C++, Lua | Linux,macOS,iOS,Android | R.Collobert, K.Kavuksuoglu, C.Farabet | 2002 

# 설치
이제 설치를 시작해보자. Chapter2에 있는 설치 가이드에 따라 Scikit-Learn과 Jupyter를 설치했다고 가정하고, pip를 사용해 텐서플로우를 설치해볼 것이다. 만약 virtualenv로 가상환경을 새로이 만들었다면, 먼저 그 환경을 활성화해줄 필요가 있다.
```
$ cd $ML_PATH #기계학습이 실행될 수 있는 가상환경 디렉토리
$ source env/bin/activate
```
다음으로, 텐서플로우를 설치해보자
```
$ pip3 install --upgrade tensorflow
```
```
GPU를 지원하는 텐서플로우를 설치하고 싶다면 tensorflow 대신 tensorflow-gpu를 설치하라.
Chapter12에서 더 자세하게 이를 다룰 것이다.
```
설치가 되었는지 테스트하려면 아래의 명령어를 입력해보자. 아래의 코드는 텐서플로우의 버전을 출력해주어야만 한다.
```
$python3 -c 'import tensorflow; print(tensorflow.__version__)'
1.4.1    # 2018.01.22 기준
```
# 첫번째 그래프를 만들어보고 세션에서 이를 실행시켜보자
아래의 코드는 앞에서 그림 9-1을 보여준 그래프를 생성하는 코드이다.
```
import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```
이게 다이다! 가장 중요한 것은 실제로는 연산을 수행하는 것처럼 보이지만 이 코드가 실제로 어떠한 연산도 수행하지 않는다는 것이다. 이는 단지 그래프만 만들어줄 뿐이다. 사실은 아직 변수들이 초기화되어있지 않다. 그래프를 평가하기 위해서는 텐서플로우 *session*을 열어주어야하고 이를 사용해서 변수들을 초기화 해준 다음에 f함수를 평가하게된다. 텐서플로우 세션은 CPU나 GPU와 같은 장치 상에서 함수들이 자리를 잡을 수 있도록 관리해주고 이를 실행시켜주며, 모든 변수의 값들을 관리한다. 아래의 코드는 세션을 만들어주고, 변수들을 초기화해주며, f함수를 평가해주고 세션을 닫아준다. 
```
>>> sess = tf.Session()
>>> sess.run(x.initializer)
>>> sess.run(y.initializer)
>>> result = sess.run(f)
>>> print(result)
42
>>> sess.close()
```
매번 `session.run()`을 반복하도록 하는 것은 약간 다루기 힘들어지지만 다행히 좀 더 나은 방법이 있다.
```
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
```
`with`블럭 단위 안에는 세션이 기본 세션값으로 설정되어있다. `x.initializer.run()`을 호출하는 것은 `tf.get_default_session().run(x.initializer)`를 호출하는 것과 같은 것이며, 비슷하게 `f.eval()`은 `tf.get_default_session().run(f)`를 호출하는 것과 같은 행위이다. 이는 코드를 읽기 더 쉽게 만들어준다. 더 나아가, 세션은 블럭이 끝나게 되면 자동적으로 닫히게 된다.

모든 단일 변수들에 대해서 초기화함수를 손수 직접 실행시키기 보다는, `global_variables_initializer()`함수를 사용할 수 있다. 이는 실제로 초기화를 즉시 수행해주는 것은 아니지만, 이를 실행시킬때, 모든 변수들을 초기화하는 그래프속 노드를 만들어준다.
```
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
```
Jupyter나 파이썬 shell안에서는 `InteractiveSession`을 만들어지는 것을 좋아할 것이다. 일반적인 `Session`과 다른점은 `InteractiveSession`이 만들어질 때 자동으로 자기 자신을 기본 세션으로 설정하는 것인데, 그래서 `with`블럭을 만들 필요가 없다는 것이다. (하지만 세션이 모두 끝나면 직접 손수 닫아주어야할 필요가 있다) 
```
>>> sess = tf.InteractiveSession()
>>> init.run()
>>> result = f.eval()
>>> print(result)
42
>>> sess.close()
```
텐서플로우 프로그램은 전형적으로 두가지 파트로 나누어진다. 첫번째 파트는 연산 그래프를 구현하는 것이고(이를 구성 단계, *Construction phase* 라고 한다), 두번째 파트는 (이를 실행 단계, *execution phase*라고 한다) 구성단계에서는 전형적으로 기계학습 모델과 학습이 필요한 연산을 표현해주는 연산 그래프를 구현한다. 실행 단계에서는 일반적으로 반복적으로 학습 스탭(예를들면 mini-batch당 한 스탭)을 평가하는 루프를 실행한다. 이제 짧게 예시를 한번 보자.
# 그래프 관리하기
우리가 만드는 어떠한 노드는 자동적으로 기본값으로 설정된 그래프에 추가된다.
```
>>> x1 = tf.Variable(1)
>>> x1.graph is tf.get_default_graph()
True
``` 
이러한 경우는 괜찮지만, 때때로 우리는 여러개의 독립적인 그래프에서 메세지를 받기를 원할수도 있다. 다음과 같이 새로운 `Graph`를 만들어서 `with`블럭에 기본 값으로 임시로 기본값의 그래프로 설정해주는 것으로 이를 할 수 있다.
```
>>> graph = tf.Graph()
>>> with graph.as_default():
...     x2 = tf.Variable(2)
...
>>> x2.graph is graph
True
>>> x2.graph is tf.get_default_graph()
False
```
```
Jupyer(혹은 파이썬 shell)에서는 실험을 하는 중에는 한번보다는 여러번 같은 명령어를 실행하는 것이
일반적이다. 결국에는 중복된 노드들을 많이 포함한 기본값 그래프를 보게될 것이다. Jupyter 커널 혹은
파이썬 shell을 재시작하는 것이 하나의 솔루션이 될 수도 있지만, 좀 더 편안한 솔루션은 그냥 기본값의
그래프를 "tf.get_default_graph()"를 실행해서 초기화하는 것이다.
```
# 노드 값의 라이프사이클
노드를 평가할 때, 텐서플로우는 자동으로 노드가 의존하고 있는 노드들의 집합을 정의하는데 이러한 노드들을 먼저 평가하게 된다. 예시로 다음의 코드들을 생각해보자.
```
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15
```
먼저 이 코드는 매우 간단한 그래프틀 정의한 것이다. 이제 세션을 시작하고 y값을 평가하기 위한 그래프를 실행시켜보자. 텐서플로우는 자동적으로 x에 y가 의존적이라는 것을 발견할 것이고, 이는 w에도 의존적이다. 그래서 평가는 w를 먼저 수행하고, 그다음에 x, 그다음에 y를 평가하여 끝에는 y의 값을 출력하게된다. 마지막으로 이 코드는 z를 평가하기 위해서 그래프를 실행하게된다. 다시한번 말하자면 텐서플로우는 w와 x를 먼저 평가해야한다는 것을 찾아낸다. 꼭 알아야 할 것은 이전에 평가된 w와 x의 결과값을 다시 사용하지는 않는다는 것이다. 짧게 말하면, 실행되는 코드는 w와 z를 두번 평가하게 된다. 

변수의 값을 제외하고 모든 노드의 값들은 그래프가 실행되는 사이에 버려지게 되는데, 이는 그래프가 실행되는 세션에 의해서 유지보수되는 것이다. (Queue와 reader도 어떠한 경우에서는 유지보수되는데, 이는 chapter12에서 보게될 것이다) 변수는 초기화 함수가 실행될 때, 그 삶을 시작하게되며, 세션이 닫히면 죽게된다. 

만약 이전의 코드에서 w와 x를 두번 평가하는 것과는 달리 y와 z를 좀 더 효율적으로 평가하고 싶다면, 아래의 코드처럼 텐서플로우에게 그래프 실행 한번에 y와 z를 평가하도록 요청해야한다.
```
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15
```
```
단일 텐서플로우 세션에서는 아무리 같은 그래프를 재사용하는 것일지라도 다중 세션이 어떤 상태에 대해서 공유를
할 수가 없다.(각각의 세션이 모든 변수들에 대해서 자신만의 복제본을 가지고 있기 때문이다) 분산된 환경의 
텐서플로우(chapter12를 참고하라)에서는 변수의 상태를 세션이 아닌 서버에 저장을 하기 때문에, 다중 세션은
같은 변수를 공유할 수 없다.
```
# 텐서플로우를 사용한 선형 회귀 모델
텐서플로우 함수(*ops* 라고 짧게 불린다)는 입력의 수를 받아서 출력의 수만큼 생성을 한다. 예를들어, 덧셈과 곱셈 연산(ops)은 각각 2개의 입력을 받아서 하나의 값을 출력한다. 상수와 변수는 입력을 받지 않는다.(이를 *source ops* 이라고 한다) 입력과 출력은 보통 다중차원의 배열인 형태로 흔히 텐서 *Tensor* 라고한다.(그래서 이름이 텐서 플로우(*Tensor flow*)이다) NumPy 배열 같이, 텐서도 자료형과 크기를 가지고 있다. 사실 파이썬 API에서 텐서는 간단하게 NumPy의 ndarray로 쉽게 나타낼 수 있다. 전형적으로 자료형은 float를 취하지만 String으로 수행하도록 사용할 수도 있다.

아래의 예시는 텐서가 단일 스칼라 값을 가지고 있지만, 당연히 어떤 크기의 배열이라도 연산을 수행할 수 있다. 예로, 아래의 코드는 Chapter2에서 사용했던 캘리포니아 하우징 데이터세트에 대해서 선형 회귀모델을 수행하기 위해서 2차원 배열을 다룬다. 먼제 데이터를 가지고오고, 모든 학습 인스턴스에 대해서 입력 특징값에 추가로 편향치(x0=1)를 붙여준다. 그리고나서 데이터와 타겟값이 들어가도록 텐서플로우 상수 노드 X와 y를 만들고, `theta`를 정의하기 위한 텐서플로우에서 제공하는 행렬연산을 사용한다. `transpose()`, `matmul()`,`matrix_inverse()`같은 행렬 함수들은 따로 설명이 필요없지만, 보통 이들은 연산을 즉시 수행하지는 않는다. 대신에 이들은 그래프가 실행될 때 수행할 것인 노드를 그래프에 생성해준다. 정규식(Chapter4에서 본 것과 같이 ![](https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cmathbf%7B%5Ctheta%7D%7D%20%3D%20%28%5Cmathbf%7BX%7D%5ET%20%5Ccdot%20%5Cmathbf%7BX%7D%29%5E%7B-1%7D%20%5Ccdot%20%5Cmathbf%7BX%7D%5ET%20%5Ccdot%20%5Cmathbf%7By%7D&mode=inline))에 상응하는 `theta`를 정의할 수 있는지를 보고싶을 것이다. 결국에 코드는 세션을 만들어서 `theta`를 평가할 것이다.
```
import numpy as np
from sklearn.datasets import fetch_california_housing

reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```
이 코드가 NumPy를 사용해서 직접적으로 정규식을 연산하는 것에 비해 얻을 수 있는 주된 장점은 만약 GPU를 가지고 있다면 텐서플로우가 알아서 GPU상으로 올릴 수 있도록 해준다는 것이다.(GPU를 지원하는 텐서플로우 버전을 설치한 경우에만 해당하며, chapter 12에서 더 자세한 사항을 볼 수 있을 것이다)
# 경사하강법 구현하기
정규식 대신에 이번엔 Chapter 4에서 소개되었던 Batch Gradient를 사용해서 구현해보자. 먼저 그래디언트를 손수 계산하는 것으로 구현해보고, 텐서플로우가 그래디언트를 자동으로 연산해주게하는 텐서플로우의 autodiff를 사용해 구현하고, 마지막에는 텐서플로우의 독창적인 최적화함수 두개를 사용해볼 것이다.
```
경사하강법을 사용할 때는, 입력 특징값 벡터들을 먼저 정규화해야하는 것이 중요하다는 것을 기억하라. 아니면
학습이 느려질 수도 있다. 이를 텐서플로우나 NumPy, Scikit-Learn의 "StandardScaler"나 여러분이
좋아하는 다른 종류의 솔루션을 사용해서 이를 수행해줄 수 있다. 이제부터 등장할 코드는 정규화작업을 이미
수행하였다고 가정하고있다.
```
## 손으로 일일이 그래디언트 연산하기
아래의 코드는 몇가지 새로운 요소들을 제외하고는 설명이 따로 필요없다.
* `random_uniform()`함수는 NumPy의 `rand()`함수처럼 주어진 크기와 범위 내에서, 임의의 값들을 포함하고 있는 텐서를 만드는 노드를 그래프상에 만들어준다.
* `assigin()`함수는 한 변수에 새로운 값을 할당하는 노드를 만들어준다. 이번 경우에서는 배치 경사하강법![](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7B%5Ctheta%7D%5E%7B%28%5Ctext%7Bnext%20step%7D%29%7D%20%3D%20%5Cmathbf%7B%5Ctheta%7D%20-%20%5Ceta%20%5Cnabla_%7B%5Cmathbf%7B%5Ctheta%7D%7D%5C%2C%20%5Ctext%7BMSE%7D%28%5Cmathbf%7B%5Ctheta%7D%29&mode=inline)을 구현한다.
* 메인 루프는 (`n_epochs` 만큼) 학습 스탭을 계속계속 실행하면서 매 100번 반복마다 현재의 평균제곱근오차(`mse`)를 출력해준다. 매 반복 번수마다 MSE가 감소하고 있는것을 볼 수 있어야한다.
```
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
```
## 텐서플로우의 `autodiff` 사용하기
앞의 코드는 잘 실행되지만, 손실함수(MSE)로 부터 그래디언트를 수학적으로 끌어내줄 필요가 있다. 선형 회귀같은 경우, 꽤 쉽지만, 심층신경망 같은 경우에는 골치아플수가 있다. 편미분으로 공식을 자동으로 찾는 기호 미분(*Simbolic differentiation*)을 사용할 수 있다. 하지만 결과로 나오는 코드는 매우 효율적이지 못하다.

왜인지 이해하기 위해서는 함수 `f(x) = exp(exp(exp(x)))`를 보자. 만약 미적분학을 알고있다면 도함수는 `f'(x) = exp(x) X exp(exp(x)) X exp(exp(exp(x)))`라는 것을 알 수 있다. 만약 f(x)와 f'(x)를 따로따로 보이는대로 코드로 만든다면, 그 코드는 상당히 비효율적일 것이다. 더 효율적인 솔루션은 첫번째 `exp(x)`를 연산한 다음에 `exp(exp(x))`를 연산하고, `exp(exp(exp(x)))`를 연산해서 이 세개를 모두 출력시켜주는 것이다. 이는 바로 f(x)의 답(세번째 식)을 바로 줄 수 있다. 그리고 미분값 f'(x)이 필요하다면 세개의 식을 모두 곱하면 끝난다. 나이브 접근법으로는, 우리는 f(x)와 f'(x) 모두 연산하기 위해서 `exp`함수를 9번 호출해야하는데, 이러한 접근법이라면 `exp`함수를 3번만 호출하면 끝난다.

이는 우리가 함수를 작성할 때 아무 코드로나 정의하면 성능이 나빠진다는 것이다. 다음의 함수에 대한 편미분을 계산할 수 있는 공식 (혹은 코드)를 찾아낼 수 있는가? 힌트: 그냥 시도도 하지말라.
```
def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z
```
다행히 텐서플로우 autodiff는 그래디언트를 자동으로, 그리고 효과적으로 연산해준다. `gradients = ...`줄을 다음의 줄로 이전의 섹션에 있는 경사하강법 코드를 바꾸어주고 다시 그 코드를 실행시켜보면 잘 작동할 것이다.
```
gradients = tf.gradients(mse, [theta])[0]
```
`gradients()`함수는 op(이번 경우에서는 mse)와 변수의 리스트 (이번 경우에서는 theta)를 받아서, 각각 번수에 대해서 op에대한 그래디언트를 연산하기 위해 변수 하나마나 op들을 적용한 리스트를 만들어낸다. 그래서 `gradients`노드는 theta에 대한 MSE의 그래디언트 벡터를 연산한다.

그래디언트를 자동으로 연산하는 네가지의 주된 접근법이 있다. 이들은 아래의 도표에 정리되어있다. 텐서플로우는 `reverse-mode autodiff`를 사용하는데, 이는 신경망 같이 입력은 많으나 출력하는 결과가 적을때 완벽(효율적이며 정확)하다. 모든 입력 값들에 대해서 결과에 대한 편도함수들 연산한다.
###### 도표 9-2. 그래디언트를 자동으로 연산해주는 메인 솔루션
 기법 | 모든 그래디언트를 연산하기 위한 그래프 트레버셜 Nb | 정확도 | 임의코드지원 | 코멘트 
 :-: | :-: | :--: | :-: | :--:
 수치 편미분 | n_input +1 | Low | Yes | 구현이 쉬움
 기호 편미분 | N/A | High | No | 매우 다양한 그래프를 구현함
 전진모드 autodiff | n_input | High | Yes | Use Dual numbers
 역행모드 autodiff | n_input +1 | High | Yes | 텐서플로우에 구현되어 있음

만약 이런 마법이 어떻게 작동하는지 알고싶다면 부록D를 확인하자.

## 최적화 함수 (Optimizer) 사용하기
텐서플로우에서 그래디언트를 알아서 계산해준다. 하지만 이는 더 간소화될 수 있는데, 경사하강법 최적화 함수들을 포함해서 몇가지 독특한 최적화 함수를 제공한다. `gradients = ...`를 아래와 같이 `training_op = ...`로 바꾸어 사용헤보면 다시 모든것이 잘 작동할 것이다.
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```
만약 다른 종류의 최적화 알고리즘을 사용하고 싶다면 간단하게 한줄만 바꾸어주면 된다. 예로, Optimizer를 다음과 같이 정의하면 (Chapter 11에서 보겠지만 경사하강법보다 더 빠르게 수렴하는)모멘텀 최적화 알고리즘을 사용할 수 있다.
```
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
```
# 학습 알고리즘에 데이터 입력하기
이제 이전의 코드를 수정해서 미니 배치 경사하강법을 구현해보자. 이를 위해서 다음 미니 배치로 매 반복마다 X와 y를 대체해줄 방법이 필요하다. 가장 간단한 방법은 placeholer 노드를 사용하는 것이다. 이런 노드들은 매우 특별한데, 이러한 노드들은 어떠한 연산도 수행하지 않기 때문이다. 이런 노드들은 실행하는 중에 결과를 출력하라고 하는 데이터를 출력할 뿐이다. 즉 플레이스 홀더라는 것은 이름에서도 의미하듯이 텐서 객체인것 같이 행동하지만 노드로 생성될 때 값을 가지지는 않는다. 대신에 실행시에 입력될 텐서를 위한 자리(place)를 가지고(holder)있다. 이는 입력 노드와 같은 효과를 가지는 것이다. 이런 노드들은 전형적으로 학습하는 중에 텐서플로우에 학습 데이터가 지나가도록 하는데 쓰이고 있다. 만약 placeholder가 실행도중에 값을 정할 수 없다면, 에러를 발생시켜 예외처리될 것이다. 

placeholder 노드를 생성하기 위해서, `placeholder()`함수를 호출해야하고, 출력될 텐서의 자료형을 명시해주어야만 한다. 추가적으로 그 크기도 명시를 해줄 수 있다. 차원수로 `None`이라고 명시하게된다면, 이는 어떠한 크기라도 받아들이겠다는 뜻이다. 예로들어서, 다음의 코드는 A라는 플레이스홀더 노드를 생성하고 `B = A + 5`라는 노드도 만든다. 우리가 B를 평가하고자 한다면, A의 값을 명시하는 `eval()` 메소드를 `feed_dict`을 통해서 전달될 것이다. A는 랭크-2를 가질 것이고(2차원이여만 한다) 3개의 Column이 있어야하지만 row의 수는 상관없다.
```
>>> A = tf.placeholder(tf.float32, shape=(None, 3))
>>> B = A + 5
>>> with tf.Session() as sess:
...     B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
...     B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
... 
>>> print(B_val_1)
[[ 6.  7.  8.]]
>>> print(B_val_2)
[[  9.  10.  11.]
 [ 12.  13.  14.]]
```
```
실제로 플레이스홀더만이 아니라 어떠한 연산의 결과도 입력할 수 있다. 이번의 경우에서는
텐서플로우는 이러한 연산을 평가하지않는다. 단지 값들을 입력할 뿐이다.
```
미니 배치 경사하강법을 구현하기 위해서는 이미 존재하는 코드를 조금 만져보기만 하면 된다. 먼저 구성 단계에서 X와 y를 플레이스홀더 노드로 변경해주기 위해서 이들의 정의를 바꾸어주자.
```
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
```
그리고나서 배치 사이즈를 정의하고 전체 배치의 수에 대해서 연산해보자
```
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
```
마지막으로, 실행단계에서, 미니 배치를 하나씩 가져와서 이들중 어느 한쪽에 의존적인 노드를 평가할 때 `feed_dict`파라미터를 통해서 X와 y에 값을 공급한다.
```
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
```
```
theta를 평가할 때에는 X와 y에 값을 입력해주지않아도 되는데, 저 둘 중 어디에도 의존적이지 않기때문이다. 
```
# 모델 저장하기 및 복원하기
한번 모델을 학습시키고 나면 디스크에 이 파라미터의 값들을 저장해서 우리가 원할때마다 불러오거나 다른 프로그램에 사용하거나 다른 모델과 비교하는데 사용할 수 있다. 더 나아가 아마 학습중에 중간마다 체크포인트를 저장해서 학습이 도중이 멈추더라도 처음부터 다시시작하는 것보다는 최근 체크포인트부터 학습을 다시하는 것을 더 좋아할 것이다.

텐서플로우는 모델을 저장하고 다시 불러오기가 너무 쉽다. 구성단계의 끝에서 `Saver`노드를 만들어주기만 하면 된다. 그리고나서 실행 단계에서는 모델의 상태를 저장하고 싶을 때마다 `save()`메소드를 호출하면된다.
```
reset_graph()

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
``` 
모델을 다시 불러오는 것도 간단하다. 전처럼 구성단계에서 `Saver`를 만드는데, 실행단계의 시작부분에서 `init`노드를 사용해서 변수를 초기화해주는 대신에 그 모델을 호출해서 사용하고 `Saver`의 `restore()`를 호출해서 사용한다.
```
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
[...]
```
기본값으로 `Saver`는 저장될 변수 이름으로 모든 변수들의 값을 저장하고 다시 호출해올수 있지만, 좀 더 세세하게 다루고 싶다면, 변수들을 사용해서 어떤 변수를 저장하거나 다시 불러올지를 정해줄 수 있다. 예를들어 다음의 `Saver`는 `weight`라는 이름을 가진 변수에 `theta`만을 저장하거나 다시 불러올 수도 있다.
```
saver = tf.train.Saver({"weights": theta})
```
기본값으로 `save()` 함수도 또한 체크포인트파일명 뒤에 확장자 `.meta`를 붙여주어 그래프의 구조도 저장해줄 수 있다. `tf.train.import_meta_graph()`함수를 사용하여 그래프 구조를 다시 불러올 수 있다. 이렇게 호출된 구조는 기본 그래프에 추가되며, 그래프의 상태를 저장하는데 사용하는 `Saver`인스턴스에 반환해준다. 
```
saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book
```
이는 구현된 코드에서 찾을 필요가 없이 변수의 값과 그래프 구조 모두를 포함해 완전하게 저장된 모델을 불러올 수 있게 해준다. 
# 텐서보드를 사용하여 그래프와 학습 곡선 시각화하기
이제 우리는 미니 배치 경사하강법을 사용하여 선형 회귀 모델을 학습 시키는 연산 그래프를 얻었고, 일정한 간격마다 체크포인트를 저장해두었다. 매우 복잡해보인다. 하지만 우리는 학습중에 시각화를 하기 위해서 `print()`함수를 사용해서 그 내용을 얻어내고있다. 좀 더 좋은 방법이 있다. 바로 텐서보드를 사용해보는 것이다. 만약 일부 학습 관련 정보를 입력하면 텐서보드가 브라우저 상에 학습 곡선과 같은 것들을 학습 관련 정보들을 사용해서 보기 좋게 상호작용적인 그래프로 보여준다. 그래프를 정의를 제공해주면 브라우저 상으로 휼륭한 인터페이스를 제공해줄 것이다. 이는 그래프를 사용하므로 bottleneck를 찾거나 에러를 찾아내는데 아주 유용하다.

첫번째 단계는 텐서보드가 읽을 수 있도록 디렉토리를 그래프의 정의와 예를들면 학습 에러 (MSE)같이 학습 관련정보를 쓸 수 있도록 프로그램을 조금 다시 만져보아야할 필요가 있다. 프로그램을 실행시킬 때마다 매번 로그파일의 이름이 다르도록 해주어야하며 그렇지 못했다면 시각화가 제대로 못이루어질 수가 있다. 가장 간단한 방법은 로그파일 이름에 실행을 시킨 시간을 적어서 저장을 하는 것이다. 
```
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```
그 다음에 다음 코드를 구성단계의 맨 마지막에 추가해주도록 한다.
```
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```
첫번째 줄은 




###### Equation 9-1. 정류된 선형 유닛:ReLU(Rectified linear unit)
![](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/Book_images/09/Eq9-1.png)




























**[뒤로 돌아가기](https://github.com/Hahnnz/handson_ml-Kor/)**
