Machine Learning Notebooks
==========================

###### [원문 페이지 : ageron/handson-ml ](https://github.com/ageron/handson-ml)

이 프로젝트는 파이썬으로 여러분들이 머신러닝의 기초를 배울 수 있도록 하였습니다. [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do) 책의 Example 코드들의 대한 예시 코드들과 솔루션을 포함하고 있습니다.


[![book](http://akamaicovers.oreilly.com/images/0636920052289/cat.gif)](http://shop.oreilly.com/product/0636920052289.do)

[Jupyter](http://jupyter.org/) notebook들을 여러분들이 좋아하는 뷰어로 열어보세요:
* [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb) 사용하기
    * [github.com's notebook viewer](https://github.com/ageron/handson-ml/blob/master/index.ipynb)도 사용할 수 있지만, 주피터 공식 사이트 뷰어보다 느리고 수학 공식들이 제대로 보여지지 않습니다.
* 아니면 이 레포지토리를 다운로드해서 주피터를 직접 실행시켜서 볼 수 있습니다. 이 방법은 여러분들이 직접 코드를 가지고 실행해볼 수 있어 좋습니다. 이러한 경우, 아래의 설치 가이드를 따라 설치하셔서 사용할 수 있습니다.


# 설치
먼저 git이 없다면 설치를 하셔야합니다.

    $ sudo apt-get install git
    
그 다음에, 터미널을 열고 다음 명령어를 입력해서 이 레포지토리를 Clone합니다.

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/handson-ml.git
    $ cd handson-ml
    
만약에 이 책의 16장에 있는 강화학습을 학습하고자 하신다면, [OpenAI gym](https://gym.openai.com/docs)을 설치하고 Atari 시뮬레이션을 지원하기 위한 의존성 라이브러리들을 설치할 필요가 있습니다.

혹시 여러분들이 Python에 익숙하고, Python 라이브러리 설치방법을 알고 계신다면, `requirements.txt`에 정리해놓은 라이브러리들을 설치해주시고 [Jupyter 시작하기](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/README.md#jupyter-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0) 섹션으로 넘어가셔도 좋습니다. 자세한 가이드가 필요하시다면 계속 읽어나가시면 됩니다.

## 파이썬과 필요한 라이브러리들
당연하지만, 여러분들 분명히 파이썬이 필요할겁니다. 요새 대부분의 시스템들에는 Python2가 기본적으로 설치되어있습니다. 어떤 시스템들은 Python3도 미리 설치가 되어있습니다. 다음 명령어를 터미널에 입력하시면 여러분들의 시스템의 Python 버전을 확인하실 수 있습니다.



Python3는 ≥3.5.이라면 어느 버전이든 상관없습니다. 만약 Python3가 없다면, 설치하시는 것을 권장드립니다. (Python 2.6이상에서도 작동하지만, 권장드리지는 않기에, Python3가 바람직합니다.) 이렇게 하시고나면, 여러분들은 몇가지의 선택사항이 있습니다. 윈도우나 Mac OS X 상에서는 [python.org](https://www.python.org/downloads/)에 가서 다운로드만 하시면 됩니다. Mac OS X에서는 [MacPorts](https://www.macports.org/)나 [Homebrew](https://brew.sh/)를 대신 사용할 수 있습니다. 리눅스 상에서는 어떻게 해야할 지 알고있는 것이 아니라면, 여러분들 시스템에서 제공하는 패키징 시스템을 사용해야만합니다. 예를들어 Ubuntu나 Debian상에서는 다음과 같이 입력하셔야 합니다.

    $ sudo apt-get update
    $ sudo apt-get install python3

또 다른 선택 사항으로는 [Anaconda](https://www.continuum.io/downloads)를 설치하는 것입니다. 이는 Python 2, 3버전 모두와, 많은 체계적인 라이브러리들을 포함한 패키지입니다. 여러분들은 Python3버전을 사용하시는 것을 권장드립니다.

만약 Anaconda를 사용하실 것이라면,ㅜ 다음 섹션을 읽어주시고, 아니라면 [pip 사용하기](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/README.md#pip-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0) 섹션으로 넘어가주세요

## Anaconda 사용하기
Anaconda를 사용할 때, 여러분들은 이 프로젝트에 맞는 독단적인 Python 환경을 추가적으로 구성해야합니다. 이는 각각의 프로젝트에 대하여 매우 다른 라이브러리들과 그 버전들로 각각 다른 환경을 구성해야하기 때문에 추천드립니다.

    $ conda create -n mlbook python=3.5 anaconda
    $ source activate mlbook

이는 `mlbook`이라는 Python 3.5 환경을 새롭게 만들어 준 것이며(원하신다면 여러분들이 원하시는 이름들로 바꾸어 사용하실 수 있습니다), 이 환경을 활성화 한 것입니다. 이 환경은 Anaconda에 달려있는 모든 체계적인 라이브러리들을 포함하고 있습니다. 우리가 필요로하는 (NumPy, Matplotlib, Pandas, Jupyter 그리고 몇가지 다른 것들) 모든 라이브러리들을 포함하고 있습니다. 하지만 TensorFlow는 깔려있지않습니다. 그러므로 다음의 명령어를 실행해줍니다.

    $ conda install -n mlbook -c conda-forge tensorflow=1.0.0

이 명령어는 `mlbook` 가상환경 안에 TensorFlow 1.0.0을 설치합니다 (`conda-forge`라는 레포지토리에서 가져옵니다). `mlbook` 환경을 설치하고 싶지 않으시다면, `-n mlbook` 옵션을 제거하고 사용하시면 됩니다.

그 다음으로, 여러분들은 Jupyter 확장판을 설치할 수 있습니다. 

    $ conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions

모든 준비가 끝났습니다! 다음으로, [Jupyter 시작하기](https://github.com/Hahnnz/Hands_on_ML-Kor/blob/master/README.md#jupyter-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)로 넘어가주세요.

## pip 사용하기
만약 Anaconda를 사용하시는 것이 아니라면, 이 프로젝트를 위해 특히 필요한 NumPy, Matplotlib, Pandas, Jupyter 그리고 TensorFlow (몇 가지 다른 것들도 )Python 라이브러리들을 설치해야합니다. 이를 위해서는, 파이썬의 집적 패키징 시스템, pip를 사용하거나, 아니면 여러분들 시스템만의 패키징 시스템을 사용해야할 지 모릅니다. (예를들어 Mac OS X의 Macport나 homebrew) pip 사용의 장점은 각각의 다양한 라이브러리들과 다양한 버전들로 여러 개의 독단적인 파이썬 가상환경을 만들기 쉽다는 점입니다. (예를 들면 한 프로젝트당 한 환경을 구현하는 것이죠.) 여러분들 시스템 고유의 패키징 시스템 사용의 장점은 여러분들의 파이썬 라이브러리들과 여러분들 시스템의 다른 패키지들 사이에 충돌을 일으킬 위험을 덜 수 있다는 것입니다. 저는 다양한 라이브러리 요구 사항들로 수많은 프로젝트를 수행해왔기 때문에, 저는 pip를 사용한 독단적인 환경을 구성하는 것을 선호합니다. 

만약 여러분들이 pip를 사용해서 요구 라이브러리들을 설치하기를 원하신다면 터미널에 다음과 같은 명령어를 입력해야합니다.
주의 : 만약 여러분들이 Python3가 아닌 Python2를 사용하고 있다면, 다음의 모든 명령어들에서 `pip3`는 `pip`로, `python3`는 `python`로 바꾸어서 사용하셔야만 합니다. 

먼저 최신 pip 버전을 설치해줍니다.

    $ pip3 install --user --upgrade pip

`--user`옵션은 오직 현재 유저의 pip만 최신 버전으로 설치를 해줍니다. 만약 시스템 전체(모든 유저들)의 pip를 최신 버전으로 설치하고 싶다면, 슈퍼유저 권한이 있어야만 합니다. (예로 리눅스 상에서 `pip3`을 대신해 `sudo pip3` 사용해야함) 그리고 `--user`옵션을 제거해야하죠. 그리고 아래의 `--user` 옵션을 사용하는 명령어들 또한 같습니다. 

다음으로 여러분들은 독단적인 환경을 추가적으로 구성해야합니다. 이는 각각의 프로젝트에 대하여 매우 다른 라이브러리들과 그 버전들로 각각 다른 환경을 구성해야하기 때문에 추천드립니다:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

이는 python3를 기반으로 하여 독립적인 파이썬 가상환경을 포함하여 `env`라고 하는 새로운 디렉토리를 현재 디렉토리에 만듭니다. 혹시 여러분들 시스템상에 Python3가 여러 버전이 설치가 되어 있다면, 당신은 `` `which python3` ``에 여러분들이 사용하고자 하는 파이썬 실행 위치를 바꾸어 사용하실 수 있습니다. 

여러분들은 이 가상환경을 활성화해야합니다. 여러분들이 이 환경을 매번 실행시킬 때마다 다음의 명령어를 입력해주어야 합니다.

다음, 요구되는 파이썬 패키지를 설치하기위해 pip를 사용합니다. 만약 virtualenv를 사용하지 않으신다면, `--user`옵션을 사용해야 합니다. (시스템 전체에 적용되도록 라이브러리를 설치하고 싶다면, 물론 관리자의 권한이 요구되며, 리눅스 상에서는 `pip3`대신에 `sudo pip3`를 사용한다.) 

    $ pip3 install --upgrade -r requirements.txt

훌륭합니다! 모든 준비가 끝났습니다. 여러분들은 바로 Jupyter를 시작할 수 있습니다.

## Jupyter 시작하기
만약 여러분들이 Jupyter 확장판을 사용하고 싶으시다면(추가로, 테이블 컨텐츠로 Jupyter의 효율적 사용이 가능하다), 먼저 이를 설치해 주어야합니다.

    $ jupyter contrib nbextension install --user

그리고나서 여러분들은 확장판을 활성화 할 수 있습니다. 예를들어 테이블 컨텐츠 2번 확장을 사용하고자 한다면, 

    $ jupyter nbextension enable toc2/main

좋습니다! 여러분들은 이제 주피터를 실행시킬 수 있습니다. 간단하게 다음을 입력해보세요.

    $ jupyter notebook

이는 당신의 브라우저에 창을 띄울 것입니다. 그리고 현재 디렉토리의 컨텐츠를 보여주는 Jupyter의 트리를 볼 수 있습니다. 만약 자동으로 브라우저가 열리지 않는다면, [localhost:8888](http://localhost:8888/tree)로 들어가보세요. 이제 시작하기 위해서 `index.ipynb`를 클릭해보세요!

알림 : [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions)로 들어가시면 Jupyter 확장판을 구성하고 활성화 시킬 수 있습니다. 

축하합니다! 이제 여러분들은 머신러닝을 배울 준비가 되었습니다!
