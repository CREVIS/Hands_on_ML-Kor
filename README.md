Machine Learning Notebooks
==========================

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

혹시 여러분들이 Python에 익숙하고, Python 라이브러리 설치방법을 알고 계신다면, `requirements.txt`에 정리해놓은 라이브러리들을 설치해주시고 [Jupyter 시작하기](#starting-jupyter) 섹션으로 넘어가셔도 좋습니다. 자세한 가이드가 필요하시다면 계속 읽어나가시면 됩니다.

## 파이썬과 필요한 라이브러리들
당연하지만, 여러분들 분명히 파이썬이 필요할겁니다. 요새 대부분의 시스템들에는 Python2가 기본적으로 설치되어있습니다. 어떤 시스템들은 Python3도 미리 설치가 되어있습니다. 다음 명령어를 터미널에 입력하시면 여러분들의 시스템의 Python 버전을 확인하실 수 있습니다.



Python3는 ≥3.5.이라면 어느 버전이든 상관없습니다. 만약 Python3가 없다면, 설치하시는 것을 권장드립니다. (Python 2.6이상에서도 작동하지만, 권장드리지는 않기에, Python3가 바람직합니다.) 이렇게 하시고나면, 여러분들은 몇가지의 선택사항이 있습니다. 윈도우나 Mac OS X 상에서는 [python.org](https://www.python.org/downloads/)에 가서 다운로드만 하시면 됩니다. Mac OS X에서는 [MacPorts](https://www.macports.org/)나 [Homebrew](https://brew.sh/)를 대신 사용할 수 있습니다. 리눅스 상에서는 어떻게 해야할 지 알고있는 것이 아니라면, 여러분들 시스템에서 제공하는 패키징 시스템을 사용해야만합니다. 예를들어 Ubuntu나 Debian상에서는 다음과 같이 입력하셔야 합니다.

    $ sudo apt-get update
    $ sudo apt-get install python3

또 다른 선택 사항으로는 [Anaconda](https://www.continuum.io/downloads)를 설치하는 것입니다. 이는 Python 2, 3버전 모두와, 많은 체계적인 라이브러리들을 포함한 패키지입니다. 여러분들은 Python3버전을 사용하시는 것을 권장드립니다.

만약 Anaconda를 사용하실 것이라면,ㅜ 다음 섹션을 읽어주시고, 아니라면 [pip 사용하기](#using-pip) 섹션으로 넘어가주세요

## Anaconda 사용하기
Anaconda를 사용할 때, 여러분들은 이 프로젝트에 맞는 독단적인 Python 환경을 추가적으로 구성해야합니다. 이는 각각의 프로젝트에 대하여 매우 다른 라이브러리들과 그 버전들로 각각 다른 환경을 구성해야하기 때문에 추천드립니다.

    $ conda create -n mlbook python=3.5 anaconda
    $ source activate mlbook

이는 `mlbook`이라는 Python 3.5 환경을 새롭게 만들어 준 것이며(원하신다면 여러분들이 원하시는 이름들로 바꾸어 사용하실 수 있습니다), 이 환경을 활성화 한 것입니다. 이 환경은 Anaconda에 달려있는 모든 체계적인 라이브러리들을 포함하고 있습니다. 우리가 필요로하는 (NumPy, Matplotlib, Pandas, Jupyter 그리고 몇가지 다른 것들) 모든 라이브러리들을 포함하고 있습니다. 하지만 TensorFlow는 깔려있지않습니다. 그러므로 다음의 명령어를 실행해줍니다.

    $ conda install -n mlbook -c conda-forge tensorflow=1.0.0

이 명령어는 `mlbook` 가상환경 안에 TensorFlow 1.0.0을 설치합니다 (`conda-forge`라는 레포지토리에서 가져옵니다). `mlbook` 환경을 설치하고 싶지 않으시다면, `-n mlbook` 옵션을 제거하고 사용하시면 됩니다.

그 다음으로, 여러분들은 Jupyter 확장판을 설치할 수 있습니다. 

    $ conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions

모든 준비가 끝났습니다! 다음으로, [Jupyter 시작하기](#starting-jupyter)로 넘어가주세요.
