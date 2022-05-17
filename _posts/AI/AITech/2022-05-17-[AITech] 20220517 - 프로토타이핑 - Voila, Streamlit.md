---
layout: single
title: "[AITech][Product Serving] 20220517 - 프로토타이핑 - Voila, Streamlit"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# 프로토타이핑 - Voila, Streamlit

## Voila

**Voila 란?**

![image-20220517132837185](https://user-images.githubusercontent.com/70505378/168735546-0dc13f7a-87b1-45d4-afce-6b5aef38ba77.png)

**Voila**는 Jupyter Notebook 베이스 프로토타이핑 라이브러리입니다. 

각자 notebook 환경에서 작업을 할 때는 서로의 환경이 다릅니다. 이를 공유하는 방법도 있겠지만, Voila를 사용하여 웹에 간단하게 프로토타이핑하여 결과를 시각화할 수 있습니다. 

이는 특히나 개발을 잘 모르는 분들에게 테스트 요청을 할 경우에 유용합니다. 

* [Voila github](https://github.com/voila-dashboards/voila)

[Voila를 사용한 예시]

![image-20220517133210344](https://user-images.githubusercontent.com/70505378/168735548-1a7b3fed-6409-443b-b387-8cdf6163ec2f.png)

<br>

Voila의 본래 목적인 '대시보드'이며 R의 Shiny, 파이썬의 Dash와 유사한 도구입니다. 

대시보드에는 다양한 도구들이 존재하는데(Superset, Metabase, Redash, Tableau), 이러한 도구들은 모두 서버에 제품을 설치한 후 연동(SQL 베이스)시켜야 한다는 불편함이 있습니다. 

[Voila의 장점]

* Jupyter Notebook 결과를 쉽게 웹 형태로 띄울 수 있음
* Ipywidget, Ipyleaflet 등을 사용하여 인터랙티브하게 사용 가능
* Jupyter Notebook의 Extension 있음(=노트북에서 바로 대시보드로 변환 가능)
* Python, Julia, C++ 코드 지원
* 고유한 텐플릿 생성 가능
* 너무 쉬운 러닝커브

<br>

**Voila 사용법**

```python
pip3 install voila

# JupyterLab 사용한다면
jupyter labextension install @jupyter-voila/jupyterlab-preview

# jupyter Notebook이나 Jupyter Server를 사용한다면
jupyter serverextension enable voila --sys-prefix

# nbextension도 사용 가능하도록 하고 싶다면 다음과 같이 설정
voila --enable_nbextensions=True
jupyter notebook --VoilaConfiguration.enable_nbextensions=True
```

Jupyter Lab을 실행시키고 좌측 아래 버튼을 확인하면 Enable이 보임

![image-20220517134607872](https://user-images.githubusercontent.com/70505378/168735552-c8df9eda-7c84-469a-b635-f8301b97524c.png)

Voila 아이콘이 보임

![image-20220517134623406](https://user-images.githubusercontent.com/70505378/168735554-c5fda1ad-9cc9-4704-ac58-ce31f83e7ff1.png)

CLI에서 사용하는 경우 `voila`를 입력하면 `localhost:8866`에서 확인 가능

![image-20220517134727947](https://user-images.githubusercontent.com/70505378/168735555-7b8c2ef9-f189-4c6f-8ca7-88bb8c4bebb4.png)

<br>

[Voila 사용시 TIP]

voila에서 idle 상태인 경우 연결을 끄는 행위를 **cull**이라고 합니다. 이러한 cull 관련 config는 아래에서 확인할 수 있습니다. 

* [Config file and command line options — Jupyter Notebook 6.4.11 documentation (jupyter-notebook.readthedocs.io)](https://jupyter-notebook.readthedocs.io/en/stable/config.html?highlight=cull_idle_timeout)

그리고 아래와 같이 cull 관련 세팅들을 커스텀 할 수 있습니다. 

* `cull_interval`: idle 커널을 확인할 간격(초)  
* `cull_idle_timeout`: 커널을 idle 상태로 판단할 기준(초). 이 시간동안 이벤트가 없으면 idle로 판단  

```python
voila voila_basic.ipynb 
--MappingKernelManager.cull_interval=60 
--MappingKernelManager.cull_idle_timeout=300
```

셀 타임아웃 제한을 할 수도 있습니다. 타임아웃이란 cell이 몇 초(default 30초) 이상 진행되면 Timeout Error를 발생시키는 것을 말합니다. 

```python
voila --ExecutePreprocessor.timeout=180

# Jupyter notebook 실행 시
jupyter notebook --ExecutePreprocessor.timeout=180
```

또는 nbextension을 사용할 수도 있습니다. 

```python
voila your_notebook.ipynb --enable_nbextensions=True

# Jupyter notebook 실행 시
jupyter notebook 
--ExecutePreprocessor.timeout=180 
--VoilaConfiguration.enable_nbextensions=True
```

마지막으로 password를 설정할 수도 있습니다. 

1. Jupyter Notebook의 설정 파일 생성하기 (있다면 skip)

![image-20220517135959175](https://user-images.githubusercontent.com/70505378/168735559-f6966aed-a31c-4bf1-b11c-9a7856f54f81.png)

2. 터미널에서 python 실행 후 아래 코드 실행

![image-20220517140026058](https://user-images.githubusercontent.com/70505378/168735561-fc299d04-7325-4afd-a705-4f77282640a2.png)

3. 아까 생성된 jupyter notebook config(`vi ~/.jupyter/jupyter_notebook_config.py  `)로 진입 후     **c.NotebookApp.password**를 찾아서 우측에 복사한 sha1 값을 붙여넣기

![image-20220517140225304](https://user-images.githubusercontent.com/70505378/168735564-f5c05233-e222-43c6-8b0d-256ac5f8199c.png)



<br>

## Streamlit

![image-20220517141048660](https://user-images.githubusercontent.com/70505378/168735573-956abd10-52d9-4353-8186-cfe2b7994e9f.png)

Voila의 경우 노트북에서 쉽게 프로토타입을 만들 수 있지만, 대스보드처럼 레이아웃을 잡기 어렵습니다. 이런 경우 웹 개발을 진행할 수 있습니다. 

웹 개발 시에도 처음부터 HTML/CSS + Flask/Fast API를 사용하여 구축할 수 있지만, 간단하게 빠른 프로토타이핑을 위해 **Streamlit**을 사용할 수 있습니다. 

* [https://streamlit.io/](https://streamlit.io/)
* [https://docs.streamlit.io/](https://docs.streamlit.io/)

아래는 일반적인 웹 개발로 프로토타이핑 했을 때와 Streamlit을 사용했을 때 workflow를 비교한 모습입니다. 

![image-20220517140708659](https://user-images.githubusercontent.com/70505378/168735571-71a1848d-b697-459f-bdfb-1d1f990cdf75.png)

<br>

아래는 Voila, Streamlit을 포함해 다른 프로토타입 도구들을 비교한 표입니다. 

![image-20220517140950458](https://user-images.githubusercontent.com/70505378/168735572-151c7ed4-9231-4567-8428-c57f22abcf6f.png)

[Streamlit의 장점]

* 파이썬 스크립트 코드를 조금만 수정하면 웹을 띄울 수 있음
* 백엔드 개발이나 HTTP 요청을 구현하지 않아도 됨
* 다양한 Component 제공해 대시보드 UI 구성할 수 있음
* Streamlit Cloud도 존재해서 쉽게 배포할 수 있음(단, Community Plan은 Public Repo만 가능)
* 화면 녹화 기능(Record) 존재  

albumentations의 데모도 Streamlit으로 만들어졌습니다. 

![image-20220517141223357](https://user-images.githubusercontent.com/70505378/168735575-2aeb715d-37ed-4249-9eae-bc6e071d5146.png)

<br>

**Streamlit 사용하기**

Streamlit 설치

```python
pip3 install streamlit
```

Streamlit 실행

```python
# CLI
streamlit run streamlit-basic.py
```

<br>

Streamlit은 text(header), button, check box, dataframe, markdown, table, metric, json, line chart, map, select box, slider, input box, caption, code, latex, sidebar, columns, expander, spinner, balloons, status box, form, file uploader 등 매우 다양한 기능들을 제공합니다. 

자세한 내용은 공식 문서에서 참고하고, 여기서는 중요한 **session state**와 **cache**에 대해 간단하게 알아봅니다. 

[Session State]

Streamlit은 화면에서 무언가 업데이트되면 **전체 streamlit 코드가 다시 실행**됩니다. 

1. Code가 수정되는 경우
2. 사용자가 streamlit의 위젯과 상호작용하는 경우 (버튼 클릭, 입력 상자에 텍스트 입력 시 등)

이에 변수가 갱신이 되기 때문에, 코드가 다시 실행되어도 갱신되지 않도록 하기 위해 session_state가 존재합니다. 

왼쪽 코드는 button을 누를 때마다 코드가 재실행되기 때문에 count_value 값이 계속 0이 되는데, 오른쪽 코드는 session_state에 count_value를 추가해주어 값이 제대로 변경됩니다. 

![image-20220517142325236](https://user-images.githubusercontent.com/70505378/168735576-67954fc7-cb6b-476c-8eb6-70445e63e139.png)

참고 자료

* Session State에 대해 이해가 되지 않는다면, 실제로 코드를 실행해보고 아래 문서 읽어보기!
  * [https://blog.streamlit.io/session-state-for-streamlit/](https://blog.streamlit.io/session-state-for-streamlit/)
* Session State를 사용한 예시
  * [https://share.streamlit.io/streamlit/release-demos/0.84/0.84/streamlit_app.py](https://share.streamlit.io/streamlit/release-demos/0.84/0.84/streamlit_app.py)

[@st.cache]

매번 다시 실행하는 특성 때문에 데이터도 매번 다시 읽게 될 수 있습니다. 

이런 경우 `@st.cache` 데코레이터를 사용해 캐싱하면 좋습니다. 

![image-20220517142555996](https://user-images.githubusercontent.com/70505378/168735579-9655fec9-e7aa-462b-abcb-36d5e7c0cc64.png)



<br>

<br>

# 참고 자료

* 
