---
layout: single
title: "[AITech] 20220118 - Module and Project"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 강의 복습 내용

### Module and project

#### **Module**

* 하나의 큰 프로그램은 여러 작은 프로그램 조각들, 즉 모듈들을 모아서 개발
* 프로그램을 모듈화시키면 다른 프로그램에서 사용하기 쉬움
  * 예) 카카오톡 게임을 위한 카카오톡 로그인 모듈
* 파이썬의 모듈 == .py 파일
* namespace
  * 모듈을 호출할 때 범위를 정하는 방법

```python
# 1. Alias 설정하기 - 모듈명을 별칭을 사용
import fah_converter as fah
print(fah.convert_c_to_f(41.6))

# 2. 모듈에서 특정 함수 또는 클래스만 호출하기
from fah_converter import convert_c_to_f
print(convert_c_to_f(41.6))

# 3. 모듈에서 모든 함수 또는 클래스를 호출
from fah_converter import *
print(conver_c_to_f(41.6))
```

#### **Package**

* 하나의 대형 프로젝트를 만드는 코드의 묶음
* \_\_init\_\_, \_\_main\_\_ 등 키워드 파일명이 사용됨

1. 기능들을 세부적으로 나눠 폴더로 만듦

   ![image-20220118152008248](https://user-images.githubusercontent.com/70505378/150050515-36aae277-dfc0-40b7-aa47-58aba10f182e.png)

2. 각 폴더별로 필요한 모듈을 구현

![image-20220118152029500](https://user-images.githubusercontent.com/70505378/150050517-e2742f2a-8deb-4796-9640-4533a17f881a.png)

3. 폴더별로 `__init__.py` 구성하기

   * 현재 폴더가 패키지임을 알리는 초기화 스크립트

   * 없을 경우 패키지로 간주하지 않음(3.3+ 부터는 상관 없음)

   * 하위 폴더와 py 파일(모듈)을 모두 포함

   * import와 `__all__` keyword 사용

```python
# game.__init__.py
__all__ = ['image', 'stage'. 'sound']

from . import image
from . import stage
from . import sound
```

![image-20220118151405601](https://user-images.githubusercontent.com/70505378/150050510-b9a19bd4-6d99-44de-8fc4-4527c9a6f4fc.png)

4. `__main__.py` 파일 만들기

```python
from stage.main import game_start
from stage.sub import set_stage_level
from image.character import show_character
from sound.bgm import bgm_play

if __name__ = '__main__':
    game_start()
    set_stage_level(5)
    bgm_play(10)
    show_character()
```

✋ [참고] package namespace

```python
# 1. 절대 참조
from game.graphic.render import render_test()

# 2. 상대 참조
from .render import render_test() # .: 현재 디렉토리 기준
from ..sound.echo import echo_test() # ..: 부모 디렉토리 기준
```

5. 패키지 이름으로 호출하기 
   * `__main__.py` 실행

<br>
