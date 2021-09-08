---
layout: single
title: "[Github] 깃허브 페이지에 카테고리 만들기"
---



<br>

## <span style="color:rgb(124, 7, 160)">카테고리 만들기</span>

---

<br>

#### 1. posts 폴더에 있는 파일에 카테고리 변수 선언

포스팅 파일의 YAML Front Matter에 **categories** 변수를 선언합니다. 

```yaml
---
layout: single
comments: ...
title: ...
subtitle: ...
description: ...
date: ...
background: ...
...
categories: ['programming', 'python']
---
```

다른 파일들에도 모두 추가합니다. 

```yaml
---
...
categories: ['programming', 'java'] 
---
```

```yaml
---
...
categories: ['programming', 'ruby'] 
---
```

<br>

<br>

#### 2. 카테고리 변수 가져오기

우리는 카테고리를 블로그의 홈(초기화면)에 띄울 것이다. 

위치는 사용자 임의로 지정할 수 있는데, 여기서는 화면의 왼쪽 수직 중앙에 배치해보도록 한다. 

<br>

layout 폴더에서 홈 화면을 나타내는 home.html에 코드를 작성한다. 

가장 바깥 위치의 태그로 < div > 를 추가해준다.  그리고 그 안에 카테고리를 생성한다. 

```html
---
layout: archive
---

{{ content }}
...

<div class="col-lg-4 col-md-2">
  ...
</div>

{% include paginator.html %}
```

<br>

여기서 알아야할 것이 있는데, 바로 지킬 블로그에서 사용하는 리퀴드 문법이다. 

여기서는 카테고리를 만드는데 꼭 필요한 부분만 알아보도록 하자. 

<br>

**카테고리를 어떻게 불러오느냐?**

카테고리는 지킬의 `site.categories`라는 변수에 리스트들이 들어있다. 

우리는 이 리스트들을 순회하면서 필요한 리스트들을 출력한다. 

```javascript
<div class="col-lg-4 col-md-2">
  {% for category in site.categories %} 

  {% endfor %}
</div>
```

site.categories에 있는 리스트를 차례로 category에 담으면서 순회를 진행합니다. 

<br>

category라는 변수를 출력해봅시다. 































