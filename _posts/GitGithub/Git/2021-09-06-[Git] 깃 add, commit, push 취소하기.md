---
layout: single
title: "[Git] 깃허브 add, commit, push 취소하기"
categories: ['GitGithub','Git']
toc: true
toc_sticky: true
---

<br>

## add 취소하기

새로 추가 또는 수정한 파일은 처음에는 untracked 또는 modified 상태로 전환되죠. 

이 때 `git add 파일명` 명령어를 이용하면 파일을 stage에 올릴 수 있습니다. 

이러한 상황에서 staged된 파일을 stage에서 내리고 싶다면 `git reset HEAD` 명령어를 사용하면 됩니다. 

이는 가장 최근 커밋으로 돌아간다는 것으로, 스테이지에 올라간 파일들을 모두 내릴 수 있습니다. 

<br>

<br>

## commit 취소하기

스테이지에 올라간 파일들은 `git commit -m 커밋메시지` 명령어로 커밋할 수 있습니다. 

만약 아직 push를 진행하지 않고 커밋까지만 된 파일들이라면 `git reset 커밋코드` 명령어를 이용해 커밋을 취소할 수 있습니다. 

이 때 사용할 수 있는 옵션이 3 가지 있습니다. 

| 옵션   | 설명                                                         |
| ------ | ------------------------------------------------------------ |
| -soft  | index 보존(add한 상태, staged 상태), 워킹 디렉터리의 파일 보존. |
| -mixed | index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 보존(default 옵션 값). |
| -hard  | index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 삭제. |

옵션을 주지 않는 경우 **-mixed** 옵션이 자동으로 적용됩니다. 

<br>

### 최근 커밋으로 돌아가기

커밋을 되돌릴 때에는 `git log` 명령어를 이용해 최근 커밋로그를 출력하고, 원하는 커밋 코드를 복사하여 사용하면 되는데, 최근 1~2개 전의 커밋으로 돌아가고 싶을 때는 이 과정이 귀찮을 수 있습니다. 

이럴 때에는 간단히 아래와 같은 명령어를 사용하면 같은 효과를 낼 수 있습니다. 

```assembly
# git reset <1개전 커밋코드>
git reset HEAD^
git reset HEAD~1

# git reset <2개전 커밋코드>
git reset HEAD^^
git reset HEAD~2
```

<br>

<br>

## push 취소하기

만약 깃허브에 push까지 한 파일이라면, commit 하기 전의 내용을 원격 저장소에도 반영해주어야 합니다. 

단순히 `reset` 명령어로는 로컬 저장소에만 반영되지, 원격 저장소에는 반영되지 않기 때문입니다. 

<br>

원격 저장소에 반영하기 위해서는 **reset**된 내용을 다시 **push**해야 하는데, 이 때 일반적으로 사용하는 `git push origin main` 명령어를 사용하면 다음과 같은 에러 메시지가 출력되며 반영되지 않습니다. 

> error: failed to push some refs to 'XXX.git'
>
> hint: Updates were rejected because the tip of your current branch is behind
>
> hint: its remote counterpart. Integrate the remote changes (e.g.
>
> hint: 'git pull ...') before pushing again.
>
> hint: See the 'Note about fast-forwards' in 'git push --help' for details.

`push`가 reject되는 경우는 몇 가지가 있는데요, 위 로그에서 **hint**를 보면 **현재 브랜치가 메인 브랜치 버전보다 뒤에 있는 경우, 머지 충돌이 일어난 경우, 풀을 하지 않아 최신 버전으로 업데이트되지 않은 경우** 등이 있습니다. 

<br>

이 중 push 취소를 할 때 발생하는 에러는 첫 번째 에러인 **'Updates were rejected because the tip of your current branch is behind'** 부분입니다. 이는 깃허브 저장소의 버전이 꼬이는 것을 방지하기 위해 사전에 막아놓은 것입니다. 

하지만 지금 같은 경우, 원격 저장소에 반영된 내용을 삭제하고 싶기 때문에 이를 **강제로 push**해줍니다. 

`git push -f origin main` 명령어를 사용하면 현재 로컬의 내용을 원격 저장소에 **강제로 push**할 수 있습니다. 

<br>

### git revert

그런데 위 방법같은 경우, 여러 명이 협업을 진행하고 있다면 다른 사람들의 버전 관리에서 문제가 생길 수 있습니다. 

이럴 때에는 **commit을 취소할 때 `reset`이 아닌 `revert` 명령어를 사용**하여 커밋 버전을 되돌렸다는 표시를 남기는 커밋을 제출합니다. 

예를 들면 아래와 같습니다. 

```assembly
git revert 커밋코드
git commit -m 커밋 메시지
git push origin main
```

위와 같이 하면, 이전 커밋을 완전히 취소하는 것이 아니라 **커밋을 되돌렸다는 커밋**을 남길 수 있게 되는 것입니다. 

<br>

<br>

이상으로 add, commit, push를 되돌릴 수 있는 방법에 대해 알아보았습니다. 

git과 github는 편리한 툴임은 분명하지만, 그만큼 복잡한 상황이 종종 생기는 것 같습니다 😂

아무튼 제 포스팅이 여러분에게 도움이 되었기를 바라며, 포스팅 마치겠습니다!