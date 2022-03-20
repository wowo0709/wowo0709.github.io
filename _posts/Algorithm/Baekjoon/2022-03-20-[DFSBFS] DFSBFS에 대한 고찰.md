---
layout: single
title: "[DFSBFS] '그래프 탐색'에 대한 고찰"
categories: ['Graph', 'DFSBFS', 'ShortestPath']
toc: true
toc_sticky: true
tag: []
---

_'그래프 탐색'에 대해 새로운 부분을 발견할 때마다 업데이트합니다._

# '그래프 탐색'에 대한 고찰

그래프 탐색은 다음과 같이 나눌 수 있다. 

* 단순 탐색
  * dfs: 모든 경우 탐색, 백트래킹
    * 백트래킹 코드에 조건을 추가하면 T자 탐색이 가능하다. 
      * https://wowo0709.github.io/implementation/bruteforce/backtracking/graph/dfsbfs/Baekjoon-14500.-%ED%85%8C%ED%8A%B8%EB%A1%9C%EB%AF%B8%EB%85%B8/ 
    * 백트래킹 알고리즘의 핵심은 '유망성 검사(가지치기, pruning)'이다. 적절한 유망성 검사는 시간을 크게 단축시킨다. 
  * bfs: 최단거리, 연결 요소 탐색
    * 정해진 목적지 없이 목적지 후보들만 있을 때, 목적지 후보들 중 최단 거리의 목적지를 찾는 용도로 사용할 수 있다. 
      * https://wowo0709.github.io/implementation/dfsbfs/Baekjoon-16236.-%EC%95%84%EA%B8%B0%EC%83%81%EC%96%B4/
  * floyd-warshall: 모든 노드 간 최단 거리
  
* 최적 탐색
  * dijkstra: 최단 거리 + 가중치 존재
  * bellman-ford: 최단 거리 + 음수 가중치 존재



<br>













