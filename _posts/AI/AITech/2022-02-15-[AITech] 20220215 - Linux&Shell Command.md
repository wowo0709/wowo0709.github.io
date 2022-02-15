---

layout: single
title: "[AITech][Product Serving] 20220215 - Linux&Shell Command"
categories: ['AI', 'AITech', 'MLOps']
toc: true
toc_sticky: true
tag: []
---



<br>

**_본 포스팅은 SOCAR의 '변성윤' 강사 님의 강의를 바탕으로 제작되었습니다._** 

# 학습 내용

## Linux

개발을 본격적으로 하려면 `Linux`를 사용할 일이 많다고 하는데... 그 이유는 무엇일까요?

* 서버에서 자주 사용하는 OS
* Free, Open Source
  * 여러 버전이 존재하며, 직접 본인만의 버전을 만들 수도 있음
* 안정성, 신뢰성이 높음
* 쉘 커맨드, 쉘 스크립트

![image-20220215145657098](https://user-images.githubusercontent.com/70505378/154063179-7c810637-ed59-464c-a038-2480b0221acc.png)

Linux와 Shell Script를 잘 학습하는 방법은 최초엔 **자주 사용하는** 쉘 커맨드, 쉘 스크립트 위주로 학습하고 필요한 코드가 있는 경우 그때마다 검색해서 찾는 것입니다. 이 때 해당 커맨드를 이해(왜 이렇게 되는가?)하고 정리하는 것이 좋습니다. 

다른 OS에서 Linux를 사용하는 방법으로는 다음의 것들이 있습니다. 

* \- VirtualBox에 Linux 설치, Docker로 설치 
* WSL 사용(윈도우) 
* Notebook에서 터미널 실행 
* Cloud에서 사용



<br>

## Shell Command

### 쉘의 종류

* **쉘**: 사용자가 문자를 입력해 컴퓨터에 명령할 수 있도록 하는 프로그램
* **터미널/콘솔**: 쉘을 실행하기 위해 문자 입력을 받아 컴퓨터에 전달. 프로그램의 출력을 화면에 작성
* **sh**: 최초의 쉘
* **bash**: Linux 표준 쉘
* **zsh**: Mac 카날리나 OS 기본 쉘

![image-20220215150314398](https://user-images.githubusercontent.com/70505378/154063182-053f1358-6276-414e-ad74-13e2ce381edb.png)



### 쉘을 사용하는 이유

* 서버에서 접속해서 사용하는 경우 
* crontab 등 Linux의 내장 기능을 활용하는 경우 
* 데이터 전처리를 하기 위해 쉘 커맨드를 사용 
* Docker를 사용하는 경우 
* 수백대의 서버를 관리할 경우 
* Jupyter Notebook의 Cell에서 앞에 !를 붙이면 쉘 커맨드가 사용됨 
* 터미널에서 python3, jupyter notebook 도 쉘 커맨드 
* Test Code 실행 
* 배포 파이프라인 실행(Github Action 등에서 실행)







### 기본 쉘 커맨드

* `man`: 쉘 커맨드의 매뉴얼 문서를 볼 수 있음 (MANual)
  * ex) man mkdir
  * 종료: ':q' 입력
* `mkdir`: 폴더 생성하기 (MaKe DIRectory)
  * ex) mkdir linux-test
* `ls`: 현재 접근한 위치의 폴더, 파일 목록 확인 (List Segments)
  * ls 뒤에 폴더를 작성하면 해당 폴더 기준에서 실행
  * 옵션
    * -a: .으로 시작하는 파일, 폴더를 포함해 전체 파일 출력
    * -l: 퍼미션, 소유자, 만든 날짜, 용량까지 출력
    * -h: 용량을 사람이 읽기 쉽도록 GB, MB 등 표현. '-l'과 같이 사용. 
  * ex) ls -lh
* `pwd`: 현재 폴더 경로를 절대 경로로 보여줌(Print Working Directory)
  * ex) pwd
* `cd`: 현재 위치 이동하기 (Change Directory)
  * ex) cd linux-test
* `echo`: Python의 print처럼 터미널에 텍스트 출력
  * ex) echo "hi" => hi 출력 / echo 'pwd' => 현재 위치를 절대 경로로 출력
* `vi`: vim 편집기로 파일 생성
  * Command Mode: vi 실행 시 기본 모드
    * 방향키를 통해 커서를 이동
    * dd : 현재 위치한 한 줄 삭제
    * i : INSERT 모드로 변경
    * x : 커서가 위치한 곳의 글자 1개 삭제(5x : 문자 5개 삭제) 
    * yy : 현재 줄을 복사(1줄을 ctrl + c) 
    * p : 현재 커서가 있는 줄 바로 아래에 붙여넣기 
    * k : 커서 위로 
    * j : 커서 아래로 
    * l : 커서 오른쪽으로 
    * h : 커서 왼쪽으로
  * Insert Mode: 파일을 수정할 수 있는 모드
    * Command mode로 다시 이동하고 싶다면 ESC 입력
  * Last Line Mode: ESC를 누른 후 콜론(:)을 누르면 나오는 모드
    * w : 현재 파일명으로 저장 
    * q : vi 종료(저장되지 않음) 
    * q! : vi 강제 종료(!는 강제를 의미) 
    * wq : 저장한 후 종료 
    * /문자 : 문자 탐색 - 탐색한 후 n을 누르면 계속 탐색 실행
    * set nu : vi 라인 번호 출력
* `bash`: bash로 쉘 스크립트 실행
  * ex) bash vi-test.sh => 파일에 작성된 내용 출력
* `sudo`: 관리자 권한으로 실행 (SUper DO, Substitute User DO)
* `cp`: 파일 또는 폴더 복사 (CoPy)
  * ex) cp vi-test.sh vi-test2.sh
  * 옵션
    * -r : 디렉토리를 복사할 때 디렉토리 안에 파일이 있으면 recursive(재귀적)으로 모두 복사 
    * -f : 복사할 때 강제로 실행
* `mv`: 파일, 폴더 이동하기 (또는 이름 바꿀 때도 사용) (MoVe)
  * ex) mv vi-test.sh vi-test3.sh
* `rm`: 파일, 디렉토리 삭제하기 (ReMove)
  * 리눅스에서는 파일 복구가 안되거나 매우 어렵기 때문에 아주 조심히 사용해야 함
  * ex) mv vi-test3.sh
  * 옵션
    * -r: 디렉토리를 삭제할 경우 사용 (recursive)
    * -f: 강제 삭제
    * -i: 디렉토리에 들어있는 내용을 하나하나 확인하면서 삭제(-r과 같이 사용)
* `cat`: 특정 파일 내용 출력 (conCATenate)
  * ex) cat vi-test.sh
  * 여러 파일을 인자로 주면 합쳐서 출력
    * ex) cat vi-test2.sh vi-test3.sh
  * 파일에 저장(OVERWRITE)하고 싶은 경우
    * ex) cat vi-test2.sh vi-test3.sh > new_test.sh
  * 파일에 추가(APPEND)하고 싶은 경우
    * ex) cat vi-test2.sh vi-test3.sh >> new_test.sh
* `clear`: 터미널 창을 깨끗하게 해 줌
* `history`: 최근에 입력한 쉘 커맨드 History 출력
  * History 결과에서 느낌표를 붙이고 숫자 입력시 그 커맨드를 다시 활용할 수 있음
    * ex) !30
* `find`: 파일 및 디렉토리 검색
  * ex) find . -name "File": 현재 폴더에서 File이란 이름을 가지는 파일 및 디렉토리 검색
* `export`: export로 환경변수 설정
  * ex) export water="물" => echo $water => '물' 출력
    * '=' 양 옆에는 공백이 없어야 함
  * export로 환경 변수 설정한 경우, 터미널이 꺼지면 사라지게 된다. 따라서 매번 쉘을 실행할 때마다 환경변수를 저장하고 싶으면 .bashrc, .zshrc에 저장하면 된다. 
    * (Linux) vi ~/.bashrc 또는 vi ~/.zshrc (자신이 사용하는 쉘에 따라 다름)
    * 제일 하단에 export water=”물"을 저장하고 나옴(ESC :wq) 
    * 그 후 source ~/.bashrc 또는 source ~/.zshrc Linux 환경 설정을 재로그인하지 않고 즉시 적용하고 싶은 경우 source 사용
* `alias`: 쉘 커맨드를 별칭으로 설정
  * ex) alias ll2='ls -l' => ll2를 입력하면 ls -1이 동작
  * '=' 양 옆에는 공백이 없어야 함







### Redirection & Pipe

**쉘 커맨드**

Redirection & Pipe에서 자주 사용되는 쉘 커맨드들을 먼저 보자. 

* `head`, `tail`: 파일의 앞/뒤 n행 출력

  * ex) head -n 3 vi-test.sh

* `sort`: 행 단위 정렬

  * 옵션
    * -r: 내림차순 정렬
    * -n: Numeric sort
  * ex) cat fruits.txt | sort -r

* `uniq`: 중복된 행이 연속으로 있는 경우 중복 제거. sort와 함께 사용.

  * 옵션
    * -c: 중복 행의 개수 출력
  * ex) cat fruits.txt | sort | uniq / cat fruits.txt | sort | uniq | wc -l
    * 'wc -l' 은 행 단위로 word count

* `grep`: 파일에 주어진 패턴 목록과 매칭되는 라인 검색

  * ex) grep 옵션 패턴 파일명
  * 옵션
    * -i : Insensitively하게, 대소문자 구분 없이 찾기 
    * -w : 정확히 그 단어만 찾기
    * -v: 특정 패턴 제외한 결과 출력 
    * -E : 정규 표현식 사용
  * 정규 표현식 패턴
    * ^단어 : 단어로 시작하는 것 찾기 
    * 단어$ : 단어로 끝나는 것 찾기 
    * . : 하나의 문자 매칭

* `cut`: 파일에서 특정 필드 추출

  * 옵션

    * -f: 잘라낼 필드 지정
    * -d: 필드를 구분하는 구분자. (Default \\t)

  * ex)

    * cat cut_file

      ```bash
      hello:my:name:is:ryan
      ans:i:like:kimchi
      and:you:?
      ```

    * cat cut_file | cut -d : -f 1,3

      ```bash
      hello:name
      ans:like
      and:?
      ```



**표준 스트림(Stream)**

Unix에서 동작하는 프로그램은 커맨드 실행 시 3개의 stream이 생성

* **stdin** : 0으로 표현, 입력(비밀번호, 커맨드 등) 
* **stdout** : 1로 표현, 출력 값(터미널에 나오는 값) 
* **stderr** : 2로 표현, 디버깅 정보나 에러 출력

![image-20220215154114862](https://user-images.githubusercontent.com/70505378/154063184-24ebddc8-5646-4a0e-858a-76aaa2830bfb.png)



**Redirection & Pipe**

* **Redirection**: 프로그램의 출력(stdout)을 다른 파일이나 스트림으로 전달
  * `>`: 덮어쓰기(Overwrite). 파일이 없으면 생성하고 저장. 
    * ex) echo "hello" > vi-test.sh
  * `>>`: 맨 아래에 추가하기(Append)
    * ex) echo "hi" >> vi-test.sh 
* **Pipe**: 프로그램의 출력(stdout)을 다른 프로그램의 입력으로 사용하고 싶은 경우
  * `A | B`: A의 Output을 B의 Input으로 사용(다양한 커맨드를 조합)
  * ex) ls | grep "vi" -> 현재 폴더에 있는 파일명 중 vi가 들어간 단어를 찾아줌
* Examples
  * ls | grep "vi" >> output.txt -> ls | grep "vi"의 결과를 output.txt 파일에 추가(append)
  * history | grep "echo" -> 최근 입력한 커맨드 중 echo가 들어간 명령어를 찾아줌











### 서버에서 자주 사용하는 쉘 커맨드

* `ps`: 현재 실행되고 있는 프로세스 출력하기 (Process Status)

  * 옵션
    * `-e`: 모든 프로세스
    * `-f`: Full Format으로 자세히 보여줌

* `curl`: Command Line 기반의 Data Transfer 커맨드 (Client URL)

  * ex) curl -X localhost:5000/ {data} -> 웹 서버를 작성한 후 요청이 제대로 실행되는지 확인할 수 있음
  * curl 외에 httpie 등도 있음(더 가독성 있게 출력)

* `df`: 현재 사용 중인 디스크 용량 확인(Disk Free)

  * 옵션
    * -h: 사람이 읽기 쉬운 형태로 출력

* `scp`: SSH을 이용해 네트워크로 연결된 호스트 간 파일을 주고받는 명령어(Secure CoPy)

  * 옵션
    * -r: 재귀적으로 복사
    * -P: ssh 포트 지정
    * -i: SSH 설정을 활용해 실행
  * ex) scp local_path user@ip:remote_directory -> local에서 remote로 전송
  * ex) scp user@ip:remote_directory local_path -> remote에서 local로 전송
  * ex) scp user1@ip1:source_remote_directory user2@ip2:target_remote_directory -> remote에서 remote로 전송

* `nohup`: 터미널 종료 후에도 계속 작업이 유지되도록 실행(백그라운드 실행)

  * ex) nohup python3 app.py &
  * nohup으로 실행될 파일을 Permission이 755여야 함
  * screen이라는 커맨드도 있음

* `chmod`: 파일의 권한을 변경(CHange MODe)

  * 유닉스에서 파일이나 디렉토리의 시스템 모드를 변경

  * Permission

    ![image-20220215212937802](https://user-images.githubusercontent.com/70505378/154063189-9f79e7f6-8a1a-4927-a93a-511e98104da2.png)

    * r: Read(읽기), 4
    * w: Write(쓰기), 2
    * x: eXecute(실행하기), 1
    * -: Denied
    * ex) 755로 퍼미션을 주세요!
      * 7 = 4 + 2 + 1 = rwx
      * 5 = 4 + 1 = r-x
      * 5 = 4 + 1 = r-x
      * 755 = rwxr-xr-x

  * ex) chmod 755 vi-test3.sh -> vi-test.sh 파일의 권한(permission)을 755(rwxr-xr-x)로 변경

  * 첫번째 3자리는 사용자, 가운데 3자리는 그룹 내 인원, 마지막 3자리는 제3자에 해당하는 권한











### 쉘 스크립트

.sh 파일을 생성하고 그 안에 쉘 커맨드를 작성하여 스크립트로 생성할 수 있습니다. 즉, 쉘 스크립트는 쉘 커맨드의 조합이며 쉘 스크립트는 하나의 프로그램으로서 실행할 수 있습니다. `bash 파일명.sh` 커맨드로 실행할 수 있습니다. 

![image-20220215213350530](https://user-images.githubusercontent.com/70505378/154063194-237dfbfa-9a2d-4ec7-a449-64101b099c05.png)





















<br>

<br>

# 참고 자료

* 







<br>
