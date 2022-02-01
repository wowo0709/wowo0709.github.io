---
layout: single
title: "[NumericalAnalysis] Matlab - 1"
categories: ['Mathematics', 'NumericalAnalysis', 'Matlab']
toc: true
toc_sticky: true
tag: ['Matlab']
---

<br>

## 1. MATLAB Environment

Matlab 시작 시 `>>` 커맨드가 보이고, 이 커맨드에 line by line으로 명령을 실행할 수 있다. 

변수 할당 없이 수행된 계산은 `ans` 변수에 할당된다. 

`.m` 파일에서 `keyboard` 명령으로 breakpoint를 설정할 수 있으며, **[Step]** 을 클릭하여 line by line으로 실행할 수 있다. 

<br>

## 2. Assignment

### Scalars

```matlab
% Echo printing(Result prompt on command line)
a = 4
% Supress Echo printing -> Semicolon(:)
A = 6;
% Complex value
x = 2 + j*4
% predefined variables
pi
```

<br>

Matlab은 기본적으로 소수점 4자리까지 출력한다. 이를 다음 명령어로 늘이고 줄일 수 있다. 

```matlab
pi
>> 3.1416
format long
>> 3.141592653589793
format short
>> 3.1416
```

<br>

### Arrays, Vectors and Matrices

```matlab
% row vector
a = [1 2 3 4 5]
% column vector
b = [2; 4; 6; 8; 10;]
% 행벡터와 열벡터는 전치 관계
a = b'
% 3x3 matrix
A = [1 2 3; 4 5 6; 7 8 9]
A = [[1 4 7]' [2 5 8]' [3 6 9]']
% 현재 생성된 변수 확인
who
% 더 디테일하게
whos
% indexing -> first index is 1
b(4)
>> 8
A(2,3)
>> 6
% built-in functions
E = zeros(2,3)
u = ones(1,3)

```

<br>

### Colon Operator

```matlab
% range operator
t = 1:5
>> [1 2 3 4 5]
t = 1:0.5:3
>> [1.0 1.5 2.0 2.5 3.0]
t = 10:-1:5
>> [10 9 8 7 6 5]
% Selective select
A(2, :)
>> [4 5 6]
t(2:4)
>> [9 8 7]
```

<br>

### linspace, logspace

```matlab
% generate linear space
% n의 default 값은 100
linspace(0,1,6)  
>> [0 0.2 0.4 0.6 0.8 1.0]
% generate log space
% n의 default 값은 50
logspace(-1,2,4)
>> [0.1 1 10 100]
```

<br>

### Character Strings

```matlab
f = 'Miles'
s = 'Davis'
% concatenate strings
x = [f s]
>> 'Miles Davis'
% next line
a = [1 2 3 4 5 ...
		6 7 8]
>> [1 2 3 4 5 6 7 8]
% for string, 
quote = ['Any fool can make a rule, ' ...
				'and any fool will mind it']
```

<br>

<br>

## 3. Mathematical Operations

### Basic Operations

```matlab
^ % exponential
- % Negation
* / % Multiplication and Division
\ % left division
+ - % Addition and subtraction
```

<br>

### Operation with Vector, Matrix

```matlab
% Inner product
a = [1 2 3 4 5]
b = [2 4 6 8 10]'
a*b
>> 110
% Outer product
b*a
% Vector - Matrix multiplication
a = [1 2 3]
b = [4 5 6]'
a*A
A*b
% Matrix multiplication
A*A
A^2
% elementwise operation(.)
A.^2
```

<br>

<br>

## 4. Use of Build-in functions

```matlab
% description of function
help log
% log function
% applicable for scalar, vector, matrix, anything (operates by elementwise)
log(A)
% matrix operations
sqrt(A) % A의 각 원소 a에 대해 root(a) 적용
sqrtm(A) % X*X = A가 되는 X 를 계산
% rounding
E = [-1.6 -1.5 -1.4 1.4 1.5 1.6];
round(E)
>> [-2 -2 -1 1 2 2]
ceil(E)
>> [-1 -1 -1 2 2 2]
floor(E)
>> [-2 -2 -2 1 1 1]
% special function for matrics and arrays
F = [3 5 4 6 1]
sum(F)
19
min(F)
1
max(F)
6
mean(F)
3.8
prod(F)
380
sort(F)
[1 3 4 5 6]
```

```matlab
% evaluation of formula
t = [0:2:20]'
length(t)
>> 11
g = 9.81;m = 68.1; c = 0.25;
v = sqrt(g*m/c)*tanh(sqrt(g*c/m)*t)
>>
0
18.7292
...
51.6416
```

<br>

<br>

## 5. Graphics

```matlab
% Plotting
plot(t,v)
% put labels
title()
xlabel()
ylabel()
legend()
grid
% customize
plot(t, v, 'o')
plot(t, v, 's--g')
% hold
hold on
hold off
% subplot
subplot(1,2,1)
subplot(1,2,2)
```

<br>

<br>


