---
layout: single
title: "[NumericalAnalysis] Matlab - 2"
categories: ['Mathematics', 'NumericalAnalysis', 'Matlab']
toc: true
toc_sticky: true
tag: ['Matlab']
---

<br>

## 1. M-Files

### Script file

```matlab
% Implement scriptdemo.m script file
% Get all variables/formulas too
scriptdemo
```

<br>

### Function file

```matlab
% Implement freefall.m function file
freefall(12,68.1,0.25)
>> 53.1878
```

**freefall.m**

```matlab
function v = freefall(t, m, c)

g = 9.81
v = sqrt(g*m/c)*tanh(sqrt(g*c/m)*t)
```

<br>

function in MATLAB can return more than one value. 

```matlab
y = [8 5 10 12 6 7.5 4];
[m, s] = stats(y)
```

**stats.m**

```matlab
function [mean, stdev] = stats(x)
n = length(x);
mean = sum(x)/n;
stdev = sqrt(sum((x-mean).^2/(n-1)));
```



### Subfunction

함수 파일에 여러 개의 subfunciton 정의 가능. 

다만 파일명과 같은 함수만 호출 가능. 

```matlab
freefallsubfunc(t,m,c) % Can execute
vel(t,m,c) % Can't execute
```

**freefuncsubfunc.m**

```matlab
function v = freefallsubfunc(t,m,c)
v = vel(t,m,c);
end

function v = vel(t,m,c)
g = 9.81;
v = sqrt(g*m/c)*tanh(sqrt(g*c/m)*t);
end
```

<br>

## 2. Input / Output

### Input and Output

```matlab
% input
m = input('Mass (kg): ');
% output
velocity = 50.6175;
fprintf('The velocity is %8.4f m/s\n', velocity)
>> 'The velocity is 50.6175 m/s'
```

#### Format Code

| Format Code | Description                        |
| ----------- | ---------------------------------- |
| %d          | integer                            |
| %e          | scientific format with lowercase e |
| %E          | scientific format with uppercase E |
| %f          | decimal                            |
| %g          | the more compact of %e or %f       |

#### Control Code

| Control Code | Description    |
| ------------ | -------------- |
| \n           | Start new line |
| \t           | tab            |

<br>

### Save and Load

계산된 변수들을 읽거나 쓰기 가능

```matlab
save veldrag v c
>> save variable v, c in 'veldrag.mat' file 
load veldrag
>> load variable v, c in 'veldrag.mat' file 
```

<br>

<br>

## 3. Structured Programming

 ### 분기문

```matlab
% if
if grade >= 60
		disp('passing grade')
end
% same as
if grade >= 60, disp('passing grade'), end
% if - elseif - else
if x > 0
		sign = 1;
elseif x < 0
		sign = -1;
else
		sgn = 0;
end
% switch
switch grade
		case 'A'
				disp('Excellent')
		case 'B'
		...
		case 'F'
				disp('Would like fries with with your order?')
		otherwise
				disp('Huh!')
end
```

✋ **Tip**

`nargin`은 함수가 전달받은 파라미터 개수를 나타낸다. 

**freefall2.m**

```matlab
function v = freefall2(t, m, cd)

switch nargin
		case 0
				error('Must enter time ans mass')
		case 1
				error('Must enter mass')
		case 2
				cd = 0.27;
end
g = 9.81;
v = sqrt(g*m/c)*tanh(sqrt(g*c/m)*t);
```



<br>

### 반복문

```matlab
% for
for i = 1:5
		disp(i)
end
>> 1 2 3 4 5
% while
x = 8;
while x > 0
		x = x - 3
		disp(x)
end
>> 5 2
```

#### Vectorization

MATLAB에서 for문은 상당히 비효율적이다. 따라서 for문을 사용하는 대신 **Vectorization**을 사용할 수 있다. 

```matlab
% vectorization: Replace the unefficient 'for' loop
i = 0;
for t = 0:0.02:50 % unefficient for loop
		i = i + 1
		y(i) = cos(t);
end
% same as
t = 0:0.02:50
y = cos(t) % 각 원소에 cos 매핑 적용
```



<br>

#### Preallocation of Memory

MATLAB은 배열에 새로운 원소를 추가하게 되면 자동으로 배열의 크기를 증가시킨다. 

이는 코드의 유연성을 증가시켜 주지만, 매우 비효율적이다(특히 느리다).

```matlab
clear y
t = 0:.01:5;
for i = 1:length(t)
		if t(i) > 1
				y(i) = 1/t(i);
		else
				y(i) = 1;
		end
end
% Instead, preallocate y array
clear y
t = 0:.01:5;
y = ones(size(t)); % Preallocation
for i = 1:length(t)
		if t(i) > 1
				y(i) = 1/t(i);
		else
				y(i) = 1;
		end
end
```

<br>

<br>

## 3. Pass functions to M-files

### Anonymous Functions

```matlab
% f1 is function
% parameters are x and y, it return value of x^2 + y^2
f1 = @(x,y) x^2 + y^2
f1(3,4)
>> 25
% anonymous function can contain variable that exists in the workspace. 
a = 4;
b = 2;
f2 = @(x) a*x^b;
f2(3)
>> 36
% The variable value of created funciton doesn't changes. 
a = 3;
f2(3);
>> 36 % not 27
% But below code works. 
f2 = @(x) a*x^b;
f2(3)
>> 27
```

<br>

### Function Functions

```matlab
vel = @(t) sqrt(9.81 * 68.1 / 0.25) * tanh(sqrt(9.81 * 0.25 / 68.1)*t);
% fplot is function that plots the value of function
fplot(vel, [0 12]) 
```

```matlab
vel = @(t) sqrt(9.81 * 68.1 / 0.25) * tanh(sqrt(9.81 * 0.25 / 68.1)*t);
funcavg(vel, 0, 12, 60)
```

**funcavg.m**

```matlab
function favg = funcavg(f, a, b, n)

x = linspace(a, b, n);
y = f(x);
favg = mean(y);
```

<br>

### Passing Parameters

Anonymous function의 파라미터를 어떻게 바꿀 수 있을까?

MATLAB은 `varargin` 이라는 파라미터를 **function function** 의 마지막 파라미터로 추가할 수 있다. 

전달된 함수가 **function function** 내에서 호출될 때마다 `varargin{:}`을 이용해 마지막 인수로 전달되었던 함수 파라미터들을 전달해야 합니다. 

```matlab
vel = @(t,m,cd) sqrt(9.81*m/cd) * tanh(aqrt(9.81*cd/m)*t);
funcavg2(vel, 0, 12, 60, 68.1, 0.25)
funcavg2(vel, 0, 12, 60, 100, 0.28)
```

**funcavg2.m**

```matlab
function favg = funcavg(f, a, b, n, varargin)

x = linspace(a, b, n)
y = f(x, varargin{:});
favg = mean(y)
```

<br>

<br>

## 4. Others

### Pause

`pause` stops until there is a keyboard input. 

```matlab
for n = 3:10
		% magic: 마방진 행렬, mesh: 3차원 visualization
		mesh(magic(n))
		% stop
		pause
end
```

<br>

### Tic and Toc

`tic` and `toc` work together to measure elaped time. 

`beep` causes the computer to emit a beep sound. 

```matlab
tic
beep
pause(5)
beep
toc
```

<br>

### animation

`projectile` animates the code. 































