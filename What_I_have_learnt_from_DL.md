딥러닝 기초
===
## 머신러닝
* 데이터를 이용해 미지의 일을 예측하기 위해 만들어진 기법
* 일반적인 프로그램이 데이터를 넣어서 답을 도출하는 과정이라면,   
  데이터를 통해 규칙을 찾아내는 것이 머신러닝의 본질적 과정.
## 학습(training)
* 데이터가 입력되고 패턴이 분석되는 과정
* 예시 :
  1. 기존 환자 데이터를 입력(진료 기록과 사망·생존 여부)
  2. 머신러닝으로 학습(규칙 발견)
  3. 새로운 환자 예측
* 랜덤포레스트, SVM, DeepLearning 등 려러가지 머신러닝 기법들이 존재
## 예제를 통한 이해
### 데이터 살펴보기 : ThoraricSurgery.csv
* shape = (470, 18)
* 속성(attribute), 특성(feature) X : 수술 환자 기록 17개 변수(종양 유형, 폐활량, 호흡곤란 여부 등)
* 클래스 Y : 생존/사망
* 딥러닝을 구동시키려면 '속성'만을 뽑아 데이터셋을 만들고, '클래스'를 담는 데이터셋을 또 따로 만들어 줘야 함

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리를 불러옴
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 시드 설정
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터를 불러들임
Data_set = np.loadtxt("./csv_data/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = Data_set[:,0:17]
Y = Data_set[:, 17]

# 딥러닝 구조를 결정함(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 딥러닝을 실행함
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=100, batch_size=10)
```

</div>
</details>

</br>

* loss는 예측이 실패할 확률, accuracy는 예측이 성공할 확률.
* 예측 성공률은 데이터를 분석해 데이터를 확장하거나, 딥러닝 구조를 적절하게 바꾸는 등의 노력으로 더 향상될 수 있음.
* 뿐만 아니라 학습에 사용되지 않은 데이터를 따로 모아 테스트를 해보면서 예측 성공률이 정말로 가능한지를 확인하는 과정까지 거치게 됨.
* 이러한 '최적화 과정'을 진행하려면 딥러닝의 구동 원리를 이해해야 함

### 과정
* Sequential()함수는 딥러닝의 구조를 한층한층 쉽게 쌓아올릴 수 있게 해 줌.
* Sequential()함수를 선언하고 나서 model.add()함수를 사용해 필요한 층을 차례로 추가하면 됨.
* 위의 코드에서 model.add()함수를 이용해 두 개의 층을 쌓아 올림.
  * activation : 다음 층으로 어떻게 값을 넘길지 결정하는 부분.   
  가장 많이 사용되는 함수 : relu() 함수, sigmoid() 함수.
  * loss : 한 번 신경망이 실행될 때마다 오차 값을 추적하는 함수.
  * optimizer : 오차를 어떻게 줄요 나갈지 정하는 함수.
* 층의 개수는 데이터에 따라 결정.
* 딥러닝의 구조와 층별 옵션을 정하고 나면 complie()함수를 이용해 이를 실행
* 입력값이 네트워크 층을 거치면 예측값을 나오고, 이를 실제값과 비교해서 Loss Score를 계산한 후에 Optimizer를 통해 Weight를 업데이트 함.
#### 기타 예제

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 가상적인 데이터 생성
X = data = np.linspace(1, 2, 200)   # 시작값=1, 종료값=2, 개수=200
y = X * 4 + np.random.randn(200) * 0.3  # x를 4배로 하고 편차 0.3 정도의 가우시간 잡음 추가

model = Sequential()
model.add(Dense(1, input_dim=1, activation="linear"))
model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
model.fit(X, y, batch_size=1, epochs=30)

predict = model.predict(data)

plt.plot(data, predict, "b", data, y, "k.")   #첫번째 그래프는 파란색 마커로
plt.show()
# 두번째 그래프는 검정색 "."으로 그린다.
```
</div>
</details>

</br>

오차 수정
===
경사 하강법
---
### 1. 경사 하강법의 개요
* a를 무한대로 키우거나 a를 무한대로 작게 할 때 오차도 무한대로 커지는 관계를 이차함수 그래프로 표현할 수 있음.
$$y=ax+b$$
<center>

평균제곱오차$(MSE) = \frac{1}{n}\Sigma(\hat{y}_i-y_i)^2$
</center>

* 컴퓨터를 이용해 optimum을 찾으려면 임의의 한 점 $a_1$을 찍고, 이 점을 optimum에 가까운 쪽으로 점점 이동시키는 과정$(a_1 \rarr a_2 \rarr a_3)$이 필요함.
* 경사 하강법(gradient descent)   
  : 미분 기울기를 이용해서 그래프에서 오차를 비교하여 가장 작은 방향으로 이동시키는 방법.
  * 순간 변화율이 $0$인 점이 optimum.(2차 함수의 경우.)
  * $\therefore$ 임의의 점 $a_1$의 미분 값이 양$(+)$이면 음의 방향, 미분 값이 음$(-)$이면 양의 방향으로 얼마간 이동시킨 $a_2$에서 또다시 미분 값을 구함.
  * 이 과정을 반복해 기울기가 $0$인 optimum을 찾음. 
### 2. 학습률(Learning Rate)
* 학습률(learning rate)   
  : 어느 만큼 이동시킬지를 신중히 결정해야 하는데, 이때 이동 거리를 정해주는 것.
* DL에서 학습률의 값을 적절히 바꾸면서 최적의 학습률을 찾는 것은 중요한 __최적화(optimization)__ 과정.

$$MSE = \frac{1}{n}\sum(\hat{y}_i-y_i)^2$$
$$\frac{1}{n}\Sigma(\hat{y}_i-y_i)^2 = \frac{1}{n}\Sigma((ax_i+b)-y_i)^2, (\because\hat{y}_i = ax_i+b)$$
<center>

$a$로 편미분 한 결과 : $\frac{\partial{MSE}}{\partial{a}}=\frac{2}{n}\sum(ax_i+b-y_i)x_i$
</center>
<center>

$b$로 편미분 한 결과 : $\frac{\partial{MSE}}{\partial{b}}=\frac{2}{n}\sum(ax_i+b-y_i)$
</center>

### 3. 코딩으로 확인하는 경사 하강법

<details>
<summary>실습 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간 x와 성적 y의 리스트를 만듦
data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 나타냄.
plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()
```
```python
# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줌.
# 인덱스를 주어 하나씩 불러와 계산이 가능하도록 하기 위함.

x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 절편 b의 값을 초기화 함
a = 0
b = 0

# 학습률을 결정
lr = 0.03

# 반복 횟수 결정. 오차수정(경사하강법) 횟수
epochs = 2001

# 경사 하강법을 시작
for i in range(epochs) :
    y_hat = a * x_data + b
    error = y_data - y_hat
    a_diff = -(2/len(x_data)) * sum(x_data * (error))   # 오차함수를 a로 미분한 값.
    b_diff = -(2/len(x_data)) * sum(error)    # 오차함수를 b로 미분한 값.
    a = a - lr * a_diff # 학습률을 반영하여 업데이트
    b = b - lr * b_diff
    if i % 100 == 0 :
        print(f"epoch={i}, 기울기={a:.3f}, 절편={b:.4f}")
```
```python
# 앞서 구한 기울기와 절편을 이용해 그래프를 그려 봄
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()
```

</div>
</details>

</br>

### 4. 다중선형 회귀란
* w 파라미터의 개수가 적다면 고차원 방정식으로 비용 함수가 되는 w 변숫값을 도출할 수 있겠지만, w 파라미터가 많으면 고차원 방정식을 동원하더라도 해결하기 어려움. 경사 하강법은 이러한 고차원 방정식에 대한 문제를 해결해주면서 비용 함수 RSS를 최소화하는 방법을 직관적으로 제공하는 뛰어난 방식.
* $R(w)$는 변수가 w 파라미터로 이루어진 함수.
$$R(w) = \frac{1}{N}\displaystyle\sum_{i=1}^n(y_i-(w_0+w_1*x_i))^2$$
<center>

$\displaystyle\frac{\partial{R(w)}}{\partial{w_1}}=\frac{2}{N}\displaystyle\sum_{i=1}^n-x_i(y_i-(w_0+w_1x_i))=-\frac{2}{N}\displaystyle\sum_{i=1}^nx_i*$(실제값$_i-$예측값$_i$)

$\displaystyle\frac{\partial{R(w)}}{\partial{w_0}}=\frac{2}{N}\displaystyle\sum_{i=1}^n-(y_i-(w_0+w_1x_i))=-\frac{2}{N}\displaystyle\sum_{i=1}^n$(실제값$_i-$예측값$_i$)
</center>

* $w_1, w_0$의 편미분 결과값인 $-\frac{2}{N}\displaystyle\sum_{i=1}^nx_i*$(실제값$_i-$예측값$_i$) 와 $-\frac{2}{N}\displaystyle\sum_{i=1}^n$(실제값$_i-$예측값$_i$) 을 반복적으로 보정하면서 $w_1, w_0$값을 업데이트 하면 비용함수 $R(w)$가 최소가 되는 $w_1, w_0$값을 구할 수 있음. 하지만 실제로는 앞의 편미분 값이 너무 클 수 있기 때문에 보정계수 $\eta$를 곱함 $\rarr$ "학습률"
  * 새로운 $w_1 =$ 이전 $w_1 - \eta\frac{2}{N}\displaystyle\sum_{i=1}^nx_i*$(실제값$_i-$예측값$_i$)
  * 새로운 $w_0 =$ 이전 $w_0 - \eta\frac{2}{N}\displaystyle\sum_{i=1}^n$(실제값$_i-$예측값$_i$)
  * 비용 함수의 값이 감소했으면 다시 $w_1, w_0$를 업데이트 하여 다시 비용 함수의 값 계산. 더 이상 비용 함수의 값이 감소하지 않으면 그 때의 $w_1, w_0$를 구하고 반복을 중지함.
### 5. 코딩으로 확인하는 다중 선형 회귀

<details>
<summary>실습 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)
# y = 4X + 6 식을 근사(w1 = 4, w0 = 6). random 값은 Noise를 위해 만듦.
X = 2 * np.random.rand(100,1)
y = 6 + 4 * X + np.random.randn(100, 1)

# X , y 데이터 셋 scatter plot으로 시각화
print(X.shape, y.shape)
plt.scatter(X, y)
```

* $w_0$와 $w_1$의 값을 최소화 할 수 있도록 업데이트를 수행하는 함수 생성.
  * 예측 배열 y_pred는 np.dot(X,w1.T) + w0임. 100개의 데이터 X(1,2,$\dots$,100)이 있다면 예측값 $w_0 + X_1w_1 + X_2w_1 + \dots + X_{100}w_1$이며, 이는 입력 배열 X와 $w_1$ 배열의 내적임.
  * 새로운 $w_1$과 $w_0$를 update함

```python
# w1과 w0 를 업데이트 할 w1_update, w0_update를 반환.
def get_weight_updates(w1, w0, X, y, learning_rate=0.01) :
    N = len(y)
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0
    diff = y - y_pred

    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성
    w0_factors = np.ones((N,1))

    # w1과 w0를 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))

    return w1_update, w0_update
```
```python
w0 = np.zeros((1,1))
w1 = np.zeros((1,1))
y_pred = np.dot(X, w1.T) + w0
diff = y - y_pred
print(diff.shape)

w0_factors = np.ones((100,1))
w1_update = -(2/100)*0.01*(np.dot(X.T, diff))
w0_update = -(2/100)*0.01*(np.dot(w0_factors.T, diff))
print(w1_update.shape, w0_update.shape)
w1, w0
```

* 반복적으로 경사 하강법을 이용하여 get_weight_updates()를 호출하여 $w_1$과 $w_0$를 업데이트 하는 함수 생성

```python
# 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함.
def gradient_descent_steps(X, y, iters=10000) :
    # w0와 w1을 모두 0으로 초기화.
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))

    # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 호출하여 w1, w0 업데이트 수행
    for ind in tqdm(range(iters)) :
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    
    return w1, w0
```

* 예측 오차 비용을 계산하는 함수 생성 및 경사 하강법 수행

```python
def get_cost(y, y_pred) :
    N = len(y)
    cost = np.sum(np.square(y - y_pred))/N
    return cost

w1, w0 = gradient_descent_steps(X, y, iters=1000)
print(f"w1:{w1[0,0]:.3f}, w0:{w0[0,0]:.3f}")
y_pred = w1[0,0] * X + w0
print(f"Gradient Descent Total Cost : {get_cost(y, y_pred):.4f}")
```

```python
# 시각화
plt.scatter(X, y)
plt.plot(X, y_pred)
```

* 확률적 경사하강법 함수 작성

```python
def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000) :
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index = 0

    for ind in tqdm(range(iters)) :
        np.random.seed(ind)
        # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 sample_X, smaple_y로 저장
        stochastic_random_index = np.random.permutation(X.shape[0])
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]

        # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update, w0_update 계산 후 업데이트
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update

    return w1, w0
```

```python
w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print(f"w1 : {round(w1[0,0], 3)}, w0 : {round(w0[0,0], 3)}")
y_pred = w1[0,0] * X + w0
print(f"Stochastic Gradient Descent Total Cost : {get_cost(y, y_pred):.4f}")
```

* 더 정확한 예측을 하려면 추가 정보를 입력해야 하며, 정보를 추가해 새로운 예측값을 구하려면 변수의 개수를 늘려 다중 선형 회귀를 만들어 주어야 함.
  * 기존의 독립변수 '공부한시간'$(x_1)$ 외에 '과외 수업 횟수'$(X_2)$ 변수를 추가해서 종속변수 '성적'$(y)$ 예측
  <table>
    <tr>
      <th>
          공부한 시간(x_1)
      </th>
      <td>
          2
      </td>
      <td>
          4
      </td>
      <td>
          6
      </td>
      <td>
          8
      </td>
    </tr>
    <tr>
      <th>
          과외 수업 횟수(x_2)
      </th>
      <td>
          0
      </td>
      <td>
          4
      </td>
      <td>
          2
      </td>
      <td>
          3
      </td>
    </tr>
    <tr>
      <th>
          성적(y)
      </th>
      <td>
          81
      </td>
      <td>
          93
      </td>
      <td>
          91
      </td>
      <td>
          97
      </td>
    </tr>
  </table>
  
* 이를 이용한 종속 변수 $y$를 만들 경우 다음과 같은 식이 나옴.
$$ y = a_1x_1 + a_2x_2 + b $$
* 경사 하강법을 이용하여 $a_1, a_2$를 구함.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부시간 x와 y의 리스트를 만듦
# 이번에는 독립변수 x의 값이 두 개이므로 다음과 같이 리스트 작성
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 확인
ax = plt.axes(projection="3d")
ax.set_xlabel("study_hours")
ax.set_ylabel("private_class")
ax.set_zlabel("Score")
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()
```

```python
from tqdm import tqdm
# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줌.
# 인덱스를 주어 하나씩 불러와 계산이 가능하도록 하기 위함.

x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b의 값을 초기화 함
a1 = 0
a2 = 0
b = 0

# 학습률을 결정
lr = 0.02

# 반복 횟수 결정. 오차수정(경사하강법) 횟수
epochs = 2001

# 경사 하강법을 시작
for i in tqdm(range(epochs)) :
    y_hat = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_hat
    a1_diff = -(2/len(x1_data)) * sum(x1_data * (error))   # 오차함수를 a1로 미분한 값.
    a2_diff = -(2/len(x2_data)) * sum(x2_data * (error))   # 오차함수를 a2로 미분한 값.
    b_new = -(2/len(x1_data)) * sum(error)    # 오차함수를 b로 미분한 값. len(x1_data) = N.
    a1 = a1 - lr * a1_diff # 학습률을 반영하여 업데이트
    a2 = a2 - lr * a2_diff
    b = b - lr * b_new
    if i % 100 == 0 :
        print(f"epoch={i}, 기울기1={a1:.3f}, 기울기2={a2:.3f}, 절편={b:.4f}")
```

* 2차원 예측 직선이 3차원 '예측 평면'으로 바뀜.
* 2차원 직선에서만 움직이던 예측 결과가 더 넓은 평면 범위 안에서 움직이게 됨.
* 이로 인해 좀 더 정밀한 예측을 할 수 있게 됨.

</div>
</details>

</br>

참 거짓 판단장치
===
로지스틱 회귀
---
### 1. 로지스틱 회귀의 정의
* 전달받은 정보를 놓고 참과 거짓 중에 하나를 판단해 다음 단계로 넘기는 장치들이 딥러닝 내부에서 쉬지 않고 작동함
* 딥러닝을 수행한다는 것은 겉으로 드러나지 않는 '미니 판단 장치'들을 이용해서 복잡한 연산을 해낸 끝에 최적의 예측 값을 내놓는 작업
* 참인지 거짓인지를 구분하는 로지스틱 회귀의 원리를 이용해 '참, 거짓 미니 판단 장치'를 만들어, 주어진 입력 값의 특징을 추출함(학습, train). 이를 저장해서 '모델(model)'을 만듦.
* 누군가 비슷한 질문을 하면 지금까지 만들어 놓은 이 모델을 꺼내어 답을 함(예측, prediction) 이것이 딥러닝의 동작 원리.
* 직선으로 해결하기에는 적절하지 않은 경우도 있음.
  * 점수가 아니라 오직 합불만 발표되는 시험이 있다고 가정함.
    * 합격을 1, 불합격을 0이라고 하고, 좌표평면 상에서 나타내면 직선으로 분포를 나타내기 어려움.
    * 점들의 특성을 정확하게 담아내려면 직선이 아닌 S자 형태가 필요 $\rarr$ 시그모이드 함수 등
* 로지스틱 회귀 :   
  선형 회귀와 마찬가지로 적절한 선을 그려가는 과정
  * 다만 직선이 아니라, 참(1)과 거짓(0) 사이를 구분하는 S자 형태의 선을 그어주는 작업.
### 2. 시그모이드 함수
* 시그모이드 함수(sigmoid function) : S자 형태로 그래프가 그려지는 함수 $\rarr$ 로지스틱 회귀를 위해서는 시그모이드 함수가 필요
$$Sigmoid : y = \frac{1}{1+e^{-(ax+b)}}$$
  * a는 그래프의 경사도를 결정함.
    * a값이 커지면 그래프의 경사가 커지고, a값이 작아지면 그래프의 경사가 작아짐.  
  * b는 그래프의 좌우 이동을 결정함. 
    * b값이 커지면 그래프가 왼쪽으로 이동하고, b값이 작아지면 그래프가 오른쪽으로 이동함. 
### 3. 오차 공식
* a와 오차와의 관계 :
  * a가 작아질수록 오차는 무한대로 커짐
  * 하지만 a가 커진다고 해서 오차가 없어지지는 않음.
* b와 오차와의 관계:
  * b값이 너무 작아지거나 커지면 오차도 이에 따라 커짐.
* 시그모이드 함수에서 a, b 값을 구하는 방법 역시 경사하강법.
  * 경사 하강법은 먼저 오차를 구한 다음 오차가 작은 쪽으로 이동시키는 방법이므로 여기서도 오차(예측값과 실제 값의 차이)를 구하는 공식이 필요.
### 4. 로그 함수
* 시그모이드 함수의 특징은 y값이 0과 1 사이라는 것.
  * 실제 값이 1일 때 예측 값이 0에 가까워지면 오차가 커짐.
  * 반대로, 실제값이 0일 때 예측 값이 1에 가까워지는 경우에도 오차는 커짐.
* 이를 공식으로 만들 수 있게 해주는 함수가 바로 로그 함수.
$$-[y_{data}\log{h}+(1-y_{data})\log{(1-h)}]$$
* 실제 값이 1이면 뒷부분$((1-y_{data})\log{(1-h)})$이 없어지고, 실제 값이 0이면 앞부분$(y_{data}\log{h})$이 없어짐.
* 실제 값에 따라 앞부분과 뒷부분 각각의 그래프를 사용할 수 있음.
### 5. 코딩으로 확인하는 로지스틱 회귀

<details>
<summary>실습 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 공부시간 x와 성적 y의 리스트를 만듦
data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

# 그래프로 나타냄.
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
plt.show()
```
```python
# 기울기 a와 절편 b의 값을 초기화
a = 0
b = 0

# 학습률을 정함
lr = 0.05

# 시그모이드 함수를 정의
def sigmoid(x) :
    return 1 / (1 + np.e ** (-x))

# 경사 하강법을 실행
for i in tqdm(range(2001)) :
    for x_data, y_data in data :
        a_diff = x_data * (sigmoid(a*x_data + b) - y_data)
        b_diff = sigmoid(a*x_data + b) - y_data
        a = a - lr * a_diff
        b = b - lr * b_diff
        if i % 1000 == 0 :
            print(f"epoch={i}, slope={a:.3f}, intercept={b:.4f}")
```
* 시그모이드 형태의 함수가 잘 만들어지도록 a와 b의 값이 수렴된 것을 알 수 있음.
* 만약 여기에 입력 값이 추가되어 세 개 이상의 입력 값을 다룬다면(__다중 분류 문제__) 시그모이드 함수가 아니라 __소프트맥스(softmax)__라는 함수를 써야 함
* Sigmoid에서 시작된 활성화 함수는 ReLU를 비롯해 다양한 종류가 있음.

</div>
</details>

</br>

### 6. 로지스틱 회귀에서 퍼셉트론으로
* 입력 값을 통해 출력 값을 구하는 함수 $y$는 다음과 같이 표현할 수 있음
$$y = a_1x_1 + a_2x_2 + b$$
  * 입력 값 : 우리가 가진 값인 $x_1, x_2$
  * 출력 값 : 계산으로 얻는 값 $y \rarr$ 출력 값 $y$를 구하려면 가중치(weight) $a_1$값, $a_2$값 그리고 편향(bias) $b$값이 필요함.
* $x_1$과 $x_2$가 입력되고, 각각 가중치 $a_1, a_2$를 만남. 여기에 $b$값을 더한 후 시그모이드 함수를 거쳐 1 또는 0의 출력값 $y$를 출력함.
* 프랑크 로젠플라트가 퍼셉트론(perceptron)이라는 이름을 붙임.
* 퍼셉트론은 이후 인공신경망(ANN), 오차 역전파 등의 발전을 거쳐 지금의 딥러닝으로 발전됨.

# 퍼셉트론
## 가중치, 가중합, 바이어스, 활성화 함수
* 인간의 뇌는 약 1000억개의 뉴런으로 이루어져 있고, 뉴런과 뉴련 사이에는 시냅스라는 연결 부위 존재
* 신경 말단에서 자극을 받으면 시냅스에서 화학 물질이 나와 임계값을 넘으면 전위 변화를 일으키는데, 이 매커니즘이 로지스틱 회귀와 많이 닮음. (입력 값을 놓고 활성화 함수에 의해 일정 수준이 넘으면 참, 그렇지 않으면 거짓을 내보내는 회로)
* 인공신경망 : 뉴런과 뉴런의 연결처럼 퍼셉트론의 연결을 통해 입력 값에 대한 판든을 하게 하는 것.
* 여러 층의 퍼셉트론을 서로 연결시키고 복잡하게 조합하여 주어진 입력 값에 대한 판든을 하게 하는 것이 바로 신경망의 기본 구조임.
* 퍼셉트론은 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 기본 단위.
* 가중합(weighted sum) : 입력 값 $x$와 가중치 $w$의 곱을 모두 더한 다음 거기에 바이어스 $b$를 더한 값.
* 활성화 함수(activation function) : 가중합의 결과를 놓고 1 또는 0을 출력해서 다음으로 보내는데, 여기서 0과 1을 판단하는 함수. (Sigmoid, ReLU, Softmax 함수 등)
## 퍼셉트론의 과제
* 단 하나의 퍼셉트론으로는 많은 것을 기대할 수 가 없음.
* 퍼셉트론의 기능은 본질적으로 선을 긋는 작업을 수행하는 것.
* 하나의 퍼셉트론만으로는 해결 불가능한 문제가 존재함 : XOR 문제
## XOR 문제
* AND 게이트
* OR 게이트
  * AND와 OR
  *  게이트는 직선을 그어 결과값이 1인 값을 구할 수 있음.
* XOR 게이트 : $x_1, x_2$ 둘 중 하나만 1일 때만 결과 값이 1로 출력되는 게이트.
  * XOR 게이트의 경우 선을 그어 구분할 수 없음.
* 다층 퍼셉트론을 이용하여 XOR 문제 해결.

다층 퍼셉트론
===
## 1. 다층 퍼셉트론의 설계
* 단일 퍼셉트론으로는 XOR 문제를 해결할 수 없음.
  * 예를 들어, 성냥개비 여섯 개로 정삼각형 네 개를 만들기 위해서는,   
  → **2차원 평면**이 아닌 __3차원의 정사면체__ 모양으로 쌓아 해결.
* XOR 문제를 극복하는 것 역시 평면을 휘어주면서 해결.
* 다층 퍼셈트론 : 좌표 평면 자체에 변화를 주는 것으로 XOR 문제를 해결.
* 은닉층(hidden layer)을 만들면 우리는 두 개의 퍼셉트론을 한 번에 계산할 수 있게 됨.
* 가운데 숨어있는 은닉층으로 퍼셉트론이 각각 자신의 가중치$w$와 바이어스$b$ 값을 보냄.
* 이 은닉층에 모인 값이 시그모이드 함수$(\sigma)$를 이용해 최종 값으로 결과를 보냄.
* 노드(node) : 은닉층에 모이는 중간 정거장.
## 2. XOR 문제의 해결
* 은닉층의 노드 $n_1$과 $n_2$의 값은 각각 단일 퍼셉트론의 값과 같음
$$n_1 = \sigma(x_1w_{11} + x_2w_{21} + b_1)$$
$$n_2 = \sigma(x_1w_{12} + x_2w_{22} + b_2)$$
* $n_1, n_2$ 결과값이 출력층으로 보내짐.
* 출력층에서는 역시 시그모이드 함수를 통해 $y$값이 정해짐.
  * 이 값을 $y_{out}$이라 할 때 식으로 표현하면 다음과 같음.
$$y_{out}=\sigma(n_1w_{31} + n_2w_{32} + b_3)$$
</br>

* 이제 각각의 가중치 $w$와 바이어스 $b$의 값을 정할 차례.
* 2차원 배열로 늘어놓으면 다음과 같이 표시할 수 있음.
  * 은닉층을 포함해 가중치 6개와 바이어스 3개.
$$W(1) = \begin{bmatrix} w_{11} \ w_{12} \\ w_{21} \ w_{22} \end{bmatrix} \quad B(1) = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$
$$W(2) = \begin{bmatrix} w_{31} \\ w_{22} \end{bmatrix} \quad B(2) = \begin{bmatrix} b_3 \end{bmatrix}$$
* NAND(Negative AND)게이트 : AND 게이트의 정반대 값을 출력함.
* NAND 게이트와 OR 게이트, 이 두가지를 내재한 각각의 퍼셉트론이 다중 레이어 안에서 각각 작동하고, 이 두가지 값에 대해 AND 게이트를 수행한 결과 값이 $y_{out}$.
  * 숨어있는 2개의 노드를 둔 다층 퍼셉트론을 구성해 XOR 문제를 해결할 수 있음.
## 3. 코딩으로 XOR 문제 해결하기

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np

# 가중치(w11, w12, w2)와 바이어스(b1, b2, b3)
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

# 퍼셉트론 함수 정의 : 0 또는 1을 출력
def MLP(x, w, b) :
    y = np.sum(w * x) + b
    if y <= 0 :
        return 0
    else :
        return 1

# NAND 게이트
def NAND(x1, x2) :
    return MLP(np.array([x1, x2]), w11, b1)

# OR 게이트
def OR(x1, x2) :
    return MLP(np.array([x1, x2]), w12, b2)

# AND 게이트
def AND(x1, x2) :
    return MLP(np.array([x1, x2]), w2, b3)

# XOR 게이트
def XOR(x1, x2) :
    return AND(NAND(x1, x2), OR(x1, x2))
```
```python
# x1, x2 값을 번갈아 대입해 가며 최종값 출력
if __name__ == '__main__' :
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)] :
        y = XOR(x[0], x[1])
        print(f"입력 값 : {str(x)} 출력값 : {str(y)}")
```

</div>
</details>

</br>


신경망에서 딥러닝으로
===
## 1. 기울기 소실 문제와 활성화 함수
### DNN과 기울기 소실 문제
* DNN은 MLP(다층 퍼셉트론)에서 은닉층의 개수를 증가시킨 것.
  * "딥(deep)"이라는 용어는 은닉층이 깊다는 것을 의미함.
  * 최근 딥러닝은 컴퓨터 비전, 음성 인식, 자연어 처리, 소셜 네트워크 필터링, 기계 번역 등에 적용되어서 인간 전문가에 필적하는 결과를 얻고 있음.
* 기울기 소실(gradient vanishing) 문제 : 은닉층이 많아지면 출력층에서 계산된 기울기가 역전파되다가 값이 점점 작아져서 없어짐.
* 과잉 적합(over fitting)
  * 퍼셉트론에서는 계단 함수(step function)를 활성화 함수로 사용했지만, MLP에서는 다양한 비선형 함수들을 활성화 함수로 사용함.
  * Sigmoid, TanH, ReLU 등
* 다층 퍼셉트론이 오차 역전파를 만나 신경망이 되었고, 신경망은 XOR 문제를 해결.
* 신경망을 쌓아 올려도 사람처럼 생각하고 판단하는 인공지는이 완성되지는 않음.   
→ 원인은 기울기 소실 문제.
### 기울기 소실 문제를 해결하기 위한 활성화 함수의 변화
* 가중치를 수정하려면 미분 값, 즉 기울기가 필요함. 층이 늘어나면서 기울기 값이 점점 작아져 맨 처음 층까지 전달되지 않는 기울기 소실 문제 발생.
  * 기울기 소실 문제가 발생하기 시작한 것은 활성화 함수로 사용된 시그모이드 함수의 특성 때문.
  * 시그모이드 함수의 특성상 아주 큰 양수나 아주 큰 음수가 들어오면 출력이 포화되어서 거의 0이 됨.
  * 시그모이드 함수를 미분하면 최대치가 0.3이므로 1보다 작으므로 계속 곱하다 보면 0에 가까워짐.
  * 이를 대체하기 위해 활성화 함수를 시그모이드가 아닌 다른 함수들로 대체함.
    * tanH, ReLU, softplus 등
        1. Hyperbolic Tangent function
           * 미분한 값의 범위가 함께 확장되는 효과를 가져옴.
           * 여전히 1보다 작은 값이 존재하므로 기울기 소실 문제는 사라지지 않음.
        2. ReLU function
           * 시그모이드 함수의 대안으로 떠오르며 현재 가장 많이 사용되는 활성화 함수임.
           * 여러 은닉층을 거치며 곱해지더라도 맨 처음 층까지 사라지지 않고 남아있을 수 있음.
           * 간단한 방법을 통해 여러 층을 쌓을 수 있게 했고, 딥러닝의 발전에 속도가 붙게 됨.
        3. softplus function
           * 이후 렐루의 0이 되는 순간을 완화
## 2. 속도와 정확도 문제를 해결하는 고급 경사 하강법
* 경사 하강법은 정확하게 가중치를 찾아가지만, 한 번 업데이트할 때마다 전체 데이터를 미분해야 하므로 계산량이 매우 많다는 단점이 있음.
* 경사 하강법의 불필요하게 많은 계산량은 속도를 느리게 할 뿐 아니라, 최적 해를 찾기 전에 최적화 과정을 멈추게 할 수도 있음.
* 이러한 점을 보완한 고급 경사 하강법이 등장하면서 딥러닝의 발전 속도는 더 빨라짐.
### 1. 확률적 경사 하강법(Stochastic Gradient Descent, SGD)
* 전체 데이터를 사용하는 것이 아니라, 랜덤하게 추출한 일부 데이터를 사용함.
* 일부 데이터를 사용하므로 더 빨리 그리고 자주 업데이트를 하는 것이 가능해짐.
* 속도가 빠르고 최적 해에 근사한 값을 찾아낸다는 장점 덕분에 경사 하강법의 대안으로 사용됨.   
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/SGD.png">

### 2. 모멘텀(Momentum)
* 모멘텀이란 단어는 '관성, 탄력, 가속도'라는 뜻.
* 모멘텀 SGD란 말 그대로 경사 하강법에 탄력을 더해 주는 것.
* 다시 말해서, 경사 하강법과 마찬가지로 매번 기울기를 구하지만, 이를 통해 오차를 수정하기 전 바로 앞 수정 값과 방향$(+, -)$을 참고하여 같은 방향으로 일정한 비율만 수정되게 하는 방법.
* 수정 방향이 양수$(+)$ 방향으로 한 번, 음수$(-)$ 방향으로 한 번 지그재그로 일어나는 현상이 줄어들고, 이전 이동 값을 고려하여 일정 비율만큼만 다음 값을 결정하므로 관성의 효과를 낼 수 있음.   
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/momentum.png">

### 3. 현재 가장 많이 사용되는 고급 경사 하강법
* 아담(ADAM) : Momentum과 RMSProp 방법을 합친 방법
  * 정확도와 보폭 크기 개선
  * keras.optimizers.Adam(lr, beta_1, beta_2, epsilon, decay)
  * RMPSProp : Adagrad의 보폭 민감도를 보완한 방법
  * Adagrad : 변수의 업데이트가 잦으면 학습률을 적게하여 이동 보폭을 조절하는 방법.
### 4. 손실함수
* 손실 함수로는 이제까지는 제곱 오차 함수(Mean Squared Error : MSE)
$$\Epsilon=\frac{1}{2}\displaystyle\sum_{i=1}^n(t-o)^2$$
* 노드의 활성화 함수로 시그모이드(sigmoid)가 사용된다면 MSE는 저속 수렴 문제(slow convergence)에 부딪치게 됨.
  * 예를 들어서 목표값이 0.0이고 출력값이 10.0이라고 하면, 차이는 무려 10.0이나 되지만 시그모이드 함수의 그래디언트는 거의 0이 됨.
  * 반대로 목표값이 0.0이고 출력값이 1.0이라고 하면, 차이는 1.0뿐이지만 그래디언트는 0.2 정도가 나옴.
  * 이것은 마치 시험 성적이 안좋은 학생에게 더 좋은 학점을 주는 격.
## 3. 코드 실습


<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import tensorflow as tf

batch_size = 128    # 가중치를 변경하기 전에 처리하는 샘플의 개수
num_classes = 10    # 출력 클래스의 개수
epochs = 20         # 에포크의 개수

# 데이터를 학습 데이터와 테스트 데이터로 나눔.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 입력 이미지를 2차원에서 1차원 벡터로 변경함.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 입력 이미지의 픽셀 값이 0.0에서 1.0 사이의 값이 되게 함.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 클래스의 개수에 따라서 하나의 출력 픽셀만이 1이 되게 함.
# 예들 들면 1 0 0 0 0 0 0 0 0 0 과 같음.

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 신경망의 모델을 구축함.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation="sigmoid", input_shape=(784,)))
model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))

model.summary()

sgd = tf.keras.optimizers.SGD(lr=0.1)

# 손실 함수를 제곱 오차 함수로 설정하고, 학습 알고리즘은 sgd 방식으로 함.
model.compile(loss="mean_squared_error",
            optimizer = sgd,
            metrics = ["accuracy"])

# 학습을 수행함.
history = model.fit(x_train, y_train,
                batch_size = batch_size,
                epochs = epochs)

# 학습을 평가함.
score = model.evaluate(x_test, y_test, verbose=0)
print(f"테스트 손실값 : {score[0]}")
print(f"테스트 정확도 : {score[1]}")

#테스트 손실값 : 0.031353116035461426
#테스트 정확도 : 0.8655999898910522
```

</div>
</details>

</br>

오차 역전파
===
## 1. 오차 역전파의 개념
* 신경망 내부의 가중치는 오차 역전파 방법을 사용해 수정함.
  * 오차 역전파는 경사 하강법의 확장 개념.
  * 가중치를 구하는 방법은 경사 하강법을 그대로 이용하면 됨.
  * 임의의 가중치를 선언하고 결과값을 이용해 오차를 구한 뒤 이 오차가 최소인 지점으로 계속해서 조금씩 이동시킴(최적화 과정)   
→ 오차 역전파(back propagation)
  * 오차가 최소가 되는 점(미분했을 때 기울기가 0이 되는 지점)을 찾는 과정.
* 역전파 알고리즘은 입력이 주어지면 순방향으로 계산하여 출력을 계산한 후에 실제 출력과 우리가 원하는 출력간의 오차를 계산함.
* 이 오차를 역방향으로 전파하면서 오차를 줄이는 방향으로 가중치를 변경함.
### 오차 역전파(back propagation) : 다층 퍼셉트론에서의 최적화 과정
* 오차 역전파 구동방식은 다음과 같이 정리할 수 있음.
    1. 임의의 초기 가중치 $w$를 준 뒤 결과 $y_{out}$를 계산함.
    2. 계산 결과와 우리가 원하는 값 사이의 오차를 구함.
    3. 경사 하강법을 이용해 바로 앞 가중치를 오차가 작아지는 방향으로 가중치 업데이트 함.
    4. 위 과정을 더이상 오차가 줄어들지 않을 때까지 반복함.
* '오차가 작아지는 방향으로 업데이트'는 의미는 미분 값이 0에 가까워지는 방향으로 나아간다는 말.
* 즉, '기울기가 0이 되는 방향'으로 나아가야 하는데, 이 말은 가중치에서 기울기를 뺐을 때 가중치의 변화가 전혀 없는 상태를 말함.
* 오차 역전파 : 가중치에서 기울기를 빼도 값의 변화가 없을 때까지 계속 가중치 수정 작업을 반복하는 것.
$$W_{t+1} = W_t-\frac{\partial \Epsilon}{\partial W}$$
* 새 가중치는 현 가중치에서 '가중치에 대한 기울기'를 뺀 값
## 2. 오차 역전파의 과정
* 입력된 실제 값고 다층 퍼셉트론의 계산 결과를 비교하여 가중치를 역전파 방식으로 수정하는 알고리즘.
  1. 환경 변수 지정 : 환경 변수에는 입력 값과 타깃 결과값이 포함된 데이터셋, 학습률 등이 포함되고 활성화 함수와 가중치 등도 선언되어야 함.
  2. 신경망 실행 : 초깃값을 입력하여 활성화 함수와 가중치를 거쳐 결과값이 나오게 함.
  3. 결과를 실제 값과 비교 : 오차를 측정함.
  4. 역전파 실행 : 출력층과 은닉층의 가중치를 수정함.
  5. 결과 출력.

데이터 다루기
===
## 1. 딥러닝과 데이터
* 데이터의 양보다 훨씬 중요한 것은 '필요한' 데이터가 얼마나 많은가임.   
  머신러닝 프로젝트의 성공과 실패는 얼마나 좋은 데이터를 가지고 시작하느냐에 영향을 많이 받음.   
  데이터가 우리가 사용하려는 머신러닝, 딥러닝에 얼마나 효율적으로 가공됐는지가 중요
* 여기서 좋은 데이터란 내가 알아내고자 하는 정보를 잘 담고 있는 데이터, 한쪽으로 치우치지 않고, 불필요한 정보를 가지고 있지 않으며, 왜곡되지 않은 데이터.
* 목적에 맞춰 가능한 한 많은 정보를 못았다면 이를 머신러닝과 딥러닝에서 사용할 수 있게 데이터 가공을 잘 해야 함.
## 2. 피마 인디언 데이터 분석
* 피마 인디언 당뇨병 예측 사례 : 당뇨가 유전 및 환경, 모두의 탓이라는 것을 증명하는 좋은 사례
* 데이터의 각 정보가 의미하는 의학, 생리학 배경 지식을 모두 알 필요는 없지만, 딥러닝을 구동하려면 반드시 속성과 클래스를 먼저 구분해야 함.
* 모델의 정확도를 향상시키기 위해서는 데이터의 추가 및 재가공이 필요할 수도 있음.
* 딥러닝의 구동에 앞서 데이터의 내용과 구조를 잘 파악하는 것이 중요.
## 3. pandas를 이용한 데이터 조사
* 데이터를 잘 파악하는 것이 딥러닝을 다루는 기술의 1단계.
* 데이터의 크기가 커지고 정보량이 많아지면 내용을 파악할 수 있는 효과적인 방법이 필요함. 이때 가장 유용한 방법이 데이터를 시각화해서 눈으로 직접 확인해 보는 것.
  
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# pandas 라이브러리를 불러옴.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋을 불러옴. 불러올 때 각 칼럼에 해당하는 이름을 지정함.
df = pd.read_csv("csv_data\diabetes.csv")
df.columns = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"]
```
```python
df.head(5)
```
```python
df.info()
```
```python
df.describe()
```
</div>
</details>

</br>

## 4. 데이터 가공
* 데이터를 잘 다루려면 데이터를 한 번 더 가공해야 함
* 이 프로젝트의 목적은 당뇨병 발병을 예측하는 것. 모든 정보는 당뇨병 발병과 어떤 관계가 있는지임.
* 임심 횟수 당 당뇨병 발병 확률의 경우 다음과 같이 계산할 수 있음.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
df[["pregnant", "class"]] \
    .groupby(["pregnant"], as_index=False) \
        .mean().sort_values(by="pregnant", ascending=True)
# 2개 컬럼만 선택하여 \ pregnant 기준으로 그룹바이 \ 평균을 낸 뒤 pregnant 기준 오름차순 정렬
```
</div>
</details>

</br>

모델 설계
===
## 1. 모델의 정의
* Sequential() 함수를 model로 선언해놓고 model.add()라는 라인을 추가하여 새로운 층을 추가함.
* 아래의 예시 코드에서는 model.add()로 시작하는 라인이 두 개 있으므로, 두 개의 층을 가진 모델을 만든 것이고, 가장 마지막 층이 결과를 출력하는 '출력층', 나머지는 모두 '은닉층'이 됨.
* model.add(Dense(30, input_dim=17, activation="relu"))
  * node 30개, 입력 데이터의 개수는 17(input_dim)
  * keras는 입력층을 따로 만드는 것이 아니라, 첫번째 은닉층에 input_dim을 적어서 첫 번째 Dense가 은닉층+입력층의 역할을 겸함.
  * 예시의 데이터(폐암 환자의 생존 여부)는 17개의 입력값(독립변수)가 있으므로, 데이터에서 17개의 값을 받아 은닉층의 30개 노드로 보낸다는 뜻.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리를 불러옴
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분.
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터를 불러들임.
Data_set = np.loadtxt("csv_data\ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 경과를 X와 Y로 구분하여 저장함.
X = Data_set[:, 0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정함(모델을 설정하고 실행하는 부분).
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))   # 노드 30 개 은닉층
model.add(Dense(1, activation="sigmoid"))               # 노드 1 개 출력층

# 딥러닝을 실행
model.compile(loss="binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
history = model.fit(X, Y, epochs=100, batch_size=10)
history2 = model.fit(X, Y, epochs=10, batch_size=10)
```

```python
history.history 
```

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], marker="*")
plt.plot(history.history["accuracy"], marker="o")
plt.show()  # 100번 에포크 수행시, 수행 횟수 증가에 따라 개선효과 체감.
```

```python
plt.plot(history2.history["loss"], marker="*")
plt.plot(history2.history["accuracy"], marker="o")
plt.show()  # 10번 수행시에도 100번 에포크 수행한 결과에 비해 크게 부족하지 않음.
            # 굳이 많이 할 필요 없다. history를 통해 적절한 수준 찾아낼 수 있음.
```

</div>
</details>

</br>

## 2. 입력층, 은닉층, 출력층
* 은닉층의 각 노드는 17개 입력 값에서 임의의 가중치를 가지고 각 노드로 전송되어 활성화 함수를 만남.
* 활성화 함수를 거친 결괏값이 출력층으로 전달됨.
* 다음에 나오는 activation 부분에 우리가 원하는 활성화 함수를 적어 주면 됨.
* 여기서는 ReLU 사용.
## 3. 모델 컴파일
* model.compile에서는 앞서 지정한 모델이 효과적으로 구현될 수 있게 여러 가지 환경 설정 및 컴파일.
* 오차 함수(loss), 최적화(optimizer), 모델 수행 결과 평가(metrics).
  * 오차 함수 : 여기서는 MSE(mean_squared_error)를 사용함.
## 4. 교차 엔트로피
* 교차 엔트로피는 주로 분류 문제에서 많이 사용되는데(ex.폐암 환자의 생존/사망 분류) 특별히 예측 값이 참과 거짓 둘 중 하나인 형식일 때는 binary_crossentropy(이항 교차 엔트로피)를 씀.
* 이를 실행하면 예측 정확도(accuracy)가 약간 향상되는 것을 알 수 있음.
## 5. 모델 실행
* 주어진 폐암 수술 환자의 생존 여부 데이터는 총 470명의 환자의 17개의 정보를 정리한 것.
  * 속성(또는 feature, 독립변수) : 17개의 환자 정보
  * 샘플 : 가로 한 줄에 해당하는 각 환자의 정보
  * 클래스(종속변수) : 생존 여부
  * 주어진 데이터에는 총 470개의 샘풀이 각각 17개씩의 속성을 가지고 있는 것

* epoch vs batch
  * epoch는 학습 프로세스가 모든 샘플에 대해 한 번 실행되는 것.
    * 위 코드에서 100 epoch이므로 각 샘플이 처음부터 끝까지 100번 재사용될 때까지 학습하라는 것.
  * batch는 샘플을 한 번에 몇 개씩 처리할지 정하는 부분.
    * batch_size = 10은 전체 470개 샘플을 10개씩 끊어서 집어넣으라는 뜻
    * batch_size가 너무 작으면 학습 속도가 느려지고, 너무 크면 결과값이 불안정해짐(컴퓨터의 메모리가 감당할 만큼의 batch_size를 설정해주는 것이 좋음).

다중 분류문제 해결
===
## 1. 다중 분류 문제(Multi classification Problem)
* 예를 들어 아이리스의 분류 문제 → 여러 품종의 분류 / 클래스가 3개
* 참(1)과 거짓(0)으로 해결하는 것이 아니라, 여러개 중에서 어떤 것이 답인지 예측하는 문제.
* 이항분류(binary classification)와는 접근 방식이 조금 다름
## 2. 상관도 그래프
* 코드를 통한 확인
  * 아이리스 데이터 확인

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 자동완성용 import
from keras.models import Sequential
from keras.layers import Dense

# 시드 부여
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species"].replace([0,1,2], iris.target_names, inplace=True)
df.head()
```

</div>
</details>

</br>

* 코드를 통한 확인
  * sns.pairplot() 함수를 써서 데이터 전체를 한번에 보는 상관도 그래프를 출력.
  * 그래프를 보니, 외관상 볼 때는 비슷해 보이던 꽃잎과 꽃받침의 크기와 너비가 품종별로 차이가 있음.
  * 속성별 연관성을 보여주는 상관도 그래프를 통해 프로젝트의 감을 잡고 프로그램 전략을 세울 수 있음.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 그래프로 확인
sns.pairplot(df, hue="species")
plt.show()
```

</div>
</details>

</br>

## 3. 원-핫 인코딩
* 원-핫 인코딩(one-hot encoding) : 여러 개의 Y 값을 0과 1로만 이루어진 형태로 바꿔주는 기법.
* Setosa, Virginica 등 데이터 안에 문자열이 포함되어 있다면 numpy보다는 pandas로 데이터를 불러와 X와 Y값을 구분하는 것이 좋음.
* Y 값이 문자열이라면, 클래스 이름을 숫자로 바꿔주어야 함. sklearn 라이브러리의 LabelEncoder() 함수를 이용할 수 있음.

* 활성화 함수를 적용하려면 Y 값이 0과 1로 이루어져 있어야 함.   
  이 조건을 만족시키려면 tf.keras.utils.categorical() 함수를 적용해야 함.   
  이에 따라 Y 값의 형태는 다음과 같이 변형 됨.
  * array([1 ,2, 3])가 다시 array([1,0,0], [0,1,0], [0,0,1])로 바뀜.
  * 원-핫 인코딩 완료.

## 4. 소프트맥스
* 모델 설정
* 소프트맥스 : 총합이 1인 형태로 바꿔서 계산해주는 함수.
* 합계가 1인 형태로 변환하면 큰 값이 두드러지게 나타나고 작은 값은 더 작아짐.   
  이 값이 교차 엔트로피를 지나 [1,0,0]으로 변하게 되면 우리가 원하는 원-핫 인코딩 값(하나만 1이고 나머지는 모두 0인 형태)으로 전환됨.


과적합 피하기
===
## 1. 데이터의 확인과 실행
* 실습: 초음파 광물 예측
  * 데이터 : sonar.csv
* 광석과 일반 돌을 가져다 놓고 음파 탐지기를 쏜 후 그 결과를 데이터로 정리함.
* 오차 역전파 알고리즘을 사용한 신경망이 광석과 돌을 구분하는 데 얼마나 효과적인지 조사.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 설정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv("./csv_data/sonar.csv", header=None)
df.head()
# 60 번 열이 클래스
```

```python
dataset = df.values
X = dataset[:, 0 : 60]
X = np.asarray(X).astype(np.float32)    # dtype이 object인데, float로 바꿔줌.

Y_obj = dataset[:, 60]

# 문자열 변환
e = LabelEncoder()
Y = e.fit_transform(Y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=5)

# 결과 출력
print(f"\n Accuracy : {model.evaluate(X, Y)[1]:.4f}")

# 정확도 100% : 과적합(overfitting)
```

</div>
</details>

</br>

## 2. 과적합 이해하기
* 과적합(overfitting) : 모델이 학습 데이터셋에서는 어느 정도 예측 정확도를 보이지만, 새로운 데이터에 적용하면 잘 맞지 않는 것.
* 과적합은 층이 너무 많을 때, 변수가 복잡할 때, 테스트셋과 학습셋이 중복될 때 생기기도 함.
* 딥러닝은 학습 단계에서 입력층, 은닉층, 출력층의 노드들에 상당히 많은 변수들이 투입됨.
* 딥러닝을 진행하는 동안 과적합에 빠지지 않게 늘 주의해야 함.
## 3. 학습셋과 테스트셋
* 과적합을 방지하려면,
  * 학습 데이터셋과 테스트 데이터셋을 구분한 다음 학습과 동시에 테스트를 병행하며 진행.
* 예를 들어, 데이터셋이 총 100개의 샘플로 이루어져 있다면 7:3으로 나눔.
  * 신경망을 만들어 70개의 샘플로 학습을 진행한 후 이 학습의 결과(모델)을 파일에 저장함.
  * 모델을 다른 셋에 적용할 경우 학습 단계에서 얻은 결과를 그대로 수행함.
  * 나머지 30개의 테스트 샘플로 실험해서 정확도를 살펴보면 학습이 잘 되었는지 파악 가능.
* 딥러닝 같은 알고리즘을 충분히 조절하여 가장 나은 모델이 만들어지면, 이를 실생활에 대입하여 활용하는 것이 머신러닝의 개발 순서.
  * 학습 데이터를 토대로 새로운 데이터를 예측하는 것이 목적이기 때문에 테스트 셋을 만들어 정확한 평가를 병행하는 것이 매우 중요함.
* 학습셋만 가지고 평가할 때, 층을 더하거나 에포크 값을 높이면 정확도가 계속해서 올라갈 수 있음.
* 학습이 깊어져서 학습셋의 정확도는 높아져도 테스트셋에서 효과가 없다면 과적합 발생으로 판단.
  * 학습을 진행해도 테스트 결과가 더 이사 좋아지지 않는 점에서 학습을 멈춰야 함. 이 때의 학습 정도가 가장 적절한 것으로 볼 수 있음.
* 검증셋(Validation sets) 개념
  * 실전에서는 더 정확한 테스트를 위해 테스트셋을 두 개로 나누어, 하나는 앞서 설명한 사용하고, 나머지 하나는 최종으로 만들어 낸 모델을 다시 한 번 테스트하는 용도로 사용하기도 함. 추가로 만들어낸 테스트셋을 검증셋이라고도 부름.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("./csv_data/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
X = np.asarray(X).astype(np.float32)    # dtype이 object인데, float로 바꿔줌.
Y_obj = dataset[:,60]

e = LabelEncoder()
Y = e.fit_transform(Y_obj)

# 학습 셋과 테스트 셋의 구분
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모형 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error",
            optimizer="adam",
            metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=130, batch_size=5)

# 테스트셋에 모델 적용
print(f"\n Test Accuracy : {model.evaluate(X_test, Y_test)[1]:.4f}")

# 정확도 0.8571이 나옴. 앞서 학습셋만으로 실행했을 때 나온 100%와 비교.
```

</div>
</details>

</br>

## 4. 모델 저장과 재사용
* 학습이 끝난 후 테스트해 본 결과가 만족스러울 때 이를 모델로 저장하여 새로운 데이터에 사용 가능.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">


```python
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("./csv_data/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
X = np.asarray(X).astype(np.float32)    # dtype이 object인데, float로 바꿔줌.
Y_obj = dataset[:,60]

e = LabelEncoder()
Y = e.fit_transform(Y_obj)

# 학습 셋과 테스트 셋의 구분
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모형 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error",
            optimizer="adam",
            metrics=["accuracy"])

history = model.fit(X_train, Y_train, epochs=130, batch_size=5)
model.save("my_model.h5")   # 모델을 컴퓨터에 저장
```

```python
del model       # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model("my_model.h5")   # 모델을 새로 불러옴
```

```python
print(f"\n Accuracy : {model.evaluate(X_test, Y_test)[1]:.4f}") # 불러온 모델로 테스트 실행
```

```python
# accuracy, error 그래프
import matplotlib.pyplot as plt

y_loss = history.history["loss"]
y_accu = history.history["accuracy"]
x_len = np.arange(len(y_loss))

plt.plot(x_len, y_loss, "o", c="red", markersize=3)
plt.plot(x_len, y_accu, "o", c="blue", markersize=3)

plt.show()
```

</div>
</details>

</br>

## 5. K-Fold 교차 검증
* 딥러닝 혹은 머신러닝 작업을 할 때 직면하는 문제 중 하나는 알고리즘을 충분히 테스트했더라도 데이터가 충분하지 않으면 좋은 결과를 내기가 어렵다는 것.
* 이러한 단점을 보완하고자 만든 방법이 K-Fold Cross Validation
* K-Fold : 데이터셋을 여러 개로 나누어 하나씩 테스트셋으로 사용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법.
* 위 방식을 통해 보유한 데이터의 100%를 테스트셋으로 사용할 수 있음.
* 코드를 통한 확인.
  * 10개의 파일을 쪼개 테스트하는 10차 교차 검증(n_fold=10).
  * StratifiedKFold() 함수 : 데이터를 원하는 숫자만큼 쪼개 각각 학습셋과 테스트셋으로 사용하는 함수.
  * 모델을 만들고 실행하는 부분을 for문으로 묶어 n_fold만큼 반복되게 함.
  * 정확도(Accuracy)를 매번 저장하여 한 번에 보여줄 수 있도록 accuracy 리스트를 만듦.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("./csv_data/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
X = np.asarray(X).astype(np.float32)    # dtype이 object인데, float로 바꿔줌.
Y_obj = dataset[:,60]

e = LabelEncoder()
Y = e.fit_transform(Y_obj)

# 10개의 파일로 쪼갬
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일, 실행
for train, test in skf.split(X, Y) :
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mean_squared_error",
                optimizer="adam",
                metrics=["accuracy"])
    history = model.fit(X_train, Y_train, epochs=130, batch_size=5)
    k_accuracy = f"{model.evaluate(X[test], Y[test])[1]:.4f}"
    accuracy.append(k_accuracy)

# 결과 출력
print(f"\n {n_fold} fold accuracy : {accuracy}")
```

```python
# accuracy, error 그래프
import matplotlib.pyplot as plt

y_loss = history.history["loss"]
y_accu = history.history["accuracy"]
x_len = np.arange(len(y_loss))

plt.plot(x_len, y_loss, "o", c="red", markersize=3)
plt.plot(x_len, y_accu, "o", c="blue", markersize=3)

plt.show()
```

</div>
</details>

</br>

이미지 인식을 위한 CNN
===
* MNIST 데이터를 통해 CNN의 이미지 인식 연습
    * MNIST 데이터셋은 70,000개의 글자 이미지에 각각 0부터 9까지 이름표를 붙인 데이터셋
## 1. 데이터 전처리

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# mnist 자료 로드
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets.mnist import load_data

# 자동완성용 라이브러리 로드
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Dense

# MNIST 데이터는 7만개의 이미지 중 6만개를 학습용으로, 1만개를 테스트용으로 미리 구분.
(X_train, y_class_train), (X_test, y_class_test) = load_data()
```

</div>
</details>

</br>

* matplotlib을 통한 시각적 확인.
    * $28\times28$ 픽셀, 0~255 까지 밝기 등급으로 표현.
    * 행렬로 이루어진 하나의 집합으로 변환됨.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import matplotlib.pyplot as plt
import numpy as np

first_image = X_train[0]
first_image = np.array(first_image, dtype=float)
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap="gray")
plt.show()
```

</div>
</details>

</br>

* 주어진 가로 28, 세로 28의 2차원 배열을 784개의 1차원 배열로 바꿔 줌.
* 이를 위해 reshape() 함수를 사용.
* keras는 데이터를 0에서 1사이의 값으로 변환한 다음 구동할 때 최적의 성능을 보임.
    * 따라서 현재 0~255 사이의 값으로 이루어진 값을 0~1 사이의 값으로 바꿔야 함.
* 데이터의 폭이 클 때 적절한 값으로 분산의 정도를 바꾸는 과정 : 데이터 정규화(normalization).
    * 정규화를 위해 astype()를 이용해 실수형으로 바꾼 뒤 255로 나눠줌(min-max).

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype("float64")
X_train = X_train / 255
```

```python
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255
```

</div>
</details>

</br>

* 레이블의 값은 5와 같이 이미지를 나타내는 카테고리형.
* 딥러닝의 분류 문제는 원-핫 인코딩 방식을 적용함.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
y_train = to_categorical(y_class_train, 10)
y_test = to_categorical(y_class_test, 10) # 뒤의 정수는 총 클래스 갯수 설정
```

</div>
</details>

</br>

## 2. 딥러닝 기본 프레임 만들기
* 입력 값(input_shape)이 784개, 은닉층이 512개 그리고 출력이 10개인 모델.
* 활성화 함수로 은닉층에서는 relu를, 출력층에서는 softmax를 사용.
* 딥러닝 실행 환경을 위해 오차 함수로 CategoricalCrossentropy, 최적화 함수로 adam을 사용.
* 모델의 실행에 앞서 모델의 성과를 저장하고 모델의 최적화 단계에서 학습을 자동 중단하게끔 설정.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 자동완성
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 모델 저장할 폴더 경로 및 파일명
model_dir = "./model/"
if not os.path.exists(model_dir) :
    os.mkdir(model_dir)

modelpath = "./model/{epoch:02d}_{val_loss:.4f}.hdf5"

# 체크 포인트 설정, 얼리 스탑 기준 지정
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1,
                                save_best_only=True)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

cols = 28*28
batch_size = 1000
model = Sequential()
model.add(Dense(units=512, input_shape=(cols, ), activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model.compile(loss="CategoricalCrossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# validation_data, callbacks 지정
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=batch_size,
                    verbose=0, callbacks=[early_stopping_callback, checkpointer])

print(f"\n Test Accuracy : {model.evaluate(X_test, y_test)[1]}")
```

</div>
</details>

</br>

* 학습셋에 대한 오차는 계속해서 줄어듦.
* 테스트셋의 과적합이 일어나기 전 학습을 끝낸 모습.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import matplotlib.pyplot as plt

y_vloss = history.history["val_loss"]

# 학습셋의 오차
y_loss = history.history["loss"]

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# 그래프에 그리드를 주고 레이블을 표시
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

</div>
</details>

</br>

## 3. 더 깊은 딥러닝
* 하나의 은닉층을 둔 아주 단순한 모델이지만 98%의 정확도
* 딥러닝은 이러한 기본 모델을 바탕으로, 프로젝트에 맞춰서 어떤 옵션을 더하고 어떤 층을 추가하느냐에 따라 성능이 좋아질 수 있음.
* CNN(Convolutional Neural Networ)를 추가
## 4. CNN
* 컨볼루션 신경망은 입력된 이미지에서 다시 한 번 특징을 추출하기 위해 마스크(필터, 윈도 또는 커널)를 도입하는 기법.
* 예를 입력된 이미지가 다음과 같은 값을 가지고 있다고 가정.
    * $\begin{bmatrix}1&0&1&0\\0&1&1&0\\0&0&1&1\\0&0&1&0\end{bmatrix}$
* 여기에 $2\times2$ 마스크를 준비함
    * 각 칸에는 가중치가 들어있음.
    * $\begin{bmatrix}\times1&\times0\\\times0&\times1\end{bmatrix}$
* 마스크를 이미지의 각 구간에 적용함.
    * 적용된 구간에 원래 있던 값에 가중치의 값을 곱해서 더해줌.
    * $(1\times1)+(0\times0)+(0\times0)+(1\times1)=2$
* 마스크를 한 칸씩 옮겨 모두 적용.
    * $\begin{bmatrix}2&1&1\\0&2&2\\0&1&1\end{bmatrix}$
* 이렇게 새롭게 만들어진 층을 컨볼루션(합성곱)이라고 부름.
* 컨볼루션을 만들면 입력 데이터로부터 더욱 정교한 특징을 추출할 수 있음.
* 마스크를 여러 개 만들 경우 여러 개의 컨볼루션이 만들어짐.
    * 예를 들어, $\begin{bmatrix}\times1&\times1\\\times0&\times0\end{bmatrix}\rarr\begin{bmatrix}1&1&1\\1&2&1\\0&1&2\end{bmatrix}$
### Keras의 CNN
* 케라스에서 컨볼루션 층을 추가하는 함수는 Conv2D()
```python
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
```
* 첫번째 인자 : 마스크를 몇 개 적용할 지 정함. 예시에서는 32개의 마스크를 적용.
* kernel_size : 마스크(커널)의 크기를 정함. kernel_size=(행, 열) 형식으로 정하며, 예시는 $3\times3$ 크기의 마스크를 사용하게끔 정함.
* input_shape : Dense 층과 마찬가지로 맨 처음 층에는 입력되는 값을 알려주어야 함. input_shape=(행,열, 색상 또는 흑백) 형식으로 정함. 만약 입력 이미지가 색상이면 3, 흑백이면 1.
* activation : 활성화 함수를 정의
```python
model.add(Conv2D(64, (3, 3), activation='relu'))
```
* 위와 같이 마스크 64개를 적용한 새로운 컨볼루션 층을 추가할 수 있음.
## 5. Max Pooling
* 컨볼루션 층을 통해 이미지 특징을 도출하였으나 그 결과가 여전히 크고 복잡하면 이를 다시 한 번 축소해야 함.
    * 이 과정을 풀링(pooling) 또는 서브 샘플링(sub sampling)이라고 함.
* 풀링 기법 중 가장 많이 사용되는 기법이 맥스 풀링(max pooling).
* 맥스 풀링은 정해진 구역 안에서 가장 큰 값만 다음 층으로 넘기고 나머지는 버림.
* 예를 들어 다음과 같은 이미지가 있을 때,
    * $\begin{bmatrix}1&0&1&0\\0&4&2&0\\0&1&6&1\\0&0&1&0\end{bmatrix}$
* 맥스 풀링을 적용하여 $(2\times2)$의 4개의 구간으로 나누어 가장 큰 값만을 추출.
    * $\begin{bmatrix}4&2\\1&6\end{bmatrix}$
### Keras의 MaxPooling
* 맥스 풀링을 통해 불필요한 정보를 간추릴 수 있음.
* 케라스에서는 MaxPooling2D()함수를 사용해서 다음과 같이 적용할 수 있음.
```python
model.add(MaxPooling2D(pool_size=2))
```
* 여기서 pool_size는 풀링 창의 크기를 정하는 것으로, 2로 정하면 전체 크기가 절반으로 줄어듦.
### 드롭아웃(drop out)과 플래튼(flatten)
#### 1. 드롭아웃
* 노드가 많아지거나 층이 많아진다고 해서 학습이 무조건 잘 되는 것이 아님.
    * 과적합 발생 가능.
* 과적합을 피하는 간단하지만 효과가 큰 기법이 드롭아웃 기법.
* 드롭아웃은 은닉층에 배치된 노드 중 일부를 임의로 꺼주는 것.
* 랜덤하게 노드를 끔으로써 학습 데이터에 지나치게 치우쳐서 학습되는 과적합을 방지할 수 있음.
* 케라스를 이용해 손쉽게 적용가능 / 25% 노드 끄는 코드
```python
model.add(Dropout(0.25))
```
#### 2. 플래튼
* 위의 과정을 다시 Dense() 함수를 이용해 기본층에 연결하려고 할때, 컨볼루션 층이나 맥스 풀링은 주어진 이미지를 2차원 배열인 채로 다룸.
* 이를 1차원으로 바꿔주는 함수가 Flatten() 함수.
```python
model.add(Flatten())
```
## 6. 컨볼루션 신경망 실행

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# mnist 자료 로드
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# 자동완성용 라이브러리 로드
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


# 데이터 로드
(X_train, y_class_train), (X_test, y_class_test) = load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_class_train)
y_test = to_categorical(y_class_test)


# 모델 저장할 폴더 경로 및 파일명
model_dir = "./model/"
if not os.path.exists(model_dir) :
    os.mkdir(model_dir)

modelpath = "./model/{epoch:02d}_{val_loss:.4f}.hdf5"

# 체크 포인트 설정, 얼리 스탑 기준 지정
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1,
                                save_best_only=True)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)


# 컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=2))
# 일반적으로 CNN에서는 드롭아웃 레이어를 Fully connected network 뒤에 놓지만,
# 상황에 따라서는 max pooling 계층 뒤에 놓기도 한다.
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation="softmax"))

model.compile(loss="CategoricalCrossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# validation_data, callbacks 지정
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=200,
                    verbose=1, callbacks=[early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print(f"\n Test Accuracy : {model.evaluate(X_test, y_test)[1]}")
```

```python
# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history["loss"]

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# 그래프에 그리드를 주고 레이블을 표시
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```
</div>
</details>

</br>

* 8번째 에포크에서 베스트 모델을 만들었고 18번째 에포크에서 학습이 자동 중단됨.
* 테스트 정확도가 98.05%에서 99.15%로 향상됨
* 위와 같이 학습의 진행에 따른 학습셋과 테스트셋의 오차 변화를 관측할 수 있음.

패션 mnist 실습
===

<details>
<summary>실습 코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("./csv_data/fashion-mnist_train.csv")
test_data = pd.read_csv("./csv_data/fashion-mnist_test.csv")
```

```python
# 데이터 형태 확인
train_data.head()
```

```python
# 데이터 분리 및 배열화
import numpy as np

y_train= np.array(train_data["label"])
train_data = train_data.drop(columns="label", axis=1)
X_train = np.array(train_data)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=121)
```

```python
# 데이터 분리 및 배열화
y_test = np.array(test_data["label"])
test_data = test_data.drop(columns="label", axis=1)
X_test = np.array(test_data)
```

```python
# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
```

```python
# 데이터 시각화를 위한 형태 변환
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
```

```python
# 시각화 하여 확인
import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap="gray")
plt.show()
```

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# 자동완성용 라이브러리 로드
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# seed 값 설정
seed = 121
np.random.seed(seed)
tf.random.set_seed(seed)


# 데이터 로드
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test/ 255


# 모델 저장할 폴더 경로 및 파일명
model_dir = "./fashion_model/"
if not os.path.exists(model_dir) :
    os.mkdir(model_dir)

modelpath = "./fashion_model/{epoch:02d}_{val_loss:.4f}.hdf5"

# 체크 포인트 설정, 얼리 스탑 기준 지정
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1,
                                save_best_only=True)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)


# 컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28,1), activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=2))
# 일반적으로 CNN에서는 드롭아웃 레이어를 Fully connected network 뒤에 놓지만,
# 상황에 따라서는 max pooling 계층 뒤에 놓기도 한다.
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation="softmax"))

model.compile(loss="CategoricalCrossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# validation_data, callbacks 지정
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=200,
                    verbose=1, callbacks=[early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print(f"\n Test Accuracy : {model.evaluate(X_test, y_test)[1]}")
```

```python
# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history["loss"]

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# 그래프에 그리드를 주고 레이블을 표시
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

</div>
</details>

</br>

RNN(Recurrent Neural Network)
===
시퀀스 배열로 다루는 순환 신경망
---
#### 1. 순서형 자료와 순환 신경망
* 인공지능이 문장을 듣고 이해한다는 것은 많은 문장을 이미 학습해 놓았다는 것.
    * 하지만 문장을 학습하는 것은 이전까지의 과정과 조금 다름.
    * 문장은 여러 개의 단어로 이루어져 있으므로, 그 의미를 전달하려면 각 단어가 정해진 순서대로 입력되어야 하기 때문임.
* 즉, 여러 데이터가 순서와 관계없이 입력되던 것과는 다르게, 이번에는 과거에 입력된 데이터와 나중에 입력된 데이터 사이의 관계를 고려해야 하는 문제가 생기는 것.
    * 이를 해결하기 위해 순환 신경망(RNN) 방법이 고안됨.
    * 순환 신경망은 여러 개의 데이터가 순서대로 입력되었을 때 앞서 입력받은 데이터를 잠시 기억해 놓는 방법.
    * 그리고 기억된 데이터가 얼마나 중요한지를 판단하여 별도의 가중치를 줘서 다음 데이터로 넘어감.
    * 모든 입력 값에 이 작업을 순서대로 실행하므로 다음 층으로 넘어가기 전에 같은 층을 맴도는 것처럼 보임.
        * 이렇게 같은 층 안에서 맴도는 성질 때문에 순환 신경망이라고 부름.
* 예를 들어 인공지능 비서에게 "오늘 주가가 몇이야?" 라고 묻는다면,
    * 모형의 순환 부분에서 단어를 하나 처리할 때마다 단어마다 기억하여 다음 입력 값의 출력을 결정함.
    * [입력1:오늘] → [모형] → [결과1:['오늘'에 대한 결과]&[기억1]]   
      [입력2:주가가] → [모형]+[기억1] → [결과2:['주가가'에 대한 결과]&[기억2]]    
      [입력3:몇이야?] → [모형]+[기억2] → [결과3:['몇이야'에 대한 결과]]
    * 순환이 되는 가운데 앞서 나온 입력에 대한 결과가 뒤에 나오는 입력 값에 영향을 줌.
    * 따라서 [입력2]의 값이 똑같이 '주가가'라 하더라도, [입력1]이 '오늘'인지 '어제'인지에 따라 계산이 달라지는 것.
$$h^{(t)}=\phi_h(W_{xh}x^{(t)}+W_{hh}h^{(t-1)}+b_h)\qquad\cdots\quad(1)$$
$$h^{(t)}=\phi_h(W_{h}[x^{(t)};h^{(t-1)}]+b_h)\;\qquad\qquad\cdots\quad(2)$$
$$o^{(t)}=\phi_o(W_{ho}h^{(t)}+b_o)\;\qquad\cdots\quad(final\ output)$$

#### 2. LSTM(Long Short Term Memory) 방법
* LSTM은 한 층 안에서 반복을 많이 해야 하는 RNN의 특성상 일반 신경망 보다 기울기 소실 문제가 더 많이 발생하고 이를 해결하기 어렵다는 단점을 보완한 방법.
* 즉, 반복되기 직전에 다음 층으로 기억된 값을 넘길지 안 넘길지를 관리하는 단계를 하나 더 추가하는 것.
#### 3. RNN 방식의 장점과 처리 예시
* RNN 방식의 장점은 입력값과 출력값을 어떻게 설정하느냐에 따라 다음과 같이 여러가지 상황에서 이를 적용할 수 있다는 것.
1. 다수입력 단일 출력
    * 예 : 문장을 읽고 뜻을 파악할 때 활용
    * [입력 및 처리:[["밥은"] → ["먹고"] → ["다니니?"]]] → [결과:[안부 인사]]
2. 단일 입력 다수 출력
    * 예 : 사진의 캡션을 만들 때 활용 / 음악 캡션 분석
    * [입력:["Wet Sand"]] → [처리 및 결과:["RHCP"],["Rock"],["Rythm"]]
3. 다수 입력 다수 출력
    * 예 : 문장을 번역할 때 활용
    * [입력:["예"] → ["그게"] → ["다에요"]] → [결과:["Yes"] → ["that's"] → ["all"]]
#### 4. 케라스의 제공 데이터
* 케라스는 딥러닝 학습에 필요한 데이터를 쉽게 내려받을 수 있게 load_data() 함수를 제공.
* MNIST 데이터셋 외에도 RNN 학습에 적절한 텍스트 대용량 데이터를 제공.
* 케라스가 제공하는 '로이터 뉴스 카테고리 분류'와 'IMDB 영화 리뷰'를 활용 가능.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(3)

rnn_layers = keras.layers.SimpleRNN(units=2, use_bias=True, return_sequences=True)
rnn_layers.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layers.weights

print(f"w_xh의 크기: {w_xh}")
print(f"w_oo의 크기: {w_oo}")
print(f"b_h의 크기: {b_h}")
```

</div>
</details>

</br>

### 1. LSTM을 이용한 로이터 뉴스 카테고리 분류하기
#### 1. 문장의 의미 파악의 정의
* 입력된 문장의 의미를 파악하는 것은 곧 모든 단어를 종합하여 하나의 카테고리로 분류하는 작업이라고 할 수 있음.
* 예를 들어 "안녕, 오늘 날씨가 참 좋네."라는 말은 '인사'카테고리에 분류해야 함.
* 그리고 다음의 예시와 같이 조금 더 길고 전문적인 말도 정확하게 분류해야 함.
    * 중부 지방은 대체로 맑겠으나, 남부 지방은 구름이 많겠습니다. → 날씨
    * 올 초부터 유동성의 힘으로 주가가 일정하게 상승했습니다. → 주식 등
#### 2. 실습
##### 1. 데이터 로드와 해석
* 실습 내용은 위와 같이 긴 텍스트를 읽고 이 데이터가 어떤 의미를 지니는지 카테고리로 분류하는 연습.
* 실습을 위해 로이터 뉴스 데이터를 사용.
* 로이터 뉴스 데이터는, 총 11,258개의 뉴스 기사가 46개의 카테고리로 나누어진 대용량 텍스트 데이터.
* 데이터는 케라스를 통해 다음과 같이 불러올 수 있음.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 로이터 뉴스 데이터셋 불러오기
from tensorflow.keras.datasets import reuters

import numpy as np

# 자동 완성용 라이브러리
from keras.datasets import reuters

# 불러온 데이터를 학습셋과 테스트셋으로 나눔
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
```

```python
# 데이터 확인
category = np.max(y_train) +1
print(f"카테고리 갯수 : {category}")
print(f"학습용 뉴스 기사 수 :{len(X_train)}")
print(f"테스트용 뉴스 기사 수 :{len(X_test)}")
print(X_train[0])
```

</div>
</details>

</br>

* print(X_train[0])에서 기사가 아닌 [1,2,2,8,43,...]과 같은 숫자 배열이 나옴.
* 이는 데이터 안에서 해당 단어가 몇 번이나 나타나는지 세어 빈도에 따라 번호를 붙인 것.
    * 예를 들어, 3이라고 하면 세 번째로 빈도가 높은 단어라는 뜻임.
* 이러한 작업을 위해서 tokenizer() 같은 함수를 이용하는데, 본 데이터는 이미 토큰화 작업을 마친 데이터를 제공해줌.
* 이때, 기사 안의 단어 중에서는 거의 사용되지 않는 것들도 있으므로, 모든 단어를 다 사용하는 것은 비효율적. → 빈도가 높은 단어만 불러와 사용.
* __num_word__ 인자에 따라 빈도수 상위 1~1000에 해당하는 단어만 선택해서 불러오는 것.
##### 2. 전처리
* 또 하나 주의해야 할 점은 각 기사의 단어 수가 제각각 다르므로 단어의 숫자를 맞춰줘야 함.
* 이때는 데이터 전처리 함수 sequendce()를 사용
* maxlen = 100은 단어 수를 100개로 맞추라는 뜻.
    * 입력된 기사의 단어 수가 100보다 크면 100개째 단어만 선택하고 나머지는 버림.
    * 입력된 기사의 단어 수가 100 에서 모자랄 때는 모자라는 부분을 모두 0 으로 채움.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU
import tensorflow as tf

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# seed 설정
np.random.seed(3)
tf.random.set_seed(3)


# 데이터 전처리
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)

# y 데이터에 원-핫 인코딩 처리를 하여 데이터 전처리 과정을 마침
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모형 설정
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation="tanh"))
# GRU 알고리즘 활용할 경우 add 뒤에 단어만 바꿔주면 됨.
# model.add(GRU(100, activation="tanh"))
model.add(Dense(46, activation="softmax"))
```

</div>
</details>

</br>

* Embedding 층과 LSTM 층이 새로 추가됨.
    * Embedding 층은 데이터 전처리 과정을 통해 입력된 값을 받아 다음층이 알아들을 수 있는 형태로 변환하는 역할을 함.
    * Embedding('불러온 단어의 총개수', '기사당 단어 수')형식으로 사용하며, 모델 설정 부분의 맨 처음에 있어야 함.
* LSTM은 앞서 설명과 같이, RNN에서 기억 값에 대한 가중치를 제어함.
    * LSTM('기사당 단어 수', '기타 옵션')의 형태로 적용됨
    * LSTM의 활성화 함수로는 Tanh 사용.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 모델의 컴파일
model.compile(loss="categorical_crossentropy",
            optimizer='adam',
            metrics=["accuracy"])

# 모델의 실행
history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_test, y_test))

# 테스트 정확도 출력
print(f"\n Test Accuracy : {model.evaluate(X_test, y_test)[1]:.4f}")
```

```python
import matplotlib.pyplot as plt

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history["loss"]

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# 그래프에 그리드를 주고 레이블을 표시
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

</div>
</details>

</br>

* 테스트셋에 대한 정확도가 0.7119를 보이고 있음.
    * GRU는 0.6972.
* 테스트 오차가 상승하기 전까지의 학습과정이 과적합 직전의 최적 학습 시간.

RNN(Recurrent Neural Network)
===
시퀀스 배열로 다루는 순환 신경망
---
### 2. LSTM과 CNN의 조합을 이용한 영화 리뷰 분류하기
* 인터넷 영화 데이터베이스(IMDb)에 있는 영화에 대한 긍정, 부정 평가를 활용해 분석 수행.
* 데이터 전처리 과정은 로이터 뉴스 데이터와 거의 같음.
* 다만 클래스가 긍정 또는 부정 두 가지뿐이라 원-핫 인코딩 과정이 없음.
#### 1. 데이터의 전처리와 모델 설정

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 라이브러리 로드
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 자동완성용
from keras.datasets.imdb import load_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Activation
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
```

```python
# seed 값 설정
np.random.seed(0)
tf.random.set_seed(0)

# 학습셋과 테스트셋 지정
(X_train, y_train), (X_test, y_test) = load_data(num_words=5000)

# 데이터 전처리 / maxlen : 길이 맞춰주는 역할
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)
```

</div>
</details>

</br>

* 모델을 다음과 같이 설정
* 마지막에 model.summary() 함수를 넣으면 현재 설정된 모델의 구조를 한 눈에 볼 수 있음.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 모델의 설정
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
# strides : 컨볼루션 커널 이동 간격 / 보폭
model.add(Conv1D(64, 5, padding='valid', activation="relu", strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
# 새로운 방식
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()
```

</div>
</details>

</br>

#### 2. 1차원 문자열 데이터와 2차원 이미지의 차이
* 앞서 Conv2D와 MaxPooling2D는 앞서 MNIST 손글씨 인식에서 2차원 배열의 이미지를 다루기 위해 사용함.
* 하지만 현재 데이터는 1차원의 문자열 데이터이기 때문에 Conv1D와 MaxPooling1D로 1차원 이동하는 컨볼루션 방식 사용.
* 1차원 컨볼루션 예시(kernel_size=3, stride=1) : 
$$\begin{bmatrix}1&4&3&2\end{bmatrix}\qquad\cdots(input\ data)$$
$$\begin{bmatrix}\times1&\times0&\times1\end{bmatrix}\qquad\cdots(mask\ or\ kernel)$$
$$1\times1+4\times0+3\times1 = 1+3=4$$
$$4\times1+3\times0+2\times1 = 4+2=6$$
$$\begin{bmatrix}4&6\end{bmatrix}\qquad\cdots(output\ data)$$
* MaxPooling1D 역시 1차원 배열상의 정해진 구간에서 가장 큰 값을 다음 층으로 넘기고 나머지는 버림.
* 1차원 MaxPooling 예시 (pool_size=2) :
$$\begin{bmatrix}1&4&3&2\end{bmatrix}\qquad\cdots(input\ data)$$
$$\begin{bmatrix}1&4\end{bmatrix}\quad\begin{bmatrix}3&2\end{bmatrix}\qquad\cdots(pool\ apply)$$
$$\begin{bmatrix}4&3\end{bmatrix}\qquad\cdots(output\ data)$$
#### 3. 모델 실행
* 번외 : 텐서보드 사용
    * os 모듈: 운영 체제와 상호 작용을 할 때, 사용하는 Python 기본 모듈로, Python 안에서 끝내는 것이 아닌 자신의 컴퓨터와 어떠한 작업을 하고자 한다면, 해당 모듈을 필수로 사용하게 된다.
    * os.curdir: 현재 디렉터리를 가지고 온다.
    * os.path.join('C:\Tmp', 'a', 'b'): 주어진 경로들을 합쳐서 하나로 만든다. 
    *   dir_name은 Tensorboard가 바라 볼 디렉터리의 이름이다.
    * 코드 실행 시, 생성된 로그 데이터가 섞이지 않도록, dir_name 디렉터리 아래에 현재 날짜와 시간으로 하위 디렉터리의 경로를 만들어낸다.  


<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import os
import datetime

# Log data를 저장할 디렉터리 이름 설정
dir_name = "Learning_log"

# main 디렉터리와 sub 디렉터리 생성 함수
def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)
```

```python
# 콜백함수 - 텐서보드
TB_log_dir = make_Tensorboard_dir(dir_name)
TensorB = TensorBoard(log_dir = TB_log_dir)

# 모델 저장 조건 설정
model_dir_name = "model"

def make_Model_dir(model_dir_name):
    root_logdir = os.path.join(os.curdir, model_dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_dir = os.path.join(root_logdir, sub_dir_name)
    file_name = "{epoch:02d}_{val_loss:.4f}.hdf5"
    return os.path.join(folder_dir, file_name)

checkpointer = ModelCheckpoint(filepath=make_Model_dir(model_dir_name), monitor="val_loss", verbose=1, save_best_only=True)

# 콜백함수 - 얼리스탑
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)

# 모델의 컴파일
model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])


# 모델의 실행
history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_test, y_test),
                    verbose=1, callbacks=[early_stopping_callback, TensorB, checkpointer])

# 테스트 정확도 출력
print(f"\n Test Accuracy:{model.evaluate(X_test,y_test)[1]:.4f}")
```

```python
# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history["loss"]

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# 그래프에 그리드를 주고 레이블을 표시
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```

</div>
</details>

</br>

#### 4. 토큰화 이전의 IMDb 데이터를 활용한 실습

<details>
<summary>실습 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 데이터 불러오기
origin = pd.read_csv("./csv_data/IMDB Dataset.csv")
origin.head()

# html 줄바꿈 삭제
review = origin.review.str.replace("<br />", "")
sentiment = origin.sentiment

# 확인
review
```

```python
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
label = lb.fit_transform(sentiment)
```

```python
# 스탑워드 라이브러리
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import re

nltk.download("stopwords")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
```

```python
def preprocess(review, stem=stemmer):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(review).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
```

```python
review_pr = review.apply(preprocess)
review_pr
```

```python
# 토큰화 및 전처리 라이브러리
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 자동완성
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_pr)

review_tk = tokenizer.texts_to_sequences(review_pr)
review_test = sequence.pad_sequences(review_tk, maxlen=100)
review_test
```

```python
# 3.에서 만든 모델 로드
model_t = load_model("./model/20220512-140010/02_0.3241.hdf5")
pred = model_t.predict(review_test)
pred
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(label, pred.round())
```

</div>
</details>

</br>

추천시스템
===
## 1. 추천시스템이란?
### 1. 추천 시스템의 추천 방식
* 사용자(User)와 상품(Item)으로 구성된 시스템.
* 특정 사용자(User)가 좋아할 상품(Item)을 추천.
* 비슷한 상품(Item)을 좋아할 사용자(User)를 추천.
* Item이든 User든 관심을 가질만한 정보(information)를 추천.
### 2. 추천 시스템의 등장 배경과 종류
* 서비스의 성장과 정보의 다양화.
* 인터넷에서 찾을 수 있는 정보가 매우 많음
* 인터넷에서 정보를 찾는데 시간이 매우 오래 걸림.
* 사용자가 정보를 수집하고 찾는 시간을 줄여주는 것이 목적.

* |검색서비스|추천서비스|
  |---|---|
  |Pull Information|Push Information|
  |사용자가 요구한 후 작동|사용자가 요구하기 전 작동|
  |사용자 스스로 원하는 바를 알고 있음|사용자 스스로 원하는 바를 정확히 알지 못함|
### 3. 사용자(User)와 상품(Item)
* 사용자와 아이템 사이의 관계를 분석하고 연관관계를 찾음.
* 해당 연관 관계를 점수화함.
* 사용자의 정보와 아이템의 정보를 활용함.
  * 사용자 정보
    * 사용자 고유 정보(나이, 성별, 지역 등)
    * 사용자 로그 분석(행동 패턴 등)
  * 아이템 정보
    * 아이템 고유 정보(가격, 색상, 내용 등)
#### 1. 사용자 프로필(User Profile)
* 사용자 또는 사용자 그룹을 분석 가능한 요소로 프로파일링(Profiling).
* 사용자를 구분할 수 있는 정보를 활용.
  * 사용자 ID : 나이, 성별, 지역, 학력 등 개인 신상정보
  * 쿠키(Cookie)
  * 인터넷 주소
  * 웹 페이지 방문 기록, 클릭 패턴 등 사용자 행동 정보
* 사용자 정보를 수집하기 위한 방법.
  * 직접적인(Explicit)방법 : 설문조사, 평가, 피드백 등
  * 간접적인(Implicit)방법 : 웹페이지 체류 시간, 클릭 패턴, 검색 로그 등
* 개인별 추천 또는 사용자 그룹별 추천 가능
## 2. 추천점수란?
* 분석된 사용자와 아이템 정보를 바탕으로 추천점수 계산.
* 사용자 또는 아이템 프로필에서 어떤 정보를 사용할지에 따라 추천 알고리즘 결정.
* 사용자 또는 아이템을 추천하기 위해 각각의 아이템 또는 사용자에 대한 정량화된 기준 필요.
* 추천 알고리즘의 **목적**은 **점수화(Scoring)** 하는 것.
## 3. 추천시스템을 위한 다양한 데이터
* 데이터
  * 사용자가 어떤 상품을 구매했는가?
  * 사용자가 어떤 제품을 검색 했는가?
  * 사용자가 무엇을 클릭했는가?
  * 사용자가 평가한 영화 평점은?
* 추천 엔진을 통과한 결과
  * 고객을 위한 최신 상품.
  * 해당 상품을 선택한 다른 사람들이 좋아하는 상품들.
  * 이런 상품을 좋아하신다면, 아래의 상품은 어떠신가요?
## 4. 추천시스템의 묘미 : 아하 모멘트
* 추천 시스템의 묘미는 사용자 자신도 좋아하는지 모르고 있었던 취향을 발견하는 것.   
추천 시스템의 신뢰가 높아지면서 사용자는 추천 아이템을 더 많이 선택하게 되고, 이로 인해 더 많은 데이터가 추천 시스템에 축적되면서 추천이 정확해지고 다양해 짐.
### 넷플릭스의 사례 : 추천시스템의 비지니스적 활용
* 자사가 생성한 콘텐츠 위주로 추천하는 경향.
## 5. 추천시스템 방식
1. 콘텐츠 기반 필터링(Content Based Filtering)
2. 협업 필터링(Collaborative Filtering)
* 추천 시스템은 이들 방식중 1가지를 선택하거나 이들을 결합하여 hybrid 방식으로 사용.
(예 : Content Based + Collaborative Filtering)
### 1. 컨텐츠기반 추천시스템(Contents-based Recommender System)
* 사용자가 과거에 좋아했던 아이템을 파악하고, 그 아이템과 비슷한 아이템을 추천.
* 예시) 스파이더맨에 4.5점 평점을 부여한 유저는 타이타닉보다 캡틴 마블을 더 좋아할 것.
  1. 유저가 과거에 접한 아이템이면서 만족한 아이템
  2. 유저가 좋아했던 아이템 중 일부 또는 전체와 비슷한 아이템 선정
  3. 선정한 아이템을 유저에게 추천
### 2. 협업 필터링(Collaborative Filtering)
* 비슷한 성향 또는 취향을 가진 다른 유저가 좋아한 아이템을 현재 유저에게 추천.
* 간단하면서 수준 높은 정확도를 보임.
* 예시) 스파이더맨에 4.5점을 준 2명의 유저 중, 유저 A가 과거에 좋아했던 캡틴 마블을 유저 B에게 추천.
  1. 유저 A와 유저 B 모두 같은 아이템에 대해 비슷한 평가를 했음.
  2. 이 때, 유저 A는 다른 아이템에도 비슷한 호감을 나타냄.
  3. 따라서 유저 A, B의 성향은 비슷할 것이므로, 다른 아이템을 유저 B에게 추천함.
### 3. 복합 추천 시스템(Hybrid Recommender System)
* Content-based와 Collaborative Filtering의 장, 단점을 상호보완
* Collaborative Filtering은 새로운 아이템에 대한 추천이 부족함.
* 여기에 대해, content-based 기법이 cold-start 문제에 도움을 줄 수 있음.
## 5. CBF 실습 : 장르 유사도 기반 영화 추천시스템
### 1. 콘텐츠 기반 필터링 구현 프로세스
1. 콘텐츠에 대한 여러 텍스트 정보들을 피처 벡터화.
2. 코사인 유사도로 콘텐츠별 유사도 계산.
3. 콘텐츠 별로 가중평점을 계산.
4. 유사도가 높은 콘텐츠 중에 평점이 좋은 콘텐츠 순으로 추천.
### 2. 데이터 로드

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

movies = pd.read_csv("./csv_data/tmdb_5000_movies.csv")

print(movies.shape)
movies.head()
```

</div>
</details>

</br>

### 3. 장르 데이터 전처리

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 장르 데이터 전처리
from ast import literal_eval

movies["genres"] = movies["genres"].apply(literal_eval)
movies["keywords"] = movies["keywords"].apply(literal_eval)
type(movies["genres"][0])
```

```python
movies["genres"] = movies["genres"].apply(lambda x : [y["name"] for y in x])
movies["keywords"] = movies["keywords"].apply(lambda x : [y["name"] for y in x])
movies[["genres", "keywords"]][:1]
# 우리에게 필요한 장르명만 뽑아옴.
```

</div>
</details>

</br>

### 4. 장르 피처 벡터화 - 영화간 유사도 계산
* 장르 문자열을 Count 기반 피처 벡터화 후에 코사인 유사도로 각 영화를 비교

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.feature_extraction.text import CountVectorizer

# CounterVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환.
movies["genres_literal"] = movies["genres"].apply(lambda x : (" ").join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies["genres_literal"])
print(genre_mat.shape)
movies["genres_literal"].head()
# bingram으로 피처 수 276개로 증가
```

```python
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:2])
```

```python
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])
# 첫번째 영화와 유사도가 높은 영화 순서
```

</div>
</details>

</br>

### 5. 특정 영화와 장르 유사도가 높은 영화 추천

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 특정 영화와 장르 유사도가 높은 영화를 반환하는 함수
def find_sim_movie(df, sorted_ind, title_name, top_n=10) :

    # 인자로 입력된 movies DataFrame에서 'title'컬럼의 값이 입력된 title_name 값인 DataFrame 추출
    title_moive = df[df['title'] == title_name]

    # title_named를 가진 DataFrame의 index 객체를 ndarray로 변환하고
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n개의 index 추출
    title_index = title_moive.index.values
    siamilar_indexes = sorted_ind[title_index, :(top_n)]

    # 추출된 top_n index들 출력. top_n index는 2차원 데이터임.
    # DataFrame에서 index로 사용하기 위해서 1차원 array로 변경
    print(siamilar_indexes)
    siamilar_indexes = siamilar_indexes.reshape(-1)

    return df.iloc[siamilar_indexes]
```

```python
similar_movies = find_sim_movie(movies, genre_sim_sorted_ind, "The Godfather", 10)
similar_movies[["title", "vote_average"]]
# 영화 대부와 장르 유사도가 높은 순서

# 문제는 평가횟수가 현저히 적은 영화들이 추천되는 경우도 있음. low quality 추천 문제.
# 우리가 전혀 모르는 영화를 추천받는 것은 엉뚱한 추천 결과를 낳을 수 있음.
# → 평가횟수를 반영한 추천 시스템이 필요
```

</div>
</details>

</br>

### 6. 가중평점 반영항 영화 추천
*  가중평점(평점&평가횟수) 반영한 영화 추천
* 가중평점(Weighted Rating) :
*    (v/(v+m))*R + (m/v+m)*C
*  v : 영화별 평점을 투표한 횟수(vote_count), 투표 횟수가 많은 영화에 가중치 부여
*  m : 평점을 부여하기 위한 최소 투표 횟수, 여기서는 투표수 상위 60%
*  R : 개별 영화에 대한 평균 평점(vote_average)
*  C : 전체 영화에 대한 평균 평점(movie["vote_average"].mean())
*  C, m 은 고정값
*  v, R 은 영화마다 변동값
* 투표 횟수가 많으면 가중치가 붙음. 최종적으로,
    1. __장르__ 가 유사한 영화 중
    2. __가중평점__ 이 높은 영화가 추천되게 됨.

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 상위 60%에 해당하는 vote_count를 최소 투표 횟수인 m으로 지정
C = movies["vote_average"].mean()   # 약 6점
m = movies["vote_count"].quantile(0.6)  # 370회
print(f"{C:.3f}, {m:.3f}")
```

```python
# 가중평점을 계산하는 함수
def weighted_vote_average(record) :
    v = record["vote_count"]
    R = record["vote_average"]

    return ((v/(v+m))*R + (m/(m+v))*C)  # 가중평균을 return
```

```python
# 기존 데이터에 가중평점 칼럼 추가
movies["weighted_vote"] = movies.apply(weighted_vote_average, axis=1)
```

```python
# 추천 ver2. 먼저 장르 유사성이 높은 영화 20개 선정 후, 가중평점순 10개 선정
def find_sim_movie_ver2(df, sorted_ind, title_name, top_n=10) :
    title_moive = df[df['title'] == title_name]
    title_index = title_moive.index.values

    # top_n의 2배에 해당하는 장르 유사성이 높은 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]
```

```python
# 영화 대부에 대해 장르 유사성, 가중평점 반영한 추천 영화 10개를 뽑음.
similar_movies = find_sim_movie_ver2(movies, genre_sim_sorted_ind, "The Godfather", 10)
similar_movies[["title", "vote_average", "weighted_vote", "genres", "vote_count"]]
# 영화 대부는 장르가 Drama, Crime 두개.
```

```python
# 응용 : Spider-Man3 좋아하는 사람 기준으로 장르가 유사한 영화를 추천.
similar_movies = find_sim_movie_ver2(movies, genre_sim_sorted_ind, "Spider-Man 3", 10)
similar_movies[["title", "vote_average", "weighted_vote", "genres", "vote_count"]]
```

```python
import pandas as pd
rating_df = pd.read_csv("./csv_data/ratings.csv")
movie_df = pd.read_csv("./csv_data/movies.csv").iloc[:10000,:]
```

```python
# 장르 문자열을 Count 기반 피처 벡터화 후에 코사인 유사도로 각 영화를 비교
from sklearn.feature_extraction.text import CountVectorizer

# CounterVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환.
movie_df["genres_literal"] = movie_df["genres"].apply(lambda x : x.replace("|", " "))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movie_df["genres_literal"])
print(genre_mat.shape)
movie_df["genres_literal"].head()
# 피처 193로 증가
```

```python
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])
# 첫번째 영화와 유사도가 높은 영화 순서
```

```python
# 특정 영화와 장르 유사도가 높은 영화를 반환하는 함수
def find_sim_movie(df, sorted_ind, title_name, top_n=10) :

    # 인자로 입력된 movies DataFrame에서 'title'컬럼의 값이 입력된 title_name 값인 DataFrame 추출
    title_moive = df[df['title'] == title_name]

    # title_named를 가진 DataFrame의 index 객체를 ndarray로 변환하고
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n개의 index 추출
    title_index = title_moive.index.values
    similar_indexes = sorted_ind[title_index, :(top_n+1)]
    similar_indexes = similar_indexes[similar_indexes != title_index]


    # 추출된 top_n index들 출력. top_n index는 2차원 데이터임.
    # DataFrame에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]
```

```python
similar_movies = find_sim_movie(movie_df, genre_sim_sorted_ind, "Toy Story (1995)", 10)
similar_movies[["title", "genres"]]
```

</div>
</details>

</br>