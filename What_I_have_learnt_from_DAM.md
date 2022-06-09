## iris_classification
<details>
<summary>데이터 세트 로딩 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
iris
iris.keys()
# iris.data 는 iris 데이터 세트에서 피처(feature)만으로 된 데이터를 배열로 가지고 있음.
iris_data = iris.data
# iris.target 은 붓꽃 데이터 세트에서 레이블 데이터(0,1,2)를 배열로 가지고 있음
iris_label = iris.target
print(f"iris target 값 : {iris_label}")
print(f"iris taget 명 : {iris.target_names}")
```
</div>
</details>

</br>

### pandas.DataFrame
* class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

</br>
  
<details>
<summary>데이터 프레임 변환 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df["label"] = iris.target
iris_df.head(3)
3. sklearn.model_selection.train_test_split
* sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# 트레이닝 데이터와 테스트 데이터 분리  /  (독립변수, 종속변수, 테스트 셋 사이즈, 시드)
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2, random_state=11)
```
</div>
</details>

</br>

### sklearn.tree.DecisionTreeClassifier
* classifier 는 0,1,2 등 이산분류, regressor 는 연속 판단
* class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

<details>
<summary>디시전 트리와 로짓분류 코드 펼치기/접기</summary>
<div markdown="1">

```python
# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)
# 로짓 분류 적용
dt_lr = LogisticRegression()
# 학습 수행
dt_clf.fit(X_train, y_train)
# 로짓 분류 학습 수행
dt_lr.fit(X_train, y_train)
# 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred = dt_clf.predict(X_test)   #predict → 테스트 셋 독립변수 집어넣어서 값 나옴
# 로짓 분류 학습 결과 테스트
pred_lr = dt_lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"예측 정확도 : {accuracy_score(y_test, pred) : .4f}") #y 테스트 값과 predict 수행한 값(0,1,2) 비교, 정확도 측정
# 소수점 n자리까지 float 출력 f"~ {변수 : .#f}"
# 로짓 회귀로 분류
print(f"예측 정확도 : {accuracy_score(y_test, pred_lr) : .4f}")
len(y_test) # 결과 값 / 테스트 셋 종속변수 갯수
X_test[0]   # X_test의 첫번째 피쳐값
pred    # X_test를 통한 예측 결과
pred_lr
y_test  # y_test의 실제 값
```
</div>
</details>

</br>

## Boston 집 값 분석

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
bst = load_boston()
bst_data = bst.data

bst_df = pd.DataFrame(data=bst_data, columns=bst.feature_names)
bst_df["price"] = bst.target

K_train, K_test, z_train, z_test = train_test_split(bst_data, bst.target,
                                            test_size=0.2,random_state=11)

dt_linR = LinearRegression()
dt_linR.fit(K_train, z_train)

pred_linR = dt_linR.predict(K_test)
dt_linR.score(K_test, z_test)   # 학습시킨 회귀모형에 K_test 배열 집어넣은 결과와 z_test 비교 R² 도출
r2_score(z_test, pred_linR)     # metrics 메소드의 r2_score 함수도 위와 같은 값 반환
mse = mean_squared_error(z_test, pred_linR)

rmse = np.sqrt(mse)     # scikit learn이 rmse 지원하지 않으므로, mse 구한 후 제곱근 값 구해야 함.-
rmse

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(pred_linR, z_test, color="black")
plt.plot(pred_linR, pred_linR, color="orange", linewidth=3)
plt.show()
```

</div>
</details>

</br>

## KFold 연습

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
iris = load_iris()

features = iris.data    # 독립변수
label = iris.target     # 종속변수
dt_clf = DecisionTreeClassifier(random_state=156)   # 디시젼트리 분류기 모형
kFold = KFold(n_splits=5)   # 5개 폴드
cv_accuracy = []            # 정확도 측정치 담을 리스트
print(f"붓꽃 데이터 세트 크기 : {features.shape[0]}")   # 행 개수
n_iter = 0      # 반복 횟수 카운터

for train_index, test_index in kFold.split(features) :
    # KFold 클래스의 split 메소드는 데이터를 받으면 값을 2개 반환
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시 마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0] # 행 갯수
    test_size = X_test.shape[0]
    print(f"""#{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기 : {train_size},
            검증데이터 크기 : {test_size}""")
    print(f"#{n_iter} 검증 세트 인덱스 : {test_index}")
    cv_accuracy.append(accuracy)

# 개별 iteration 별 정확도를 합하여 평균 정확도 계산
print(f"평균 검증 정확도 : {np.mean(cv_accuracy)}")
```
</div>
</details>

</br>

### Stratified KFold 연습

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df["label"] = iris.target
iris_df["label"].value_counts() # 레이블의 갯수/비율 측정
kfold = KFold(n_splits=3)   # 데이터셋 3개로 분할, 단순 KFold
n_iter = 0

for train_index, test_index in kfold.split(iris_df) : # kfold의 각 분할 부분마다
    n_iter += 1
    label_train = iris_df["label"].iloc[train_index]
    label_test = iris_df["label"].iloc[test_index]
    print(f"교차검증 : {n_iter}")
    print(f"학습 레이블 데이터 분포 : \n{label_train.value_counts()}")
    print(f"검증 레이블 데이터 분포 : \n{label_test.value_counts()}")

# 결과 보면 레이블 0,1,2가 모두 들어있지 않음.
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter = 0

# StratifiedKFold 의 경우, 스플릿할 데이터와, 분할 기준이 되는 array 필요
for train_index, test_index in skf.split(iris_df, iris_df["label"]) :
    n_iter += 1
    # print(train_index)  # 행 넘버 뽑음
    # print(test_index)
    label_train = iris_df["label"].iloc[train_index]
    label_test = iris_df["label"].iloc[test_index]
    print(f"교차검증 : {n_iter}")
    print(f"학습 레이블 데이터 분포 : \n{label_train.value_counts()}")
    print(f"검증 레이블 데이터 분포 : \n{label_test.value_counts()}")

# StratifiedKFold는 0,1,2 레이블의 비율 일정하게 만들어줌.
```
</div>
</details>

</br>

## Cross Validation Score
#### sklearn.model_selection.cross_val_score
* sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)
* cv → 폴드 갯수
* fit_params → 교차 검증으로 나온 최적 모형을 학습까지 시켜줌   
dict


<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state = 156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scroes = cross_val_score(dt_clf, data, label, scoring = "accuracy", cv = 3)
print("교차 검증별 정확도 : ", np.round(scroes, 4))
print("평균 검증 정확도 : ", np.round(np.mean(scroes), 4))

# KFold 에서 사용해야 하는 for문 없이 바로 결과 나옴
#### GridSerchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에
* class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# 데이터를 로딩하고 학습데이터와 테스트 데이터 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,
                                test_size = 0.2,
                                random_state = 121)

dtree = DecisionTreeClassifier()

# parameter 들을 dictionary 형태로 설정
parameters = {"max_depth" : [1,2,3], "min_samples_split":[2,3]}

# 아래의 GridSearchCV 에서 사용
import pandas as pd
from sklearn.metrics import accuracy_score

# param_grid 의 하이퍼 파라미터들을 3개의 train, test_set_fold 로 
# 나누어서 테스트 수행 설정.
# refit = True 가 default 임. True 이면 가장 좋은 파라미터 설정으로 재학습 시킴.
grid_dtree = GridSearchCV(dtree, param_grid = parameters, cv = 3, refit = True)

# 붓꽃 Train 데이터로 param_grid 의 하이퍼 파라미터들을 순차적으로 학습/평가.
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과 추출하여 DataFrame 으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[["params", "mean_test_score", "rank_test_score",
            "split0_test_score", "split1_test_score", "split2_test_score"]]
print(f"GridSearchCV 최적 파라미터 : {grid_dtree.best_params_}")
print(f"GridSearchCV 최고 정확도 : {grid_dtree.best_score_ : .4f}")
# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCV의best_estimator_는 이미 최적 하이퍼 파라미터로 학습이 됨
pred = estimator.predict(X_test)
print(f"테스트 데이터 세트 정확도 {accuracy_score(y_test, pred):.4f}")
X_test
y_test
estimator.predict([[7. , 3.2, 4.7, 1.4]])
estimator.predict_proba([[7. , 3.2, 4.7, 1.4]]) # 확률로 보여줌.
                            # 아마 가장 높은 확률 또는 90% 이상
```

</div>
</details>

</br>

# 데이터 인코딩
* 데이터 전처리는 알고리즘 만큼 중요 (Garbage in, Garbage out)
* 데이터는 문자열을 입력 값으로 허용 하지 않음  
따라서 문자열을 인코딩하여 숫자로 변환 → feature vectorization 기법
* Label Encoding과 One-Hot Encoding

### sklearn.preprocessing.LabelEncoder
* class sklearn.preprocessing.LabelEncoder

<details>
<summary>레이블 인코더 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.preprocessing import LabelEncoder
# 레이블 인코딩 이용  /  각 레이블에 숫자 순서대로 하나씩 붙여줌
items = ["TV", "냉장고", "전자렌지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]

# LabelEncoder 를 객체로 생성한 후, fit() 과 transform() 으로 label 인코딩 수행.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print("인코딩 변환값:", labels)

# 넘버 붙여줌
print("인코딩 클래스:", encoder.classes_)
# 넘버 순서대로 보여줌
print("디코딩 원본 값:", encoder.inverse_transform([4,5,2,0,1,1,3,3]))
# 넘버를 원래 인덱스 값으로 재변환
```
</div>
</details>

</br>

### sklearn.preprocessing.OneHotEncoder
* class sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse=True, dtype=<class 'numpy.float64'>, handle_unknown='error')

<details>
<summary>One-Hot 인코더 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.preprocessing import OneHotEncoder
# One-Hot 인코딩은 숫자값으로 나타낸 인덱스를 0,1 어레이로 나타내줌
items = ["TV", "냉장고", "전자렌지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]

# 먼저 숫자값으로 변환을 위해 LabelEncoder로 변환합니다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)   # 요소 8개짜리 시리즈
# 2차원 데이터로 변환
labels = labels.reshape(-1, 1)      # .T 안됨, 8행 1열로 변환
print(labels.shape)

# 라벨 인코딩 대신, items에 직접 튜플로 번호 붙여줘도 가능
# items = [("TV", 1), ("냉장고", 2)] 등

# One-Hot 인코딩 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print("One-Hot Encoding Data\n", oh_labels.toarray())   # array 형태로 출력
print("Dimensions of One-Hot Encoding Data\n", oh_labels.shape)
```
</div>
</details>

</br>

#### 판다스에서지원하는 원 핫 인코딩 API get_dummies()

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd

items = ["TV", "냉장고", "전자렌지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]
df = pd.DataFrame({"item" : items})
pd.get_dummies(df)
```
</div>
</details>

</br>

### 피쳐 스케일링과 정규화
* 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업 : feature scaling  
→ 표준화(Standardization)와 정규화(Normalization)

#### 표준화(Standardization)
* 평균이 0이고 분산이 1인 Gaussian distribution으로 변화
* $x'=\frac{(x - \mu(x))}{\sigma(x)}$


#### 정규화(Normaliztion)
* 서로 다른 피쳐의 크기를 통일하기 위해 크기를 변환하는 개념
* Min-Max 방식 : $x' = \frac{x-min(x)}{max(x) - min(x)}$
* 사이킷 런의 Normalizer 모듈은 선형대수에서의 정규화개념이 적용,   
개별벡터의 크기를 맞추기 위해 변환
* $x' = \frac{x}{\sqrt{x²+y²+z²}}$

## 사이킷런 피쳐 스케일링 지원
#### StandardScaler
* 평균이 0, 분산이 1인 정규분포 형태로 변환
#### MinMaxScaler
* 데이터 값을 0과 1사이의 범위 값으로 변환(음수값이 있으면 -1~1)

<details>
<summary>평균 분산 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_iris
import pandas as pd

# 붓꽃 데이터 셋을 로딩하고 DataFrame으로 변환

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)

print(f"featrue들의 평균 값 : \n{iris_df.mean()}")
print(f"feature들의 분산 값 : \n{iris_df.var()}")

import matplotlib.pyplot as plt
plt.scatter(iris.target, iris_df.iloc[:, 0], alpha=0.2)
plt.scatter(iris.target, iris_df.iloc[:, 1], alpha=0.2)
plt.scatter(iris.target, iris_df.iloc[:, 2], alpha=0.2)
plt.scatter(iris.target, iris_df.iloc[:, 3], alpha=0.2)
plt.show()
```
</div>
</details>

</br>

<details>
<summary>Standard Scaler 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.preprocessing import StandardScaler

# StandardScaler 객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 셋 변환. fit()과 transform() 호출.
scaler.fit(iris_df) # 평균, 분산 계산이 fit
iris_scaled = scaler.transform(iris_df) # 찾아낸 평균, 분산으로 변환수행

# transform()시 scale 변환된 데이터 셋이 numpy ndarray로 반환되어 이를 DF로 변환.
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print(f"feature 평균 : \n{iris_df_scaled.mean()}")
print(f"feature 분산 : \n{iris_df_scaled.var()}")
iris_df_scaled.describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.subplot(4,2,1)
plt.scatter(iris.target, iris_df_scaled.iloc[:, 0])
plt.subplot(4,2,2)
plt.hist(iris_df_scaled.iloc[:, 0], bins=50)
plt.subplot(4,2,3)
plt.scatter(iris.target, iris_df_scaled.iloc[:, 1])
plt.subplot(4,2,4)
plt.hist(iris_df_scaled.iloc[:, 1], bins=50)
plt.subplot(4,2,5)
plt.scatter(iris.target, iris_df_scaled.iloc[:, 2])
plt.subplot(4,2,6)
plt.hist(iris_df_scaled.iloc[:, 2], bins=50)
plt.subplot(4,2,7)
plt.scatter(iris.target, iris_df_scaled.iloc[:, 3])
plt.subplot(4,2,8)
plt.hist(iris_df_scaled.iloc[:, 3], bins=50)
plt.tight_layout()
plt.show()
```
</div>
</details>

</br>

X_train의 경우 fit 시키고, transform   /  X_test 는 그냥 transform만 시킴. X_train으로 얻은 평균분산을 통해 X_test로 추론하기 위해서임.

## 피마 인디언 당뇨 분석 연습
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression

from modules import Eval

origin = pd.read_csv(".\csv_data\diabetes.csv")
df_dbt = pd.DataFrame(origin)

# 피처 데이터 세트 X, 레이블 데이터 세트 y를 추출.
# 맨 끝이 Outcome 컬럼으로 레이블 값임, 컬럼위치 -1 을 이용해 추출
dbt_data = df_dbt.iloc[:, 0:-1]
dbt_label = df_dbt.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(dbt_data, dbt_label,
                                            test_size = 0.2, 
                                            random_state = 121, 
                                            stratify=dbt_label)

dtree = DecisionTreeClassifier()

parameters = {"max_depth" : [5, 6, 7], "min_samples_split":[2,3]}

grid_dtree = GridSearchCV(dtree, param_grid = parameters, cv = 3, refit = True)

grid_dtree.fit(X_train, y_train)

scores_df = pd.DataFrame(grid_dtree.cv_results_)

scores_df[["params", "mean_test_score", "rank_test_score",
            "split0_test_score", "split1_test_score", "split2_test_score"]]
print(f"GridSearchCV 최적 파라미터 : {grid_dtree.best_params_}")
print(f"GridSearchCV 최고 정확도 : {grid_dtree.best_score_ : .4f}")

estimator = grid_dtree.best_estimator_

pred = estimator.predict(X_test)
print(f"테스트 데이터 세트 정확도 {accuracy_score(y_test, pred):.4f}")
```

```python
# 로지스틱 회귀로 검증
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

Eval.get_clf_eval(y_test, pred)
Eval.precision_recall_curve_plot(y_test, pred_proba)

plt.hist(df_dbt["Glucose"], bins=10)

# 0값을 검사할 피처명 리스트 객체 설정
zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# 전체 데이터 건수
total_count = df_dbt["Glucose"].count()

# 피처별로 반복 하면서 데이터 값이 0인 데이터 건수 추출하고, 퍼센트 계산
for feature in zero_features :
    # 전체 데이터에서, 지정된 feature column에서 0이 들어있는 row 인덱스 번호 꺼내고, 다시 거기서 그 feature column만 불러옴.
    zero_count = df_dbt[df_dbt[feature] == 0][feature].count()
    print(f"{feature} 0 건수는 {zero_count}, 비율은 {100*zero_count/total_count:.2f}%")
# zero_features 리스트 내부에 저장된 개별 피처들에 대해서 0값을 평균으로 대체
df_dbt[zero_features] = df_dbt[zero_features].replace(0, df_dbt[zero_features].mean())
```
```python
from sklearn.preprocessing import StandardScaler

X = df_dbt.iloc[:, :-1]
y = df_dbt.iloc[:, -1]

# StandardScaler 클래스를 이용해 피처 데이터 세트에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=156, stratify=y)

# 로지스틱 회귀 적용
lr_clf_2 = LogisticRegression()
lr_clf_2.fit(X_train, y_train)
pred_2 = lr_clf_2.predict(X_test)
pred_proba_2 = lr_clf_2.predict_proba(X_test)[:, 1]

Eval.get_clf_eval(y_test, pred_2)
```
</div>
</details>

</br>

# 모델평가
## Confusion matrix

||예측P|예측N||
|:---:|:---:|:---:|:---:|
|실제P|TP|FN|Recall(Sensitivity)|
|실제N|FP|TN|Specificity|
||Precision|NegPredVal|Accuracy|


### Accuracy = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수
* 정확도는 직관적으로 모델 예측 성능을 나타내는 평가 지표.  
하지만 이진 분류의 경우 데이터의 구성에 따라 ML 모형의 성능 왜곡할 수 있음.   
정확도 수치 하나만 가지고 성능을 평가하지 않음
* 특히 정확도는 imbalanced 레이블 값 분포하에서는 적합하지 않은 지표
### Recall = $\frac{TP}{TP + FN}$ (예측과 실제가 모두 참 / 실제 참인 값)
* 재현율은 실제 값이 P인 대상 중에 예측과 실제값이 P로 일치한 데이터의 비율
* 병이 있을 때 있다고 판단한 비율
### Precision = $\frac{TP}{TP + FP}$ (예측과 실제가 모두 참 / 참으로 예측한 값)
* 정밀도는 예측을 P로 한 대상 중에 예측과 실제 값이 P로 일치한 데이터의 비율
### Specificity = $\frac{TN}{FP + TN}$ (예측과 실제가 모두 거짓 / 실제 거짓인 값)
* 병이 없을 때 없다고 판단한 비율
### 정밀도와 재현율의 Trade Off
- 정밀도와 재현율이 강조될 경우 Threshhold를 조정해 해당 수치 조정 가능
- 상호 보완적인 수치이므로 Trade Off 작용
- 모든 환자를 양성으로 판정 → FN = TN = , Recall = 100%
- 확실한 경우만 양성, 나머지 모두 N → FP = 0, TP = 1

<details>
<summary>모형 평가 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def get_clf_eval(y_test, pred) :
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print("Confusion Matrix")
    print(confusion)
    print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}")

import numpy as np
import pandas as pd
from modules import DtPre

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv("./csv_data/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]   # 레이블 데이터 셋 추출
X_titanic_df = titanic_df.drop("Survived", axis=1)  # 피쳐 데이터 셋에서 레이블셋은 삭제

X_titanic_df = DtPre.transform_features(X_titanic_df) # 만들어둔 전처리 함수 적용

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,
                                                    test_size=0.2, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)

# 사이킷런의 혼동행렬
#  TN    FP  실제 N
#  FN    TP  실제 P
# 예측N 예측P
```
</div>
</details>

</br>


### ROC & AUC
* TPR = $\frac{TP}{TP + FN}$ (= Recall) | TPR은 1에 가까울 수록 좋다
* FPR = $\frac{FP}{FP + TN}$ (= $1-$Specificity)  |  FPR은 0에 가까울 수록 좋다
* Decision Threshold를 높인다면 P로 분류되는 경우가 작아짐   
만약 DT = 1 → TPR = FPR = 0     
만약 DT = 0 → TPR = FPR = 1
* 따라서 DT에 의해 Trade Off
* ROC는 (FPR, TPR)쌍을 평면상에 그림    
AUC는 ROC 아래 면적 → 1에 가까울 수록 좋은 수치     
AUC = 1 → TPR = 1, FPR = 0
### F1 Score
* 정밀도와 재현율을 결합한 지표, 어느 한쪽으로 치우치지 않는 수치일 때   
상대적으로 높은 값을 가짐 : 기하평균
* F1 = $\frac{2}{\frac{1}{recall} + \frac{1}{precision}}$ = $2\frac{precision \times recall}{precision + recall}$
## 정확도의 함정
* 더미 분류기 만들어도 데이터의 성질 또는 분포에 따라 정확도 높을 수 있음

<details>
<summary>더미 분류기 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator) :
    # fit() 메소드는 아무것도 학습하지 않음
    def fit(self, X, y=None) :
        pass
    # predict() 메소드는 단순히 Sex feature가 1 이면 0, 그렇지 않으면 1로 예측함
    def predict(self, X) :
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]) :
            if X["Sex"].iloc[i] == 1 :
                pred[i] = 0
            else :
                pred[i] = 1
                
        return pred
# 위에서 생성한 DummyClassifier에 적용
myclf = MyDummyClassifier()
myclf.fit(X_train, y_train)

mypredictions = myclf.predict(X_test)
print(f"DummyClassifier의 정확도는 : {accuracy_score(y_test, mypredictions):.4f}")
# 아무런 학습도 시키지 않고 단순히 여자면 살았다고 판단하기만 해도 83%정확도
```

```python
from sklearn.datasets import load_digits

class MyFakeClassifier(BaseEstimator) :
    # 학습 안함
    def fit(self, X, y) :
        pass

    # 입력값으로 들어오는 X 데이터 셋의 크기만큼 0 행렬로 반환
    def predict(self, X) :
        return np.zeros((len(X), 1), dtype=bool)

# 사이킷런의 내장 데이터 셋인 load_digits()를 이용하여 MNIST 데이터 로딩
digits = load_digits()

print(digits.data)
print(f"### digits.data.shape : {digits.data.shape}")
print(digits.target)
print(f"###digits.target.shape : {digits.target.shape}")
digits.target == 7
# digits 번호가 7번이면 True, 이를 astype(int)로 1로 변환, 7이 아니면 False, 0
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y,
                                                    random_state=11)
# 불균형한 레이블 데이터 분포도 확인
print(f"레이블 테스트 세트 크기 : {y_test.shape}")
print(f"테스트 세트 레이블의 0과 1의 분포도")
print(pd.Series(y_test).value_counts())

# FakeClassifier 적용
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train)
fakepred = fakeclf.predict(X_test)
print(f"모든 예측을 0으로 하여도 정확도는 : {accuracy_score(y_test, fakepred):.3f}")
# 앞절의 예측 결과인 fakepred와 실제 결과인 y_test의 Confusion Matrix 출력
confusion_matrix(y_test, fakepred)
# 정확도 높지만 정밀도와 재현율 0
print(f"정밀도 : {precision_score(y_test, fakepred)}")
print(f"재현율 : {recall_score(y_test, fakepred)}")
```
</div>
</details>

</br>

# 분류 결정 임곗값에 따른 Positive 예측 확률 변화
<details>
<summary>임곗값 연습 코드 펼치기/접기</summary>
<div markdown="1">

```python
titanic_df = pd.read_csv("./csv_data/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]   # 레이블 데이터 셋 추출
X_titanic_df = titanic_df.drop("Survived", axis=1)  # 피쳐 데이터 셋에서 레이블셋은 삭제

X_titanic_df = DtPre.transform_features(X_titanic_df) # 만들어둔 전처리 함수 적용

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,
                                                    test_size=0.2, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)
pred_proba = lr_clf.predict_proba(X_test)
pred = lr_clf.predict(X_test)
print(f"pred_proba()결과 Shape : {pred_proba.shape}")
print(f"pred_proba array에서 앞 5개만 샘플로 추출 : \n {pred_proba[:5]}")

# 예측 확률 array와 예측 결과값 array를 concatenate하여
# 예측 확률과 결과값을 한눈에 확인
# pred는 [1,0,0...] 형태의 array
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1,1)], axis=1)
# 앞이 0, 뒤가 1이 될 확률
print(f"두개의 class 중에서 더 큰 확률을 클래스 값으로 예측 \n {pred_proba_result[:5]}")
```
```python
from sklearn.preprocessing import Binarizer

X = [[1, -1, 2],
     [2, 0, 0],
     [0, 1.1, 1.2]]

# threshold 기준값보다 같거나 작으면 0을, 크면 1을 반환  /  초과
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))
# Binarizer의 threshold 설정값, 분류 결정 임곗값
custom_threshold = 0.5

# predict_proba() 반환값의 두번째 컬럼, 즉 Positive 클래스 컬럼 하나만 추출하여
# Binarizer 적용  /  Positive 일 확률이 0.5를 초과하는가?
pred_proba_1 = pred_proba[:, 1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) # 확률검사
custom_predict = binarizer.transform(pred_proba_1)  # 0, 1 변환

get_clf_eval(y_test, custom_predict)
# 앞과 동일한 결과
```
```python
# Binarizer의 threshold 설정값을 0.4로 변경. 분류 결정 임곗값을 낮춤.
custom_threshold = 0.4

pred_proba_1 = pred_proba[:, 1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) # 확률검사
custom_predict = binarizer.transform(pred_proba_1)  # 0, 1 변환

get_clf_eval(y_test, custom_predict)
# 재현율 높아짐 / 정밀도 낮아짐
```
</div>
</details>

</br>

<details>
<summary>임곗값 변화에 따른 스코어 변화 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 테스트를 수행할 모든 임곗값을 리스트 객체로 저장
thresholds = [0.4,0.45, 0.5, 0.55, 0.6]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds) :
    # thresholds list 객체 내의 값을 차례로 iteration 하면서 Evaluation 수행
    for custom_threshold in thresholds :
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print(f"임곗값 : {custom_threshold}")
        get_clf_eval(y_test, custom_predict)

get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
# 임곗값 상승할 수록 재현율 하락, 정밀도 상승 trade off
from sklearn.metrics import precision_recall_curve

# 레이블 값이 1일때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]

# 실제값 데이터 셋과 레이블 값이 1일 때의 예측 확률을
# precision_recall_curve 인자로 입력
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
print(f"반환된 분류 결정 임곗값 배열의 Shape : {thresholds.shape}")
print(f"반환된 precision 배열의 Shape : {precisions.shape}")
print(f"반환된 recalls 배열의 Shape : {recalls.shape}")

print(f"thresholds 5 samples : {thresholds[:5]}")
print(f"precisions 5 samples : {precisions[:5]}")
print(f"recalls 5 samples : {recalls[:5]}")

# 반환된 임계값 배열 행이 147건이므로 샘플 10건만 추출, 임곗값을 15 steps로 추출
thr_index = np.arange(0, thresholds.shape[0], 15)
print(f"샘플 추출을 위한 임곗값 배열의 index 10개 : {thr_index}")
print(f"샘플용 10개의 임곗값 : {np.round(thresholds[thr_index], 2)}")

# 15 steps 단위로 추출된 임곗값에 따른 정밀도와 재현율 값
print(f"샘플 임곗값별 정밀도 : {np.round(precisions[thr_index],3)}")
print(f"샘플 임곗값별 재현율 : {np.round(recalls[thr_index], 3)}")
```

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def precision_recall_curve_plot(y_test, pred_proba_c1) :
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 plot 수행.
    # 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision")
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # threshold값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    # thresholds의 값을 0.10394, 0.2~ 이런식으로 배열하고 소숫점 2자리 반올림
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel("Threshold value"); plt.ylabel("Precision and Recall value")
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr_clf.predict_log_proba(X_test)[:,1])
```
</div>
</details>

</br>

## 사이킷런 ROC 곡선 및 AUC 스코어
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
print(f"F1 scores : {f1:.4f}")
def get_clf_eval(y_test, pred) :
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    # F1 스코어 추가
    f1 = f1_score(y_test, pred)
    print("Confusion Matrix")
    print(confusion)
    # f1 score print 추가
    print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F1 :{f1:.4f}")
thresholds = [0.4,0.45, 0.5, 0.55, 0.6]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
# 임곗값의 변화와 재현율, 정밀도에 따라 F1 스코어 변동
from sklearn.metrics import roc_curve

# 레이블 값이 1일때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]

fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)
# 반환된 임곗값 배열에서 샘플로 데이터를 추출하되, 임곗값을 5 steps로 추출
# thresholds[0]는 max(예측확률)+1로 임의 설정.
# 이를 제외하기 위해 np.arange 는 1부터 시작
thr_index = np.arange(1, thresholds.shape[0], 5)
print(f"샘플 추출을 위한 임곗값 배열의 Index : {thr_index}")
print(f"샘플 index로 추출한 임곗값 : {np.round(thresholds[thr_index], 2)}")

# 5 steps 단위로 추출된 임곗값에 따른 FPR, TPR 값
print(f"샘플 임곗값별 FPR : {np.round(fprs[thr_index], 3)}")
print(f"샘플 임곗값별 TPR : {np.round(tprs[thr_index], 3)}")
```
```python
def roc_curve_plot(y_test, pred_proba_c1) :
    # 임곗값에 따른 FPR, TPR 값을 반환 받음
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림
    plt.plot(fprs, tprs, label="ROC")
    # 가운데 대각선 직선을 그림
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    # FPR X cnrdml Scale을 0.1 단위로 변경, X,Y 축명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlabel("FPR(1 - Sensitivity)"); plt.ylabel("TPR(Recall)")
    plt.legend()
    plt.show()

roc_curve_plot(y_test, lr_clf.predict_proba(X_test)[:,1])
```
```python
from sklearn.metrics import roc_auc_score

# pred = lr_clf.predict(X_test)
# roc_score = roc_auc_score(y_test, pred)

pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_proba)
print(f"ROC AUC 값 : {roc_score:.4f}")
```
</div>
</details>

</br>

# 타이타닉 데이터를 이용한 ML 연습

### 데이터 전처리
<details>
<summary>전처리 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
titanic_df = pd.read_csv("./csv_data/titanic_train.csv")
titanic_df.head()
print("### train 데이터 정보 ### \n")
print(titanic_df.info())    #891 개 중에 null 갯수 알 수 있음
titanic_df.isnull().sum()
titanic_df.describe()
titanic_df["Age"].fillna(titanic_df["Age"].mean(), inplace=True)
titanic_df["Cabin"].fillna("N", inplace=True)
titanic_df["Embarked"].fillna("N", inplace=True)    # 전처리, 데이터의 특성 숙지
print(f"\n {titanic_df.info()} \n {titanic_df.isnull().sum()}")
print(f"\n Sex 값 분포 : \n {titanic_df['Sex'].value_counts()}")
print(f"\n Cabin 값 분포 : \n {titanic_df['Cabin'].value_counts()}")
print(f"\n Embarked 값 분포 : \n {titanic_df['Embarked'].value_counts()}")
titanic_df["Cabin"] = titanic_df["Cabin"].str[:1]   # 첫글자만 따서 저장
print(titanic_df["Cabin"].head())
titanic_df["Cabin"].value_counts()
titanic_df.groupby(["Sex", "Survived"])["Survived"].count()
sns.barplot(x="Sex", y="Survived", data=titanic_df) # 비율로 보여줌
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=titanic_df)
# hue 는 각각 분할하는 것
# 입력 age에 따라 구분값을 반환하는 함수 설정. DataFrame 의 apply lambda 식에 사용.
def get_category(age) :
    cat = ''
    if age <= -1 : cat = "Unknown"
    elif age <= 5 : cat = "Baby"
    elif age <= 12 : cat = "Child"
    elif age <= 18 : cat = "Teenager"
    elif age <= 25 : cat = "Student"
    elif age <= 35 : cat = "Young Adult"
    elif age <= 60 : cat = "Adult"
    else : cat = "Elderly"
    
    return cat
# 막대그래프의 크기 figure를 더 크게 설정
plt.figure(figsize=(10,6))

# X 축의 값을 순차적으로 표시하기 위한 설정  /  sns.barplot(order=)
group_names = ["Unknown", "Baby", "Child", "Teenager", "Student", "Young Adult", "Adult", "Elderly"]

# lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지정.
# get_category(X)는 입력값으로 "Age" 컬럼값을 받아서 해당하는 cat 반환
titanic_df["Age_cat"] = titanic_df["Age"].apply(lambda x : get_category(x))
sns.barplot(x="Age_cat", y="Survived", hue="Sex", data=titanic_df, order=group_names)
titanic_df.drop("Age_cat", axis=1, inplace=True)    # 한번 하고 없앰
from sklearn.preprocessing import LabelEncoder
# 스트링으로 표현된 데이터들 넘버링  /  레이블 인코딩
def encode_features(dataDF) :
    features = ["Cabin", "Sex", "Embarked"]
    for feature in features :
        le = LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF

titanic_df = encode_features(titanic_df)
titanic_df.head()
```
</div>
</details>

</br>

### 전처리 과정을 함수로 리팩토링 → 학습 데이터에 적용하여 쉽게 처리
<details>
<summary>전처리 함수 코드 펼치기/접기</summary>
<div markdown="1">

```python
# Null 처리 함수
def fillna(df) :
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("N", inplace=True)
    df["Embarked"].fillna("N", inplace=True)
    df["Fare"].fillna(0, inplace=True)
    return df
# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df) :
    df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    return df
# 레이블 인코딩 수행
def format_features(df) :
    df["Cabin"] = df["Cabin"].str[:1]
    features = ["Cabin", "Sex", "Embarked"]
    for feature in features :
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df
# 앞에서 설정한 Data Processing 함수 호출
def transform_features(df) :
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

```
</div>
</details>

</br>

### 학습수행
<details>
<summary>학습용 데이터 전처리 코드 펼치기/접기</summary>
<div markdown="1">

```python
titanic_df = pd.read_csv("./csv_data/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]   # 레이블 데이터 셋 추출
X_titanic_df = titanic_df.drop("Survived", axis=1)  # 피쳐 데이터 셋에서 레이블셋은 삭제

X_titanic_df = transform_features(X_titanic_df) # 만들어둔 전처리 함수 적용
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,
                                                    test_size=0.2, random_state=121)
# 학습 데이터 셋 분할
```
</div>
</details>

</br>

#### 디시전 트리, 랜덤 포레스트, 로지스틱 회귀 모형 이용
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 결정트리, Random Forest, 로지스틱 회귀를 위한 Classifier
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# 디시전 트리 모형으로 검증
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print(f"DecisionTreeClassifier 정확도 : {accuracy_score(y_test, dt_pred) : .4f}")

# 랜덤 포레스트 모형으로 검증
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print(f"RandomForestClassifier 정확도 : {accuracy_score(y_test, rf_pred) : .4f}")

# 로지스틱 모형으로 검증
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print(f"LogisticRegression 정확도 : {accuracy_score(y_test, lr_pred) : .4f}")
```
</div>
</details>

</br>

#### KFold 모형 사용
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5) :
    # 폴드 세트가 5개인 KFold객체를 생성, 폴드 수만큼 예측결과 저장을 위해 리스트 객체 생성
    kfold = KFold(n_splits=folds)
    scores = []

    # KFold 교차 검증 수행
    for iter_count , (train_index, test_index) in enumerate(kfold.split(X_titanic_df)) :
        # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 형성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        # Classifier 검증
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print(f"교차검증 {iter_count} 정확도 : {accuracy : .4f}")

    # 5개 fold 에서의 평균 정확도 계산
    mean_score = np.mean(scores)
    print(f"평균 정확도 : {mean_score:.4f}")

# exec_kfold 호출  /  위에서 설정한 디시전 트리 모형 가져옴
exec_kfold(dt_clf, folds=5)
```
</div>
</details>

</br>

#### Cross_val_score 사용
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.model_selection import cross_val_score

# 위에서 설정한 디시전 트리 모형에 대해 피쳐, 레이블 값 넣어서 5회 교차검증
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores) :
    print(f"교차검증 {iter_count} 정확도 : {accuracy:.4f}")

print(f"평균 정확도 : {np.mean(scores):.4f}")
```
</div>
</details>

</br>

#### GridSearchCV 모형 사용
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.model_selection import GridSearchCV

parameters = {"max_depth" : [2, 3, 5, 10],      # 최대 깊이
              "min_samples_split" : [2, 3, 5],  # 분할되기 위해 노드가 가져야 하는 최소 샘플 수
              "min_samples_leaf" : [1, 5, 8]}   # 리프 노드가 가지고 있어야 할 최소 샘플 수

# 위에서 미리 분할해 놓은 훈련 세트 / test_train_split
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring="accuracy", cv=5)
grid_dclf.fit(X_train, y_train)

print(f"GridSearchCV 최적 하이퍼 파라미터 : {grid_dclf.best_params_}")
print(f"GridSearchCV 최고 정확도 : {grid_dclf.best_score_:.4f}")
best_dclf = grid_dclf.best_estimator_

# GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print(f"테스트 세트에서의 DecisionTreeClassifier 정확도 : {accuracy:.4f}")
```
</div>
</details>

</br>

## 결정트리와 앙상블
* 결정 트리는 매우 쉽고 유연하게 적용할 수 있는 알고리즘.   
데이터의 스케일링이나 정규화 등의 사전 가공의 영향이 매우 적음.     
예측 성능 향상을 위해 복잡한 규칙 구조를 가져야 함.     
이로 인해 과적합(overfitting)이 발생해 반대로 예측 성능이 저하될 수 있다는 단점

* 그러나 이러한 단점이 앙상블 기법에서는 오히려 장점으로 작용.      
앙상블은 매우 많은 여러개의 약한 학습기(예측 성능이 상대적으로 떨어지는 학습 알고리즘)를 결합해 확률적 보완과 오류가 발생한 부분에 대한 가중치를 계속 업데이트 하면서 예측 성능을 향상시킴.     
결정트리가 좋은 약한 학습기가 됨(GBM, XGBoost, LightGBM 등)
#### Decision Tree
* 결정 트리 알고리즘은 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내어 트리(Tree) 기반의 분류 규칙을 만듦(If-Else 기반 규칙).
* 따라서 데이터의 어떤 기준을 바탕으로 규칙을 만들어야 가장 효율적인 분류가 될 것인가가 알고리즘의 성능을 크게 좌우함.
* 루트 노드 -(분할)-    
규칙노드(규칙조건, 브랜치/ 서브트리 : 새로운 규칙 노드 기반의 서브 트리 생성) - (분할) -     
                         리프노드(결정된 분류값)     
                         규칙노드 - (분할) - ...
# 정보 균일도 측정 방법
### 정보 이득(Inforamtion Gain)
* 정보 이득은 엔트로피라는 개념을 기반으로 함.   
엔트로피는 주어진 데이터 집합의 혼잡도를 의미함.    
서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞여 있으면 엔트로피가 낮음.      
정보 이득 지수는 1에서 엔트로피 지수를 뺀 값. 즉, 1-엔트로피 지수.       
$(Entropy(T) = -\displaystyle\sum_i ^k p_i log_2 p_i)$
결정 트리는 이 정보 이득 지수로 분할 기준을 정함. 즉, 정보 이득이 높은 속성을 기준으로 분할함.
### 지니 계수
* 지니 계수는 원래 경제학에서 불평등 지수를 나타낼 때 사용하는 계수.    
0이 가장 평등, 1로 갈수록 불평등.   
$G(A) = 1-\displaystyle\sum_{k=0}^n p_k ^2$
머신러닝에 적용될 때는 의미론 적으로 재해석돼 데이터가 다양한 값을 가질수록 평등, 특정 값으로 쏠릴 경우에는 불평등한 값.    
즉, 다양성이 낮을 수록 균일도가 높다는 의미, 1로 갈수록 균일도가 높으므로 지니 계수가 높은 속성을 기준으로 분할하는 것.
## Decision Tree 분할 규칙
* 기본적으로 지니 계수를 이용해 데이터 세트를 분할.
* 정보이득이 높거나 지니 계수가 낮은 조건을 찾아서 자식트리에 노드를 분할.
* 데이터가 모두 특정 분류에 속하게 되면 분할을 멈추고 분류 결정.
### Decision Tree의 특징
* "균일도" 직관적이고 쉽다.
* 트리의 크기를 사전에 제한하는 것이 성능 튜닝에 효과적.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 모든 데이터를 만족하는 완벽한 규칙은 만들 수 없음.
#### max_depth
* 트리의 최대 깊이를 규정.
* 디폴트는 None. None으로 설정하면 min_sample_split까지 완전 분할, overfitting 우려.
#### max_features
* 최적 분할을 위해 고려할 최대 피처 개수. 디폴트 None, 모든 피처 사용하여 분할.
* int 형으로 지정하면 대상 피처의 개수, float형으로 지정하면 전체 피처 중 대상 피처의 퍼센트.
* 'sqrt'는 전체 피처 중 √(전체 피처) 개수 만큼 선정
* 'auto'로 지정하면 sqrt와 동일
* 'log'는 전체 피처 중 log2(전체 피처 개수)선정
* 'None'은 전체 피처 선정
#### min_samples_split
* 노드를 분할하기 위한 최소한의 샘플 데이터수로 과적합을 제어하는 데 사용.
* 디폴트는 2이고 작게 설정할수록 분할되는 노드가 많아져서 과적합 가능성 증가.
#### min_samples_leaf
* 말단 노드(Leaf)가 되기 위한 최소한의 샘플 데이터 수
* min_samples_split와 유사하게 과적합 제어 용도. 그러나 비대칭적(imbalanced) 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 이 경우는 작게 설정 필요.
#### max_leaf_nodes
* 말단 노드(Leaf)의 최대 개수

<details>
<summary>디시전 트리 시각화 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# DecisionTreeClassifier 생성
dt_clf = DecisionTreeClassifier(random_state=121)

# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,
                                                    test_size=0.2, random_state=11)

# DecisionTreeCalssifier 학습.
dt_clf.fit(X_train, y_train)
from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함.
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names,
                feature_names=iris_data.feature_names, impurity=True, filled=True)
# impurity - 불순도를 gini로 보여줌
```
```python
import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz로 읽어서 code 상에서 시각화
with open("tree.dot") as f :
    dot_graph = f.read()
graphviz.Source(dot_graph)
# max_depth=3 짜리 트리  /  보통 홀수 깊이 설정
with open("tree2.dot") as f :
    dot_graph = f.read()
graphviz.Source(dot_graph)
dt_clf.feature_importances_
import seaborn as sns
import numpy as np

# feature importance 추출
print(f"Feature Importance : {np.round(dt_clf.feature_importances_, 3)}")

# feature 별 importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_) :
    print(f"{name} : {value:.3f}")

# feature importance를 column별로 시각화 하기
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
```
</div>
</details>

</br>

## 결정트리 과적합(Overfitting)
<details>
<summary>디시전 트리 과적합 시각화 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

plt.title("3 Class values with 2 Features Sample data creation")

# 2차원 시각화를 위해서 feature는 2개, 결정값 클래스는 3가지 유형의
# classification 샘플 데이터 생성
X_features, y_labels = make_classification(n_features=2, n_redundant=0,
                                          n_informative=2, n_classes=3,
                                          n_clusters_per_class=1, random_state=0)

# plot 형태로 2개의 features로 2차원 좌표 시각화, 각 클래스값은 다른 색깔로 표시됨.
plt.scatter(X_features[:,0], X_features[:,1], marker="o", c=y_labels, s=25,
            cmap="rainbow", edgecolors="k")
```
```python                                          
import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y) :
    # 서브플롯들의 형태와 개별 플롯
    fig, ax = plt.subplots()

    # 학습 데이터 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap="rainbow", edgecolors="k", 
               clim=(y.min(), y.max()), zorder=3)
    ax.axis("tight")
    ax.axis("off")
    xlim_start, xlim_end = ax.get_xlim()
    ylim_start, ylim_end = ax.get_ylim()

    # 호출 파라미터로 들어온 training 데이터로 model 학습
    model.fit(X,y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행
    # meshgrid는 격자모양의 좌표평면
    xx, yy = np.meshgrid(np.linspace(xlim_start, xlim_end, num=200),
                         np.linspace(ylim_start, ylim_end, num=200))
    # np.c_는 2차원축을 기준으로 병합 → 평탄화 된 xx와 yy 결합
    # ndarray.ravel는 평탄화
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # contourf()를 이용하여 class boundary를 visualization 수행
    n_classes = len(np.unique(y))
    # levels는 컨투어 라인의 갯수를 나타내는 array, 증가형태여야 함 또는 int
    # X,y는 Z에 들어가는 좌표값들
    # Z는 컨투어가 그려지는 높이 값들
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                          levels=np.arange(n_classes + 1) - 0.5,
                          cmap="rainbow", clim=(y.min(), y.max()),
                          zorder=1)
```
```python
from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약없는 결정트리의 Decision Boundary 시각화
dt_clf_bdry = DecisionTreeClassifier()
visualize_boundary(dt_clf_bdry, X_features, y_labels)
# 특정한 트리 생성 min_sample_leaf 건 Decision Boundary 시각화
dt_clf_bdry_sbj = DecisionTreeClassifier(min_samples_leaf=6)
visualize_boundary(dt_clf_bdry_sbj, X_features, y_labels)
```
</div>
</details>

</br>

## Ensemble Learning
* 앙상블의 유형은 일반적으로 Voting, Bagging, Boosting 으로 구분, 이외에 Stacking 등의 기법이 있음
* 대표적인 배깅은 Random Forest 알고리즘이 있으며, 부스팅에는 에이다 부스팅, 그래디언트 부스팅, XGBoost, LightGBM 등이 있음.    
배깅은 bootstrap + aggregating  /  데이터 샘플 복원추출 + 집계       
정형 데이터의 분류나 회귀에서는 GBM 부스팅 계열의 앙상블이 전반적으로 높은 예측 성능을 나타냄.
* 넓은 의미로는 서로 다른 모델을 결합한 것들을 앙상블로 지칭하기도 함
#### 앙상블의 특징
* 단일 모델의 약점을 다수의 모델들을 결합하여 보완
* 뛰어난 성능을 가진 모델들로만 구성하는 것보다 성능이 떨어지더라도 서로 다른 유형의 모델을 섞는 것이 오히려 전체 성능에 도움이 될 수 있음
* 랜덤 포레스트 및 뛰어난 부스팅 알고리즘들은 모두 결정 트리 알고리즘을 기반 알고리즘으로 적용함
* 결정 트리의 단점인 overfitting을 수십~수천개의 많은 분류기를 결합해 보완하고 장점인 직관적인 분류 기준은 강화됨
#### Voting Type
* Hard Voting은 단순 다수결
* Soft Voting은 클래스의 확률로 결과를 보정
* 일반적으로 HV 보다는 SV이 예측 성능이 상대적으로 우수하여 주로 사용됨
* 사이킷런은 VotingClassifier 클래스를 통해 Voting을 지원

## Voting Classifier
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head()
# 개별 모델은 로지스틱 회귀와 KNN
lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 Soft Voting 기반의 앙상블 모델로 구현한 분류기
vo_clf = VotingClassifier(estimators=[("LR",lr_clf), ("KNN", knn_clf)],
                          voting="soft")

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    test_size=0.2, random_state=156)

# VotingClassifier 적용
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print(f"Voting 분류기 정확도 : {accuracy_score(y_test, pred): .4f}")

# 개별 모델의 적용
classifiers = [lr_clf, knn_clf]
for classifier in classifiers :
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print(f"{class_name} 정확도 : {accuracy_score(y_test, pred):.4f}")
```
</div>
</details>

</br>

## Decision Tree Prac
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import matplotlib.pyplot as plt

# features.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음. 이를 DF로 로드.
featuer_name_df = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/features.txt", sep="\s+",
                              header=None, names=["column_index", "column_name"])

# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
featuer_name = featuer_name_df.iloc[:,1].values.tolist()
print(f"전체 피처명에서 10개만 추출 : {featuer_name[:10]}")

feature_dup_df = featuer_name_df.groupby("column_name").count()
print(feature_dup_df[feature_dup_df["column_index"]>1].count())

# 중복 feature명에 대해서는 뒤에 번호 붙이는 함수
def get_new_feature_name_df(old_feature_name_df) :
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby("column_name").cumcount(),
                                  columns=["dup_cnt"])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how="outer")
    new_feature_name_df["column_name"] = new_feature_name_df[["column_name", "dup_cnt"]].apply(lambda x : x[0]+"_"+str(x[1])
                                                                                        if x[1]>0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(["index"], axis=1)
    return new_feature_name_df
def get_human_dataset() :

    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당
    featuer_name_df = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/features.txt", sep="\s+",
                              header=None, names=["column_index", "column_name"])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df() 를 이용, 신규 피처명 DF 생성
    new_feature_name_df = get_new_feature_name_df(featuer_name_df)

    # DataFrame에 피처명을 칼럼으로 부여하기 위해 리스트 객체로 다시 변환
    featuer_name = new_feature_name_df.iloc[:, 1].values.tolist()

    # 학습 피처 데이터 셋과 테스스 피처 데이터를 DF로 로딩, 컬럼명은 feature_name 적용
    X_train = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt",
                          sep="\s+", names=featuer_name)
    X_test = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt",
                          sep="\s+", names=featuer_name)

    # 학습 레이블과 테스트 레이블 데이터를 DF로 로딩하고 컬럼명은 action으로 부여
    # "\s+" 데이터 사이 간격 공백으로 구분
    y_train = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt",
                          sep="\s+", header=None, names=["action"])
    y_test = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt",
                          sep="\s+", header=None, names=["action"])

    # 로드된 학습/테스트용 DF를 모두 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()
print("## 학습 피처 데이터셋 info()")
print(X_train.info())
print(y_train["action"].value_counts())
```
```python
X_train.isna().sum().sum()  # Null값 확인
```
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 예제 반복 시 마다 동일한 예측 결과 도출을 위해 random_state 설정
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"결정 트리 예측 정확도 : {accuracy:.4f}")

# DecisionTreeClassifier의 하이퍼 파라미터 추출
print(f"DecisionTreeClassifier 기본 하이퍼 파라미터 : \n {dt_clf.get_params()}")
```
```python
from sklearn.model_selection import GridSearchCV

params = {
    "max_depth" : [6,8,10,12,16,20,24]
}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print(f"GridSearchCV 최고 평균 정확도 수치 : {grid_cv.best_score_}")
print(f"GridSearchCV 최적 하이퍼 파라미터 : {grid_cv.best_params_}")
```
```python
# GridSearchCV 객체의 cv_results_ 속성을 DataFrame으로 생성
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

# max_depth 파라미터 값과 그때의 테스트(Evaluation)셋, 학습 데이터 셋의 정확도 수치 추출
cv_results_df[["param_max_depth", "mean_test_score"]]

# 최적 깊이 16
max_depths = [6,8,10,12,16,20,24]
# max_depth 값을 변화 시키면서 그때마다 학습과 테스트 셋에서의 예측 성능 측정
for depth in max_depths :
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"max_depth = {depth} 정확도 {accuracy:.4f}")

# 최고 정확도 깊이 8
params = {
    "max_depth" : [8, 12, 16,20],
    "min_samples_split" : [16, 24]
}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print(f"GridSearchCV 최고 평균 정확도 수치 : {grid_cv.best_score_}")
print(f"GridSearchCV 최적 하이퍼 파라미터 : {grid_cv.best_params_}")
```
```python
best_df_clf = grid_cv.best_estimator_
pred1 = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred1)
print(f"결정 트리 예측 정확도 {accuracy:.4f}")
```
```python
import seaborn as sns

ftr_importance_values = best_df_clf.feature_importances_
# Top 중요도로 정렬을 쉽게 하고, 시본의 막대그래프로 쉽게 표현하기 위해 Series변환
ftr_importance = pd.Series(ftr_importance_values, index=X_train.columns)
# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importance.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title("Feature Importance Top 20")
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
```
</div>
</details>

</br>

### Random Forest
<details>
<summary>랜덤 포레스트 연습 코드 펼치기/접기</summary>
<div markdown="1">

```python
def get_new_feature_name_df(old_feature_name_df) :
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby("column_name").cumcount(),
                                  columns=["dup_cnt"])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how="outer")
    new_feature_name_df["column_name"] = new_feature_name_df[["column_name", "dup_cnt"]].apply(lambda x : x[0]+"_"+str(x[1])
                                                                                        if x[1]>0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(["index"], axis=1)
    return new_feature_name_df
```
```python
def get_human_dataset() :

    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당
    featuer_name_df = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/features.txt", sep="\s+",
                              header=None, names=["column_index", "column_name"])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df() 를 이용, 신규 피처명 DF 생성
    new_feature_name_df = get_new_feature_name_df(featuer_name_df)

    # DataFrame에 피처명을 칼럼으로 부여하기 위해 리스트 객체로 다시 변환
    featuer_name = new_feature_name_df.iloc[:, 1].values.tolist()

    # 학습 피처 데이터 셋과 테스스 피처 데이터를 DF로 로딩, 컬럼명은 feature_name 적용
    X_train = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt",
                          sep="\s+", names=featuer_name)
    X_test = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt",
                          sep="\s+", names=featuer_name)

    # 학습 레이블과 테스트 레이블 데이터를 DF로 로딩하고 컬럼명은 action으로 부여
    # "\s+" 데이터 사이 간격 공백으로 구분
    y_train = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt",
                          sep="\s+", header=None, names=["action"])
    y_test = pd.read_csv("./csv_data/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt",
                          sep="\s+", header=None, names=["action"])

    # 로드된 학습/테스트용 DF를 모두 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()
```
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 결정 트리에서 사용한 get_human_dataset()을 이용해 DF반환
X_train, X_test, y_train, y_test = get_human_dataset()

# 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"랜덤 포레스트 정확도 : {accuracy:.4f}")
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators" : [100],
    "max_depth" : [6, 8, 10, 12],
    "min_samples_leaf" : [8, 12, 18],
    "min_samples_split" : [8, 16, 20]
}
# RandomForestClassifier 객체 생성 후 GridSerchCV 수행
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)  # pc의 모든 리소스 사용
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train, y_train)

print(f"최적 하이퍼 파라미터 : \n {grid_cv.best_params_}")
print(f"최고 예측 정확도 : {grid_cv.best_score_:.4f}")
rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8,
                                 min_samples_split=8, random_state=0)
rf_clf1.fit(X_train, y_train)
pred = rf_clf1.predict(X_test)
print(f"예측 정확도 : {accuracy_score(y_test, pred):.4f}")
```
##### Feature Importance Top 20
```python
import matplotlib.pyplot as plt
import seaborn as sns

ftr_importance_values = rf_clf1.feature_importances_
ftr_importance = pd.Series(ftr_importance_values, index=X_train.columns)
ftr_top20 = ftr_importance.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title("Feature Importance Top 20")
sns.barplot(x=ftr_top20, y = ftr_top20.index)
plt.show()
```
</div>
</details>

</br>

# 타이타닉 데이터를 이용한 최적 피처추출 적용 연습
<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules import DtPre

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
titanic_df = pd.read_csv("./csv_data/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]   # 레이블 데이터 셋 추출
X_titanic_df = titanic_df.drop("Survived", axis=1)  # 피쳐 데이터 셋에서 레이블셋은 삭제

X_titanic_df = DtPre.transform_features(X_titanic_df) # 만들어둔 전처리 함수 적용

titanic_test_df = pd.read_csv("./csv_data/titanic_test.csv")
y_titanic_test_df = titanic_test_df["Survived"]   # 레이블 데이터 셋 추출
X_titanic_test_df = titanic_test_df.drop("Survived", axis=1)  # 피쳐 데이터 셋에서 레이블셋은 삭제

X_titanic_test_df = DtPre.transform_features(X_titanic_test_df) # 만들어둔 전처리 함수 적용
titanic_clf = DecisionTreeClassifier(random_state=121)

titanic_clf.fit(X_titanic_df, y_titanic_df)
# feature importance 추출
print(f"Feature Importance : {np.round(titanic_clf.feature_importances_, 3)}")
```
```python
# feature 별 importance 매핑
features = []
for name, value in zip(X_titanic_df.columns, titanic_clf.feature_importances_) :
    print(f"{name} : {value:.3f}")
    features.append({name : np.round(value, 3)})

# feature importance를 column별로 시각화 하기
sns.barplot(x=titanic_clf.feature_importances_, y=X_titanic_df.columns)
titanic_lr = LogisticRegression()

titanic_lr.fit(X_titanic_df, y_titanic_df)

pred_all = titanic_lr.predict(X_titanic_df)

print(np.round(accuracy_score(y_titanic_df, pred_all),3))
X_train_feat = X_titanic_df[["Pclass", "Sex", "Age", "Fare", "Cabin"]]
X_test_feat = X_titanic_test_df[["Pclass", "Sex", "Age", "Fare", "Cabin"]]
titanic_lr_feat = LogisticRegression()

titanic_lr_feat.fit(X_train_feat, y_titanic_df)

pred_feat = titanic_lr_feat.predict(X_test_feat)

print(np.round(accuracy_score(y_titanic_test_df, pred_feat),3))
```
</div>
</details>

</br>

# 여러가지 학습 알고리즘
## Boosting
* 부스팅 알고리즘은 여러 개의 약한 학습기(weak learner)를 순차적으로 학습-예측하면서 잘못 예측한 데이터에 가중치 부여를 토앻 오류를 개선해 나가는 학습 방법
* weak learner → 과적합 피하기 쉬움
* 부스팅의 대표적인 구현은 AdaBoost(Adaptive boosting)와 그래디언트 부스트가 있음
* 그래디언트 부스팅 → 기울기 기반 오류 감소 모형   
일차 분류의 오류에 대해서 다시 분류, 반복
* XGB → 손실함수 지정, 오버피팅, 소요시간 감소
### GBM Hyper Parameters
* loss : 경사 하강법에서 사용할 비용 함수 지정. 특별한 값 없으면 기본값인 'deviance'를 그대로 적용
* learning_rate : GBM이 학습을 진행할 때마다 적용하는 학습률. 0~1 사이의 값.    
너무 작으면 반복이 완료되어도 못 찾음, 반대로 너무 커도 오류 못 찾고 지나칠 수 있음.
* n_estimators : weak learner의 개수. 개수 많으면 일정 수준까지는 좋아질 수 있음.   
반면 개수가 많아지면 시간이 오래 걸림. 기본값 100
* subsample : 데이터 샘플링 비율. 기본값 1, 전체 기반 학습.     
0~1사이 설정 가능.

<details>
<summary>그래디언트 부스팅 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.ensemble import GradientBoostingClassifier
import time
from modules import HmDt
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = HmDt.get_human_dataset()
# GBM 수행 시간 측정을 위함. 시작 시간 설정.
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print(f"GBM 정확도 : {gb_accuracy:.4f}")
print(f"GBM 수행시간 : {time.time() - start_time}")
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators' : [100, 500],
    'learning_rate' : [0.05, 0.1]
}

grid_cv = GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)
print(f"최적 하이퍼 파라미터 : {grid_cv.best_params_}")
print(f"최고 예측 정확도 : {grid_cv.best_score_:.4f}")
```
```python
# GridSearchCV를 이용하여 최적으로 학습된 estimator로 predict 수행
gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"GBM 정확도 : {gb_accuracy:.4f}")
```
</div>
</details>

</br>

## XGBoost
#### class xgboost.XGBRegressor(*, objective='reg:squarederror', **kwargs)
<details>
<summary>XGB 파라미터 펼치기/접기</summary>
<div markdown="1">
Parameters

-   **n_estimators**  ([_int_](https://docs.python.org/3.6/library/functions.html#int "(in Python v3.6)")) – Number of gradient boosted trees. Equivalent to number of boosting rounds.
    
-   **max_depth**  (_Optional__[_[_int_](https://docs.python.org/3.6/library/functions.html#int "(in Python v3.6)")_]_) – Maximum tree depth for base learners.
    
-   **learning_rate**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Boosting learning rate (xgb's "eta")
    
-   **verbosity**  (_Optional__[_[_int_](https://docs.python.org/3.6/library/functions.html#int "(in Python v3.6)")_]_) – The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    
-   **objective**  ([_Union_](https://docs.python.org/3.6/library/typing.html#typing.Union "(in Python v3.6)")_[_[_str_](https://docs.python.org/3.6/library/stdtypes.html#str "(in Python v3.6)")_,_ [_Callable_](https://docs.python.org/3.6/library/typing.html#typing.Callable "(in Python v3.6)")_[__[_[_numpy.ndarray_](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.22)")_,_ [_numpy.ndarray_](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.22)")_]__,_ [_Tuple_](https://docs.python.org/3.6/library/typing.html#typing.Tuple "(in Python v3.6)")_[_[_numpy.ndarray_](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.22)")_,_ [_numpy.ndarray_](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.22)")_]__]__,_ _NoneType__]_) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
    
-   **booster**  (_Optional__[_[_str_](https://docs.python.org/3.6/library/stdtypes.html#str "(in Python v3.6)")_]_) – Specify which booster to use: gbtree, gblinear or dart.
    
-   **tree_method**  (_Optional__[_[_str_](https://docs.python.org/3.6/library/stdtypes.html#str "(in Python v3.6)")_]_) – Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. It's recommended to study this option from the parameters document:  [https://xgboost.readthedocs.io/en/latest/treemethod.html](https://xgboost.readthedocs.io/en/latest/treemethod.html).
    
-   **n_jobs**  (_Optional__[_[_int_](https://docs.python.org/3.6/library/functions.html#int "(in Python v3.6)")_]_) – Number of parallel threads used to run xgboost. When used with other Scikit-Learn algorithms like grid search, you may choose which algorithm to parallelize and balance the threads. Creating thread contention will significantly slow down both algorithms.
    
-   **gamma**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
    
-   **min_child_weight**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Minimum sum of instance weight(hessian) needed in a child.
    
-   **max_delta_step**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Maximum delta step we allow each tree's weight estimation to be.
    
-   **subsample**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Subsample ratio of the training instance.
    
-   **colsample_bytree**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Subsample ratio of columns when constructing each tree.
    
-   **colsample_bylevel**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Subsample ratio of columns for each level.
    
-   **colsample_bynode**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Subsample ratio of columns for each split.
    
-   **reg_alpha**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – L1 regularization term on weights (xgb's alpha).
    
-   **reg_lambda**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – L2 regularization term on weights (xgb's lambda).
    
-   **scale_pos_weight**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – Balancing of positive and negative weights.
    
-   **base_score**  (_Optional__[_[_float_](https://docs.python.org/3.6/library/functions.html#float "(in Python v3.6)")_]_) – The initial prediction score of all instances, global bias.
    
-   **random_state**  (_Optional__[__Union__[_[_numpy.random.RandomState_](https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState "(in NumPy v1.22)")_,_ [_int_](https://docs.python.org/3.6/library/functions.html#int "(in Python v3.6)")_]__]_) –
</div>
</details>

</br>

<details>
<summary>XGB 코드 펼치기/접기</summary>
<div markdown="1">

```python
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
                            recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from modules import Eval

import warnings
warnings.filterwarnings("ignore")

data_set = load_breast_cancer()
brst_data = data_set.data
brst_labels = data_set.target

brst_df = pd.DataFrame(data=brst_data, columns=data_set.feature_names)
brst_df["target"] = brst_labels

brst_df.head()
```
```python
X_train, X_test, y_train, y_test = train_test_split(brst_data, brst_labels,
                                                    test_size=0.3, random_state=121)

brst_xgb_clt = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
brst_xgb_clt.fit(X_train, y_train)

brst_pred = brst_xgb_clt.predict(X_test)
brst_pred_proba = brst_xgb_clt.predict_proba(X_test)

Eval.get_clf_eval(y_test, brst_pred, brst_pred_proba)
```
```python
fig, ax = plt.subplots(figsize=(10, 12))

plot_importance(brst_xgb_clt, ax=ax)
```

##### XGBoost Early Stop
```python
brst_xgb_clt_es = xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
brst_xgb_clt_es.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="logloss", eval_set=[(X_test, y_test)])

brst_pred_es = brst_xgb_clt_es.predict(X_test)
brst_pred_proba_es = brst_xgb_clt_es.predict_proba(X_test)

Eval.get_clf_eval(y_test, brst_pred_es, brst_pred_proba_es)
```
</div>
</details>

</br>

## SVM
* SVM은 분류에 사용되는 지도학습 머신러닝 모델이다.
* SVM은 서포트 벡터(support vectors)를 사용해서 결정 경계(Decision Boundary)를 정의하고, 분류되지 않은 점을 해당 결정 경계와 비교해서 분류한다.
* 서포트 벡터(support vectors)는 결정 경계에 가장 가까운 각 클래스의 점들이다.
* 서포트 벡터와 결정 경계 사이의 거리를 마진(margin)이라고 한다.
* SVM은 허용 가능한 오류 범위 내에서 가능한 최대 마진을 만들려고 한다.
* 파라미터 C는 허용되는 오류 양을 조절한다. C 값이 클수록 오류를 덜 허용하며 이를 하드 마진(hard margin)이라 부른다. 반대로 C 값이 작을수록 오류를 더 많이 허용해서 소프트 마진(soft margin)을 만든다.
* SVM에서는 선형으로 분리할 수 없는 점들을 분류하기 위해 커널(kernel)을 사용한다.
* 커널(kernel)은 원래 가지고 있는 데이터를 더 높은 차원의 데이터로 변환한다. 2차원의 점으로 나타낼 수 있는 데이터를 다항식(polynomial) 커널은 3차원으로, RBF 커널은 점을 무한한 차원으로 변환한다.
* RBF 커널에는 파라미터 감마(gamma)가 있다. 감마가 너무 크면 학습 데이터에 너무 의존해서 오버피팅이 발생할 수 있다.
<details>
<summary>SVM 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris, y_iris, test_size=0.5
)
```
```python
from sklearn.svm import SVC

# 감마값이 작으면 언더피팅 위험  /  C 값이 크면 하드마진에 의한 오버피팅 위험
sv_clf = SVC(gamma=0.001, C=100.)
sv_clf.fit(X_iris_train, y_iris_train)

sv_pred = sv_clf.predict(X_iris_test)
print(y_iris_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_iris_test, sv_pred))
```
</div>
</details>

</br>

## KNN (k-Nearest Neighbour)
* 레이블 값을 미리 주는 지도학습
* 장점 : 어떤 종류의 학습이나 준비 시간이 필요 없음
* 단점 : 특징 공간에 있는 모든 데이터에 대한 정보가 필요함      
가장 가까운 이웃을 찾기 위해 새로운 데이터에서 모든 기존 데이터까지의 거리를 확인해야 하기 때문     
데이터와 클래스가 많이 있으면, 많은 메모리 공간과 계산 시간이 필요함
<details>
<summary>KNN 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
sepal = iris.data[:, 0:2]
kind = iris.target

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.plot(sepal[kind==0][:,0], sepal[kind==0][:,1],
        "ro", label="Setosa")
plt.plot(sepal[kind==1][:,0], sepal[kind==1][:,1],
        "bo", label="Versicolor")
plt.plot(sepal[kind==2][:,0], sepal[kind==2][:,1],
        "yo", label="Verginica")

plt.legend()
X_i_knn = iris.data
y_i_knn = iris.target

X_i_knn_train, X_i_knn_test, y_i_knn_train, y_i_knn_test = train_test_split(
    X_i_knn, y_i_knn, test_size=0.2, random_state=121
)

print(X_i_knn_train.shape)
print(X_i_knn_test.shape)
```
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_i_knn_train, y_i_knn_train)

knn_pred = knn.predict(X_i_knn_test)
print(accuracy_score(y_i_knn_test, knn_pred))
```
```python
knn2 = KNeighborsClassifier(n_neighbors=5)

knn2.fit(X_i_knn, y_i_knn)

# 0 = setosa, 1 = versicolor, 2 = virginica
classes = {
    0 : "Setosa",
    1 : "Versicolor",
    2 : "Virginica"
}

# 새로운 데이터 제시
X_new = [[3,4,5,2],
        [5,4,2,2]]
y_predict = knn.predict(X_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
```
</div>
</details>

</br>

## Pipeline
* 파이프라인을 사용하면 데이터 사전 처리 및 분류의 모든 단계를 포함하는 단일 개체를 만들 수 있다.

* train과 test 데이터 손실을 피할 수 있다.
* 교차 검증 및 기타 모델 선택 유형을 쉽게 만든다.
* 재현성 증가
<details>
<summary>Pipeline 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import make_regression, make_classification

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_rand, y_rand = make_classification(n_samples=100, n_features=10, n_informative=2)

X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(X_rand, y_rand,
                                                        test_size=0.3, random_state=121)

X_rand_train.shape, X_rand_test.shape, y_rand_train.shape, y_rand_test.shape
```
```python
# it takes a list of tuples parameter
pipline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

# use the pipeline object as you would do with a regular classifier
pipline.fit(X_rand_train, y_rand_train)

y_rand_pred = pipline.predict(X_rand_test)

accuracy_score(y_rand_test, y_rand_pred)
```
</div>
</details>

</br>


# Regression

### sklearn.linear_model.LinearRegression
* class sklearn.linear_model.LinearRegression(*, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False)

</br>

# 시계열분석
## 1. 시계열 분석 개요
### 시계열 자료(time series data)
* 연도별, 분기별, 월별, 일별, 시간별 등 시간의 흐름에 따라 순서대로 관측되는 자료
* 예: GDP, 물가지수, 판매량, KOSPI, 강우량, 실험 및 관측자료 등
* 시간 단위 외에도 사회적 변화나 환경적 변동요인을 기준으로 시계열자료를 구분하는 경우도 있음
* 일반적으로 시계열 자료는 추세변동, 순환변동, 계절변동, 불규칙변동 요인으로 구성
* Countinuous time series / Discrete time series
* 실제로 많은 시계열들이 연속적으로 생성, 일정 시차를 두고 관측 → 이산시계열 형태를 지니는 경우가 많음
### 시계열 분석 개요(the nature of time series analysis)
* 시계열 자료들은 시간의 경과에 따라 관측, 시간에 영향 받음
    * 시계열 자료를 분석할 때 관측시점들 간의 시차(time lag)가 중요한 역할을 함
    * 관측 시점과 가까운 관측 시점의 자료들의 상관관계가 더 큼
    * 시계열은 일반적으로 시간 t를 하첨자로 하여 $[Z_t : t = 1, 2, 3, ...]$
* 시계열 분석의 목적
    * 과거 시계열자료의 패턴이 미래에도 지속적으로 유지된다는 가정하에서 현재까지 수집된 자료들을 분석하여 미래에 대한 예측(forecast)을 한다
    * 시계열자료를 분석할 때 관측시점들 간의 시차가 중요한 역할을 함
    * 시계열자료가 생셩된 시스템 또는 확률과정을 모형화하여 시스템 또는 확률과정을 이해하고 제어(control)할 수 있게 함
* 예측, 계획 그리고 목표
    * **예측**은 경영분야에 있어서 생산 계획, 수송, 인사에 관한 결정을 알리거나 장기 전략계획을 세우는데 도움을 줄 수 있는 흔한 통계적 업무
    * **목표**는 예측 및 계획과 연관되어 있는 것이 좋지만, 항상 일어나는 것은 아님. 목표를 어떻게 달성할지에 대한 계획과 목표가 실현 가능한지에 대한 예측 없이 목표를 세우는 경우가 너무 많음
    * **계획**은 예측과 목표에 대한 대응. 계획은 예측과 목표를 일치시키는데 필요한 적절한 행동을 결정하는 일을 포함. 
##### 시계열 분석 개요 코드

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# 추세변동(trend variation)

import numpy as np
import pandas as pd

# DatetimeIndex
dates = pd.date_range("2020-01-01", periods=48, freq="M")
# pd.date_range → 시작지점부터 월 freq 주면 월별 끝날짜

# additive model : tredn + cycle + seasonality + irregular factor
timestamp = np.arange(len(dates))
trend_factor = timestamp*1.1
cycle_factor = 10 * np.sin(np.linspace(0, 3.14 * 2, 48))
sesonal_factor = 7 * np.sin(np.linspace(0, 3.14 * 8, 48))
np.random.seed(2004)
irregular_factor = 2 * np.random.randn(len(dates))

df = pd.DataFrame({"timeseries": trend_factor + cycle_factor + sesonal_factor + irregular_factor,
                "trend" : trend_factor,
                "cycle" : cycle_factor,
                "seasonal" : sesonal_factor,
                "irregular" : irregular_factor},
                index=dates)
# Time series plot
```

```python
import matplotlib.pyplot as plt

plt.figure(figsize=[10,6])
df.timeseries.plot()
plt.title("Time Series (Additive Model)", fontsize=16)
plt.ylim(-12, 55)
plt.show()
```

```python
# -- Trend variation
# timestamp = np.arange(len(dates))
# trend_factor = timestamp * 1.1

plt.figure(figsize=[10,6])
df.trend.plot()
plt.title("Trend Factor", fontsize=16)
plt.ylim(-12, 55)
plt.show()
```

```python
# 순환변동(cyclical variation)

# 4년 주기
# -- Cycle variation
# cycle_factor = 10 * np.sin(np.linspace(0, 3.14 * 2, 48))

plt.figure(figsize=[10,6])
df.cycle.plot()
plt.title("Cycle Factor", fontsize=16)
plt.ylim(-12, 55)
plt.show()
```

```python
# 계절변동(seasonal variation)

# -- Seasonal factor
# sesonal_factor = 7 * np.sin(np.linspace(0, 3.14 * 8, 48))

plt.figure(figsize=[10,6])
df.seasonal.plot()
plt.title("Seasonal Factor", fontsize=16)
plt.ylim(-12, 55)
plt.show()
```

```python
# 불규칙변동(irregular variation or random variation)

# -- Irregular factor
# irregular_factor = 2 * np.random.randn(len(dates)) where np.random.seed(2004)

plt.figure(figsize=[10,6])
df.irregular.plot()
plt.title("Irregular Factor", fontsize=16)
plt.ylim(-12, 55)
plt.show()
```

```python
# Time series plot
import matplotlib.pyplot as plt

plt.figure(figsize=[10,6])
plt.title("Time Series (Additive Model)", fontsize=16)
df.timeseries.plot()
df.trend.plot()
df.cycle.plot()
df.seasonal.plot()
df.irregular.plot()
plt.legend()
plt.ylim(-12, 55)
```
</div>
</details>

</br>

## 2. 평활화 기법(Smoothing Methods)
* 데이터 셋을 모델링 하기 전에 기술통계와 시각화로 데이터셋을 탐색하는 과정이 있듯, 시계열에서도 복잡한 모델 구성에 앞서 수치나 시각화로 시계열을 기술하는 일이 분석작업의 출발점.
* 평활화는 분석잡업 중 하나로, 시계열의 복잡한 추세(trend)를 명확하게 파악하기 위한 방법
* 시계열은 전형적으로, 명백한 불규칙(or 오차)성분을 포함
* 시계열 자료는 특정 패턴을 파악하기 위해, 이같은 급격한 파동을 줄이는 평활화, 곡선 플롯으로 변환시키는 방법이 평활법. 대표적인 평활법은 이동평균법과 지수평활법
### 이동평균법(moving average method)
* 시계열을 평활화하는 가장 단순한 방법
* 시계열 자료의 특정시점(a time point) 관측치와 이 관측치의 이전과 이후 관측치의 평균으로 대체하는 방법을 "중심이동평균"(centered moving average)라고 함. 한 시점 앞 뒤 관측치를 평균내는 방법. 따라서 이평법을 하면 전체 관측치의 개수가 줄어듦
* 3기간 M3, 5기간 M5 등
* 이평을 이용할 때 가장 중요한 문제는 사용하는 과거자료의 적정개수, n의 크기
* 시계열에 뚜렷한 추세 존재, 불규칙 변동 심하지 않으면 작은 n의 개수 사용, 그렇지 않은 경우 n의 개술를 크게 함.     
$M_t = \frac{Z_t + Z_{t-1} + ... + Z_{t-n+1}}{n}$
#### Simple Moving Average(SMA)
* pandas.DataFrame.rolling
  * DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')

<details>
<summary>Parameter 펼치기/접기</summary>
<div markdown="1">

**window** : int, offset, or BaseIndexer subclass

Size of the moving window.

If an integer, the fixed number of observations used for each window.

If an offset, the time period of each window. Each window will be a variable sized based on the observations included in the time-period. This is only valid for datetimelike indexes. To learn more about the offsets & frequency strings, please see  [this link](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

If a BaseIndexer subclass, the window boundaries based on the defined  `get_window_bounds`  method. Additional rolling keyword arguments, namely  `min_periods`,  `center`, and  `closed`  will be passed to  `get_window_bounds`.

**min_periods** : int, default None

Minimum number of observations in window required to have a value; otherwise, result is  `np.nan`.

For a window that is specified by an offset,  `min_periods`  will default to 1.

For a window that is specified by an integer,  `min_periods`  will default to the size of the window.

**center** : bool, default False

If False, set the window labels as the right edge of the window index.

If True, set the window labels as the center of the window index.

**win_type** : str, default None

If  `None`, all points are evenly weighted.

If a string, it must be a valid  [scipy.signal window function](https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows).

Certain Scipy window types require additional parameters to be passed in the aggregation function. The additional parameters must match the keywords specified in the Scipy window type method signature.

**on** : str, optional

For a DataFrame, a column label or Index level on which to calculate the rolling window, rather than the DataFrame’s index.

Provided integer column is ignored and excluded from result since an integer index is not used to calculate the rolling window.

**axis** : int or str, default 0

If  `0`  or  `'index'`, roll across the rows.

If  `1`  or  `'columns'`, roll across the columns.

**closed** : str, default None

If  `'right'`, the first point in the window is excluded from calculations.

If  `'left'`, the last point in the window is excluded from calculations.

If  `'both'`, the no points in the window are excluded from calculations.

If  `'neither'`, the first and last points in the window are excluded from calculations.

Default  `None`  (`'right'`).

Changed in version 1.2.0: The closed parameter with fixed windows is now supported.

**method** : str {‘single’, ‘table’}, default ‘single’

New in version 1.3.0.

Execute the rolling operation per single column or row (`'single'`) or over the entire object (`'table'`).

This argument is only implemented when specifying  `engine='numba'`  in the method call.

Returns

`Window`  subclass if a  `win_type`  is passed

`Rolling`  subclass if  `win_type`  is not passed

</div>
</details>

</br>

##### 애플 주가 분석 예제

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 2020년 8월부터 일요일 8개를 조회
# start : 시작일, periods : 생설할 날짜의 개수, freq : 생성할 날짜의 주기
date = pd.date_range(start="2020-08", periods=8, freq="W")

# 데이터 프레임 컬럼으로 사용했을 때와 3주치 평균 컬럼을 추가
df = pd.DataFrame({
    "week" : date,
    "sales" : [39,44,40,45,38,43,39,np.nan],
    "3MA" : [0]*8
})

# 0~2 주차 평균을 3주차에 shift해서 적용
df["3MA"] = df[["sales"]].rolling(3).mean().shift(1)
df
```

```python
plt.figure(figsize=(10,8))
df.sales.plot()
df["3MA"].plot()
plt.show()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import FinanceDataReader as fdr

# 2010년~현재까지의 애플 주가를 데이터 프레임으로 불러옴
df_apple = fdr.DataReader("AAPL", start="2010")

# 가장 마지막(최신)의 10일치 주가 출력
df_apple.tail(10)
```

```python
df_apple[["Close"]].plot(figsize=(20,10))
df_apple["Close_7Days_Mean"] = df_apple["Close"].rolling(7).mean()
plt.title("Close Price for Apple")
```

```python
# 2010~2022년까지 애플의 종가 그래프에 7일전 평균값을 shift 하여 이평 그래프를 추가함.
# 7일 평균값으로 부드러워진 곡선 그래프를 확인할 수 있음.

last_day = datetime(2022,1,2)
df_apple.loc[last_day, "Close"] = np.nan
df_apple["Close_7Days_Mean"] = df_apple["Close"].rolling(7).mean().shift(1)
df_apple[["Close", "Close_7Days_Mean"]].plot(figsize=(30,20))

# 7일전 평균값 그래프 + 종가 그래프
```
```python
# pandas DataFrame에는 resample이라는 데이터프레임의 시계열 인덱스 기준으로 샘플링을
# 편하게 해주는 메소드가 있음. 아래와 같이하면 월단위로 시계열 데이터를 다시 만들어 줌.

# 월단위로 주식 가격의 평균을 샘플링  /  resample 함수
df_apple_monthly = df_apple.resample(rule="M").mean()

# 마지막 컬럼(Close_7Days_Mean) 제외
df_apple_monthly = df_apple_monthly.iloc[:, :-1]

# 월별 주가(종가)를 시각화
df_apple_monthly[["Close"]].plot(figsize=(20,10))
plt.title('Monthly Mean Close Price for Apple')
```

```python
# 월단위 평균값을 또 3개월치씩 이동평균을 적용하는 코드
df_apple_monthly[["Close_3Month_Mean"]] = df_apple_monthly[["Close"]].rolling(3).mean().shift(1)
df_apple_monthly[["Close", "Close_3Month_Mean"]].plot(figsize=(15,20))
df_apple_monthly["Cummulative"] = df_apple_monthly["Close"].expanding(3).mean()
df_apple_monthly[["Close", "Close_3Month_Mean", "Cummulative"]].plot(figsize=(15,20))
```

</div>
</details>

</br>

### Exponential Mobing Average(EMA)
* EMA는 새로운 데이터에 더 큰 가중치 부여하여 최근 데이터에 더욱 초점을 맞춤
* EMA는 모든 값에 동일한 가중치가 주어지는 SM에 비해 추세변화에 더 민감함
* pandas.Series.ewm()
  * DataFrame.ewm(com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None, method='single')
* Exactly one parameter:  `com`,  `span`,  `halflife`, or  `alpha`  must be provided.

<details>
<summary>Parameter 펼치기/접기</summary>
<div markdown="1">

**com** : float, optional

Specify decay in terms of center of mass

$α=\frac{1}{(1+com)}$, for  $com≥0$.

**span** : float, optional

Specify decay in terms of span

$α=\frac{2}{(span+1)}$, for  $span≥1$.

**halflife** : float, str, timedelta, optional

Specify decay in terms of half-life

$α=1−exp⁡(\frac{−ln⁡2}{halflife})$, for  $halflife>0$.

If  `times`  is specified, the time unit (str or timedelta) over which an observation decays to half its value. Only applicable to  `mean()`, and halflife value will not apply to the other functions.

New in version 1.1.0.

**alpha** : float, optional

Specify smoothing factor  α  directly

0<$α$≤1.

**min_periods** : int, default 0

Minimum number of observations in window required to have a value; otherwise, result is  `np.nan`.

**adjust** : bool, default True

Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings (viewing EWMA as a moving average).

-   When  `adjust=True`  (default), the EW function is calculated using weights  $wi=(1−α)^i$. For example, the EW moving average of the series $[x_0,x_1,...,x_t]$ would be:
    

$y_t=\frac{x_t+(1−α)x_{t−1}+(1−α)^2x_{t−2}+...+(1−α)^tx_0}{1+(1−α)+(1−α)^2+...+(1−α)^t}$

-   When  `adjust=False`, the exponentially weighted function is calculated recursively:
    

$y_0=x_0$
$y_t=(1−α)y_{t−1}+αx_t$,

**ignore_na** : bool, default False

Ignore missing values when calculating weights.

-   When  `ignore_na=False`  (default), weights are based on absolute positions. For example, the weights of  $x_0$  and  $x_2$  used in calculating the final weighted average of $[x_0, None,  x_2]$ are  $(1−α)^2$  and  $1$  if  `adjust=True`, and  $(1−α)^2$  and  $α$  if  `adjust=False`.
    
-   When  `ignore_na=True`, weights are based on relative positions. For example, the weights of  $x_0$  and  $x_2$  used in calculating the final weighted average of $[x_0, None,  x_2]$ are  $1−α$  and  $1$  if  `adjust=True`, and  $1−α$  and  $α$  if  `adjust=False`.
    

**axis** : {0, 1}, default 0

If  `0`  or  `'index'`, calculate across the rows.

If  `1`  or  `'columns'`, calculate across the columns.

**times** : str, np.ndarray, Series, default None

New in version 1.1.0.

Only applicable to  `mean()`.

Times corresponding to the observations. Must be monotonically increasing and  `datetime64[ns]`  dtype.

If 1-D array like, a sequence with the same shape as the observations.

Deprecated since version 1.4.0: If str, the name of the column in the DataFrame representing the times.

**method** : str {‘single’, ‘table’}, default ‘single’

New in version 1.4.0.

Execute the rolling operation per single column or row (`'single'`) or over the entire object (`'table'`).

This argument is only implemented when specifying  `engine='numba'`  in the method call.

Only applicable to  `mean()`

Returns : 

`ExponentialMovingWindow`  subclass

</div>
</details>

</br>

##### EMA 예제

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import numpy as np

data = {"val" : [1,4,2,3,2,5,13,10,10,12,14,np.NaN,16,12,20,22]}
df = pd.DataFrame(data).reset_index()
# df["val"].plot.bar(rot=0, subplot=True)
df.plot(kind="bar", x="index", y="val")
```

```python
import matplotlib.pyplot as plt

# val열에 ewm 메서드 적용 후 df에 추가
df2 = df.assign(ewm=df["val"].ewm(alpha=0.3).mean())

# ax2에 df2의 line chart 생성후 ax에 추가
ax = df.plot(kind="bar", x="index", y="val")
ax2 = df2.plot(kind="line", x="index", y="ewm", color="red", ax=ax)

plt.show()
```

```python
# myEWMA는 지수이동평균값을 df.ewm(span=3).mean()과 같이 계산해주도록 정의한 메소드

import pandas as pd
import numpy as np
df = pd.DataFrame({
    "week" : pd.date_range(start="2020-08", periods=8, freq="W"),
    "sales" : [39,44,40,45,38,43,39,np.nan],
    "3EMA" : [0]*8
})

# 지수 이동 평균을 계산할 함수
# data : 지수 이동 평균을 계산할 데이터
# span : 지수이동평균의 거리
```

```python
def myEWMA(data, span) :
    # 지수 이평을 계산해서 저장할 리스트
    ewma = [0]*len(data)
    # 지수 이평의 분자
    molecule = 0
    # 지수이평의 평균분모
    denominator = 0
    # 값에 곱해지는 가중치
    alpha = 2.0 / (1.0+span)
    
    for i in range(len(data)) :
        # 분자계산 data(1-alpha)앞의 데이터 / 누적되어서 옛날 데이터 영향 작아짐
        molecule = (data[i] + (1.0-alpha)*molecule)
        # 분모계산 (1-alpha)의 i승 / 
        denominator += (1-alpha)**i
        print("index", i)
        print("molecule", molecule)
        print("denominator", denominator)
        # 지수 이동 평균 계산
        ewma[i] = molecule/denominator
        print("ewma", ewma[i])
        print("="*30)
    
    return ewma
```

```python
df["ewma"] = myEWMA(df['sales'], 3)
```

```python
df["sales"].plot()
df["ewma"].plot()
plt.legend()
plt.show()
```

```python
plt.plot(myEWMA(df_apple_monthly["Close"], 3))
```

</div>
</details>

</br>

## 3.자기상관과 부분 자기상관
### 시차를 적용한 시계열 데이터
* 자기상관은 시차(lag)를 적용한 시계열 데이터를 이용하여 계산
* 시차를 적용한다는 것은 특정 시차만큼 관측값을 뒤로(즉 과거의 시점으로) 이동시키는 것을 의미
### 자기상관함수(ACF; Auto-Correlation Function)
* 자기상관은 다른 시점의 관측값 간 상호 연관성을 나타내므로 이는 시차를 적용한 시계열 데이터 간의 상관관계를 의미
* 자기상관 $AC_k$는 원래의 시계열 데이터$(y_t)$와 $k$ 시차가 고려된, 즉 k 기간 뒤로 이동한 시계열데이터 $(y_{t-k})$ 간의 상관관계로 정의
  * 예를 들어, $AC_1$은 시차0 시계열 데이터와 시차1 시계열 데이터 간의 상관관계
  * $AC_0$는 동일한 시계열 데이터 간의 상관관계이므로 언제나 1
### 편자기상관함수
* 편자기상관(partial autocorrelation)은 시차가 다른 두 시계열 데이터 간의 순수한 상호 연관성을 나타냄
* 편자기상관 $PAC_k$는 원래의 시계열 데이터 $(y_t)$와 시차$k$ 시계열 데이터 $(y_{t-k})$ 간의 순수한 상관관계로서 두 시점 사이에 포함된 모든 시계열 데이터(y_{t-1},y_{t-2},...,y_{t-k+1})의 영향은 제거됨
* 시차에 따른 일련의 편자기상관 $\{PAC_1,PAC_2,...,PAC_k\}$를 편자기상관함수(PACF)라고 함

## 중요! 정상성
* 정상성(stationarity)을 나타내는 시계열의 특징은 관측된 시간과 무관함
* 즉 $Y_t$가 정상성을 나타내는 시계열이라면, 모든 $s$에 대해 $[Y_t, Y_{t+1}, ..., Y_{t+s}]$의 분포에서 $t$와 무관함
* 따라서 추세나 계절성이 있는 시계열은 정상성을 나타내는 시계열이 아님.   
장기적으로 볼때, 예측할 수 있는 패턴을 나타내지 않을 것임.    
어떤 주기적인 행동이 있다 할지라도, 시간 그래프는 일정한 분산을 가지고 시계열의 평균이 시간축에 평행한 형태.
* 시계열 자료가 시계열 모형으로 적합시키기 위한 전제 조건에 해당함. 즉, 추세와 동향이 있는 상태로는 모형을 만들기 어려움.
### 차분(differencing)
* 차분은 시계열 수준에서 나타나는 변화를 제거하여 시계열의 평균 변화를 일정하게 만드는 것을 도움
* 결과적으로 추세나 계절성이 제거(또는 감소)되는 것
* ACF는 어떤 무작위의 신호가 두 시각에 취하는 값의 상관계수를 나타내는 함수
### 정상 시계열로의 변환
* 변동폭이 일정하지 않으면 로그 변환을 통해 시간의 흐름에 따라 분산이 일정하게 유지되는 정상 시계열로 변환
* 추세나 계절적 요인이 관찰되면 차분(differencing, 시계열 $y_t$의 각 관측값을 $y_t-y_{t-1}$로 대체) 과정을 통해 전 기간에 걸쳐 평균이 일정한 정상 시계열로 변환
* 변동폭이 일정하지 않고 추세와 계절적 요인 또한 존재하면 로그 변환과 차분 과정을 모두 적용하여 정상 시계열로 변환
##### 차분 예제

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# getting drug sales dataset

file_path= 'https://raw.githubusercontent.com/selva86/datasets/master/a10.csv' 

df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
df.head()
```

```python
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
fig = plot_acf(df)
fig2 = plot_pacf(df, method="ywm")
```

```python
# Variance Stabilizing Transformation(VST) by Taking Logarithm
# 로그를 취해주면서 변동폭 축소
df_vst = np.log(df.value)
df_vst.head()
```

```python
# De-trend by Differencing
# 2차 차분의 경우 = Delta2_Z(t) = Z(t) - Z(t 1) -(Z(t 1) - Z(t 2)) = Z(t) - 2Z(t 1) + Z(t 2)
# 1차 차분에 대해 다시 차분
# 차분을 통해 추세제거
df_vst_diff1 = df_vst.diff(1).dropna()
df_vst_diff1.head()
```

```python
# Stationary Process : De-seasonality by Seasonal Differencing
# 계절요인 제거를 위해 1년단위 차분
df_vst_diff1_diff12 = df_vst_diff1.diff(12).dropna()    # 1년 12개월 단위
fig, axes = plt.subplots(2,2, figsize=(12,10))
```

```python
axes[0,0].set_title("Non-Stationary Process : \nIncreasing Variance + Trend + Seasonality", fontsize=16)
axes[0,0].plot(df)

axes[0,1].set_title("Variance Stabilizing Transformation \nby Taking Logarithm", fontsize=16)
axes[0,1].plot(df_vst)

axes[1,1].set_title("De-trend by 1st order Differencing", fontsize=16)
axes[1,1].plot(df_vst_diff1)

axes[1,0].set_title("De-seasonality by Seasonal Differencing", fontsize=16)
axes[1,0].plot(df_vst_diff1_diff12)


plt.show()
```

</div>
</details>

</br>

## 시계열 모형 종류
1. AR(p) - 자기 회귀 모형
2. MA(q) - 이동평균 모형
3. ARMA(p,q)
4. ARIMA(p,d,q) - 자기회귀누적이동평균 모형 : 차수의 개수(d)는 거의 2를 넘지 않음
5. SARIMA(Seasonal ARIMA) - 계절 자기회귀이동평균 모형
#### AR 모형 실습
* statsmodels.tsa.arima_process.ArmaProcess
* class statsmodels.tsa.arima_process.ArmaProcess(ar=None, ma=None, nobs=100)

<details>
<summary>Parameter 펼치기/접기</summary>
<div markdown="1">

**ar** : [array_like](https://numpy.org/doc/stable/glossary.html#term-array_like "(in NumPy v1.22)")

Coefficient for autoregressive lag polynomial, including zero lag. Must be entered using the signs from the lag polynomial representation. See the notes for more information about the sign.

**ma** : [array_like](https://numpy.org/doc/stable/glossary.html#term-array_like "(in NumPy v1.22)")

Coefficient for moving-average lag polynomial, including zero lag.

**nobs** : [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.10)"),  `optional`

Length of simulated time series. Used, for example, if a sample is generated. See example.

</div>
</details>

</br>

<details>
<summary>예제 코드 펼치기/접기</summary>
<div markdown="1">
  
```python
# AR 모형 실습
from statsmodels.tsa.arima_process import ArmaProcess

# ArmaProcess로 모형 생성하고 nobs 만큼 샘플 생성
def gen_arma_samples(ar,ma,nobs) :
    arma_model = ArmaProcess(ar=ar, ma=ma) # 모형 정의
    arma_samples = arma_model.generate_sample(nobs) # 샘플 생성
    return arma_samples

# drift가 있는 모형은 ArmaProcess에서 처리가 안 되어서 수동으로 정의해줘야 함
# drift → 절편 존재
def gen_random_walk_w_drift(nobs, drift) :
    init = np.random.normal(size=1, loc=0)
    e = np.random.normal(size=nobs, scale=1)
    y = np.zeros(nobs)
    y[0] = init
    for t in (1,nobs) :
        y[t] = drift + 1*y[t-1] + e[t]
        return y
```

```python
# 백색 잡음 모형, 임의 보행 모형, 표류가 있는 임의 보행 모형,
# 정상성을 만족하는 pi=0.9인 AR(1)모형을 각각 250개씩 샘플을 생성하여 그림
np.random.seed(12345)

white_noise = gen_arma_samples(ar=[1], ma=[1], nobs=250)
# y_t = epsilon_t
# y_{t-1} = 0  /  pi = 0, c = 0
random_walk = gen_arma_samples(ar=[1,-1], ma=[1], nobs=250)
# (1-L)y_t = epsilon_t
# y_t = Ly_{t-1}  /  타임랙 적용  /  ∴ (1-L)y_t = epsilon_t
random_walk_w_drift = gen_random_walk_w_drift(250,2)
# y_t = 2 + y_{t-1} + epsilon_t
# c, 즉 드리프트 적용함
stationary_ar_1 = gen_arma_samples(ar=[1,-0.9], ma=[1], nobs=250)
# (1 - 0,9L)y_t = epsilon_t
# -1 < pi < 1 인 y_t = c + pi_1*y_{t-1} + epsilon_t

fig, ax = plt.subplots(1,4)
ax[0].plot(white_noise)
ax[0].set_title("Wihte Noise", fontsize=12)

ax[1].plot(random_walk)
ax[1].set_title("Random Walk", fontsize=12)

ax[2].plot(random_walk_w_drift)
ax[2].set_title("Random Walk with Drift = 3")

ax[3].plot(stationary_ar_1)
ax[3].set_title("Stationary AR(1)")

fig.set_size_inches(16, 4)

plt.show()
```

```python
plot_acf(df_apple_monthly["Close"])
plot_pacf(df_apple_monthly["Close"])
plt.show()
```

```python
diff_1 = df_apple_monthly["Close"].diff(1)
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
```

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df_apple_monthly["Close"], order=(2,0,1), trend="c")
model_fit = model.fit()
print(model_fit.summary())
```

```python
model_fit.plot_diagnostics().tight_layout()
```

```python
model_fit.predict().tail(30).plot()
df_apple_monthly["Close_3Month_Mean"].tail(30).plot()
plt.legend()
```
</div>
</details>

</br>

시계열 모형 종류
===
AR - 자기회귀 모형
---
* AR(Autoregressive) 모델은 자기회귀 모델로 자기상관성을 시계열 모델로 구성한 것.
* 예측하고 싶은 특정 변수의 과거 자신의 데이터와 선형 결합을 통해 특정 시점 이후 미래값을 예측하는 모델.
* 이름 그대로 이전 자신의 데이터가 이후 자신의 미래 관측값에 영향을 끼친다는 것을 기반으로 나온 모델.
* AR(1) 에 적용하기 위해선 $1<ϕ_1<1$ 조건 이 필요.

### statsmodels의 ArmaProcess
* statsmodels.tsa.arima_process.ArmaProcess
<details>
<summary> Parameters 펼치기/접기</summary>
<div markdown="1">

**ar** : [array_like](https://numpy.org/doc/stable/glossary.html#term-array_like "(in NumPy v1.22)")

Coefficient for autoregressive lag polynomial, including zero lag. Must be entered using the signs from the lag polynomial representation. See the notes for more information about the sign.

**ma** : [array_like](https://numpy.org/doc/stable/glossary.html#term-array_like "(in NumPy v1.22)")

Coefficient for moving-average lag polynomial, including zero lag.

**nobs** : [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.10)"),  `optional`

Length of simulated time series. Used, for example, if a sample is generated. See example.

</div>
</details>

</br>

### AR 모형 실습

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
# AR 모형 실습
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt

# ArmaProcess로 모형 생성하고 nobs 만큼 샘플 생성
def gen_arma_samples(ar,ma,nobs) :
    arma_model = ArmaProcess(ar=ar, ma=ma) # 모형 정의
    arma_samples = arma_model.generate_sample(nobs) # 샘플 생성
    return arma_samples

# drift가 있는 모형은 ArmaProcess에서 처리가 안 되어서 수동으로 정의해줘야 함
# drift → 절편 존재
def gen_random_walk_w_drift(nobs, drift) :
    init = np.random.normal(size=1, loc=0)
    e = np.random.normal(size=nobs, scale=1)
    y = np.zeros(nobs)
    y[0] = init
    for t in (1,nobs) :
        y[t] = drift + 1*y[t-1] + e[t]
        return y
```
```python
# 백색 잡음 모형, 임의 보행 모형, 표류가 있는 임의 보행 모형,
# 정상성을 만족하는 pi=0.9인 AR(1)모형을 각각 250개씩 샘플을 생성하여 그림
np.random.seed(12345)

white_noise = gen_arma_samples(ar=[1], ma=[1], nobs=250)
# y_t = epsilon_t
# y_{t-1} = 0  /  pi = 0, c = 0
random_walk = gen_arma_samples(ar=[1,-1], ma=[1], nobs=250)
# (1-L)y_t = epsilon_t
# y_t = Ly_{t-1}  /  타임랙 적용  /  ∴ (1-L)y_t = epsilon_t
random_walk_w_drift = gen_random_walk_w_drift(250,2)
# y_t = 2 + y_{t-1} + epsilon_t
# c, 즉 드리프트 적용함
stationary_ar_1 = gen_arma_samples(ar=[1,-0.9], ma=[1], nobs=250)
# (1 - 0,9L)y_t = epsilon_t
# -1 < pi < 1 인 y_t = c + pi_1*y_{t-1} + epsilon_t

fig, ax = plt.subplots(1,4)
ax[0].plot(white_noise)
ax[0].set_title("Wihte Noise", fontsize=12)

ax[1].plot(random_walk)
ax[1].set_title("Random Walk", fontsize=12)

ax[2].plot(random_walk_w_drift)
ax[2].set_title("Random Walk with Drift = 2")

ax[3].plot(stationary_ar_1)
ax[3].set_title("Stationary AR(1)")

fig.set_size_inches(16, 4)

plt.show()
```

</div>
</details>

</br>

MA - 이동평균모형
---
* 회귀에서 목표 예상 변수(forecast variable)의 과거 값을 이용하는 대신에, 이동 평균 모델은 회귀처럼 보이는 모델에서 과거 예측 오차(forecast error)을 이용함.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$y_t=c+ε_t+θ_1ε_{t−1}+θ_2ε_{t−2}+⋯+θ_qε_{t−q}$,    
여기에서 $ε_t$ 는 백색잡음. $q$차 이동 평균 모델인 MA($q$) 모델. 물론, εt 의 값을 관찰하지 않기 때문에, 이것은 실제로는 보통 생각하는 회귀가 아님.
* $θ_1$의 값에 따라 다른 시계열 패턴이 나타남.

### statsmodles를 이용한 MA 모형 구현
* ArmaProcess(ar=[1] ma=[$1, \theta_1, \theta_2, ..., \theta_q$]) 로 생성

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
np.random.seed(12345)

ma_1 = gen_arma_samples(ar = [1], ma = [1, 1], nobs = 250)
# y_t = (1+L)epsilon_t
ma_2 = gen_arma_samples(ar = [1], ma = [1, 0.5], nobs = 250)
# y_t = (1+0.5L)epsilon_t
ma_3 = gen_arma_samples(ar = [1], ma = [1, -2], nobs = 250)
# y_t = (1-2L)epsilon_t

fig, ax = plt.subplots(1, 3, figsize=(12,4))

ax[0].plot(ma_1)
ax[0].set_title("MA(1) with thetha_1 = 1")
# MA(1) 모형의 theta 값만 변화

ax[1].plot(ma_2)
ax[1].set_title("MA(1) with thetha_1 = 0.5")

ax[2].plot(ma_3)
ax[2].set_title("MA(1) withe thetha_1 = -2")

plt.show()
```

</div>
</details>

</br>

### AR 모형과 MA 모형을 언제 사용해야 하는지
<img src="C:\Users\user\Desktop\Vocational_Training\FinTech\Identification ARIMA model.jpg" width="40%">
ARIMA 모형
---
* Auto-regressive Integrated Moving Average Model
* AR과 MA모형을 결합한 ARMA 모형을 일반화한 것.
* ARMA 모델이 안정적 시계열(Stationary Series)에만 적용 가능한 것에 비해, 분석 대상이 다소 비안정적인 시계열(Non Stationary Series)의 특징을 보여도 적용이 가능.
* 안정적인 시계열이란 시간의 추이와 관계없이 평균 및 분산이 불변하거나 시점 간의 공분산이 기준시점과 무관한 형태의 시계열.    
시계열이 안정적이지 않을 때는 로그를 이용하거나 차분을 통해 시계열을 안정적으로 변환한 뒤에 분석을 진행.
* $X_t−ϕ_1X_{t−1}−⋯−ϕ_pX_{t−p}=Z_t+θ_1Z_{t−1}+⋯+θ_qZ_{t−q},  (t=0,±1,±2,⋯)$
* $X_t$은 ARIMA를 통해 예측하고자 하는 데이터이고 $Z_t$은 백색잡음(White Noise)으로, 모두 독립적이고 동일하게 분산된(IID) 확률 변수.

### pmdarima의 auto_arima 라이브러리
* pmdarima.arima.auto_arima(y, X=None, start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, max_order=5, m=1, seasonal=True, stationary=False, information_criterion='aic', alpha=0.05, test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1, start_params=None, trend=None, method='lbfgs', maxiter=50, offset_test_args=None, seasonal_test_args=None, suppress_warnings=True, error_action='trace', trace=False, random=False, random_state=None, n_fits=10, return_valid_fits=False, out_of_sample_size=0, scoring='mse', scoring_args=None, with_intercept='auto', sarimax_kwargs=None, **fit_args)

<details>
<summary> Parameters 펼치기/접기</summary>
<div markdown="1">

**y**  : array-like or iterable, shape=(n_samples,)

> The time-series to which to fit the  `ARIMA`  estimator. This may either be a Pandas  `Series`  object (statsmodels can internally use the dates in the index), or a numpy array. This should be a one-dimensional array of floats, and should not contain any  `np.nan`  or  `np.inf`  values.

**X**  : array-like, shape=[n_obs, n_vars], optional (default=None)

> An optional 2-d array of exogenous variables. If provided, these variables are used as additional features in the regression operation. This should not include a constant or trend. Note that if an  `ARIMA`  is fit on exogenous features, it must be provided exogenous features for making predictions.

**start_p**  : int, optional (default=2)

> The starting value of  `p`, the order (or number of time lags) of the auto-regressive (“AR”) model. Must be a positive integer.

**d**  : int, optional (default=None)

> The order of first-differencing. If None (by default), the value will automatically be selected based on the results of the  `test`  (i.e., either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or the Phillips–Perron test will be conducted to find the most probable value). Must be a positive integer or None. Note that if  `d`  is None, the runtime could be significantly longer.

**start_q**  : int, optional (default=2)

> The starting value of  `q`, the order of the moving-average (“MA”) model. Must be a positive integer.

**max_p**  : int, optional (default=5)

> The maximum value of  `p`, inclusive. Must be a positive integer greater than or equal to  `start_p`.

**max_d**  : int, optional (default=2)

> The maximum value of  `d`, or the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to  `d`.

**max_q**  : int, optional (default=5)

> The maximum value of  `q`, inclusive. Must be a positive integer greater than  `start_q`.

**start_P**  : int, optional (default=1)

> The starting value of  `P`, the order of the auto-regressive portion of the seasonal model.

**D**  : int, optional (default=None)

> The order of the seasonal differencing. If None (by default, the value will automatically be selected based on the results of the  `seasonal_test`. Must be a positive integer or None.

**start_Q**  : int, optional (default=1)

> The starting value of  `Q`, the order of the moving-average portion of the seasonal model.

**max_P**  : int, optional (default=2)

> The maximum value of  `P`, inclusive. Must be a positive integer greater than  `start_P`.

**max_D**  : int, optional (default=1)

> The maximum value of  `D`. Must be a positive integer greater than  `D`.

**max_Q**  : int, optional (default=2)

> The maximum value of  `Q`, inclusive. Must be a positive integer greater than  `start_Q`.

**max_order**  : int, optional (default=5)

> Maximum value of p+q+P+Q if model selection is not stepwise. If the sum of  `p`  and  `q`  is >=  `max_order`, a model will  _not_  be fit with those parameters, but will progress to the next combination. Default is 5. If  `max_order`  is None, it means there are no constraints on maximum order.

**m**  : int, optional (default=1)

> The period for seasonal differencing,  `m`  refers to the number of periods in each season. For example,  `m`  is 4 for quarterly data, 12 for monthly data, or 1 for annual (non-seasonal) data. Default is 1. Note that if  `m`  == 1 (i.e., is non-seasonal),  `seasonal`  will be set to False. For more information on setting this parameter, see  [Setting m](https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period).

**seasonal**  : bool, optional (default=True)

> Whether to fit a seasonal ARIMA. Default is True. Note that if  `seasonal`  is True and  `m`  == 1,  `seasonal`  will be set to False.

**stationary**  : bool, optional (default=False)

> Whether the time-series is stationary and  `d`  should be set to zero.

**information_criterion**  : str, optional (default=’aic’)

> The information criterion used to select the best ARIMA model. One of  `pmdarima.arima.auto_arima.VALID_CRITERIA`, (‘aic’, ‘bic’, ‘hqic’, ‘oob’).

**alpha**  : float, optional (default=0.05)

> Level of the test for testing significance.

**test**  : str, optional (default=’kpss’)

> Type of unit root test to use in order to detect stationarity if  `stationary`  is False and  `d`  is None. Default is ‘kpss’ (Kwiatkowski–Phillips–Schmidt–Shin).

**seasonal_test**  : str, optional (default=’ocsb’)

> This determines which seasonal unit root test is used if  `seasonal`  is True and  `D`  is None. Default is ‘OCSB’.

**stepwise**  : bool, optional (default=True)

> Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters. The stepwise algorithm can be significantly faster than fitting all (or a  `random`  subset of) hyper-parameter combinations and is less likely to over-fit the model.

**n_jobs**  : int, optional (default=1)

> The number of models to fit in parallel in the case of a grid search (`stepwise=False`). Default is 1, but -1 can be used to designate “as many as possible”.

**start_params**  : array-like, optional (default=None)

> Starting parameters for  `ARMA(p,q)`. If None, the default is given by  `ARMA._fit_start_params`.

**method**  : str, optional (default=’lbfgs’)

> The  `method`  determines which solver from  `scipy.optimize`  is used, and it can be chosen from among the following strings:
> 
> -   ‘newton’ for Newton-Raphson
> -   ‘nm’ for Nelder-Mead
> -   ‘bfgs’ for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
> -   ‘lbfgs’ for limited-memory BFGS with optional box constraints
> -   ‘powell’ for modified Powell’s method
> -   ‘cg’ for conjugate gradient
> -   ‘ncg’ for Newton-conjugate gradient
> -   ‘basinhopping’ for global basin-hopping solver
> 
> The explicit arguments in  `fit`  are passed to the solver, with the exception of the basin-hopping solver. Each solver has several optional arguments that are not the same across solvers. These can be passed as  [**](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#id1)fit_kwargs

**trend**  : str or None, optional (default=None)

> The trend parameter. If  `with_intercept`  is True,  `trend`  will be used. If  `with_intercept`  is False, the trend will be set to a no- intercept value.

**maxiter**  : int, optional (default=50)

> The maximum number of function evaluations. Default is 50.

**offset_test_args**  : dict, optional (default=None)

> The args to pass to the constructor of the offset (`d`) test. See  `pmdarima.arima.stationarity`  for more details.

**seasonal_test_args**  : dict, optional (default=None)

> The args to pass to the constructor of the seasonal offset (`D`) test. See  `pmdarima.arima.seasonality`  for more details.

**suppress_warnings**  : bool, optional (default=True)

> Many warnings might be thrown inside of statsmodels. If  `suppress_warnings`  is True, all of the warnings coming from  `ARIMA`  will be squelched. Note that this will not suppress UserWarnings created by bad argument combinations.

**error_action**  : str, optional (default=’warn’)

> If unable to fit an  `ARIMA`  for whatever reason, this controls the error-handling behavior. Model fits can fail for linear algebra errors, convergence errors, or any number of problems related to stationarity or input data.
> 
> > -   ‘warn’: Warns when an error is encountered (default)
> > -   ‘raise’: Raises when an error is encountered
> > -   ‘ignore’: Ignores errors (not recommended)
> > -   ‘trace’: Logs the entire error stacktrace and continues the
> >     
> >     search. This is the best option when trying to determine why a model is failing.
> >     

**trace**  : bool or int, optional (default=False)

> Whether to print status on the fits. A value of False will print no debugging information. A value of True will print some. Integer values exceeding 1 will print increasing amounts of debug information at each fit.

**random**  : bool, optional (default=False)

> Similar to grid searches,  `auto_arima`  provides the capability to perform a “random search” over a hyper-parameter space. If  `random`  is True, rather than perform an exhaustive search or  `stepwise`  search, only  `n_fits`  ARIMA models will be fit (`stepwise`  must be False for this option to do anything).

**random_state**  : int, long or numpy  `RandomState`, optional (default=None)

> The PRNG for when  `random=True`. Ensures replicable testing and results.

**n_fits**  : int, optional (default=10)

> If  `random`  is True and a “random search” is going to be performed,  `n_iter`  is the number of ARIMA models to be fit.

**return_valid_fits**  : bool, optional (default=False)

> If True, will return all valid ARIMA fits in a list. If False (by default), will only return the best fit.

**out_of_sample_size**  : int, optional (default=0)

> The  `ARIMA`  class can fit only a portion of the data if specified, in order to retain an “out of bag” sample score. This is the number of examples from the tail of the time series to hold out and use as validation examples. The model will not be fit on these samples, but the observations will be added into the model’s  `endog`  and  `exog`  arrays so that future forecast values originate from the end of the endogenous vector.
> 
> For instance:
> 
> y = [0, 1, 2, 3, 4, 5, 6]
> out_of_sample_size = 2
> 
> > Fit on: [0, 1, 2, 3, 4]
> > Score on: [5, 6]
> > Append [5, 6] to end of self.arima_res_.data.endog values

**scoring**  : str, optional (default=’mse’)

> If performing validation (i.e., if  `out_of_sample_size`  > 0), the metric to use for scoring the out-of-sample data. One of (‘mse’, ‘mae’)

**scoring_args**  : dict, optional (default=None)

> A dictionary of key-word arguments to be passed to the  `scoring`  metric.

**with_intercept**  : bool or str, optional (default=”auto”)

> Whether to include an intercept term. Default is “auto” which behaves like True until a point in the search where the sum of differencing terms will explicitly set it to True or False.

**sarimax_kwargs**  : dict or None, optional (default=None)

> Keyword arguments to pass to the ARIMA constructor.

****fit_args**  : dict, optional (default=None)

> A dictionary of keyword arguments to pass to the  [`ARIMA.fit()`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.fit "pmdarima.arima.ARIMA.fit")  method.

</div>
</details>

</br>

### 실습

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import ndiffs
import pmdarima as pm
```
```python
df_krx = fdr.StockListing("KRX")
symbol = df_krx[df_krx["Name"] == "삼성전자"]["Symbol"]
SSEL = fdr.DataReader(''.join(symbol.values), '2020-01-01', "2021-12-31")
plt.plot(SSEL["Close"])
```
```python
# 학습 데이터와 테스트 데이터 분할
y_train = SSEL["Close"][:int(0.7*len(SSEL))]
y_test = SSEL["Close"][int(0.7*len(SSEL)):]
y_train.plot()
y_test.plot()
```
```python
# 차분 차수 찾는 라이브러리
kpss_diffs = ndiffs(y_train, alpha=0.05, test="kpss", max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test="adf", max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"적정 차분 차수 : {n_diffs}")
```
```python
# 최적모형 탐색
model = pm.auto_arima(y=y_train,
            d =1,
            start_p=0,
            max_p=3,
            start_q=0,
            max_q=3,
            seasonal=False,
            stepwise=True,
            trace=True)
```
```python
# 모형에 데이터 학습
model.fit(y_train)
```
```python
# 예측
y_pred = model.predict(n_periods=len(y_test))
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=["Prediction"])
y_pred_df
```
```python
# 시각화
fig, axs = plt.subplots(figsize=(12,4))

plt.plot(y_train, label="Train")

plt.plot(y_test, label="Test")

plt.plot(y_pred_df, label="Prediction")

plt.legend()
plt.show()
```
```python
# 함수 설정
def forecast_one_step() :
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)

    return (fc.tolist()[0], np.asarray(conf_int).tolist()[0])
```
```python
forecast_one_step()
```
```python
# y_test 내의 데이터 하나하나씩 예측하여 정확도 높여줌
forecast_list = []
y_pred_list = []
pred_upper = [] 
pred_lower = [] 

for i in y_test :
    fc, conf = forecast_one_step()
    y_pred_list.append(fc)      # 예측치
    pred_upper.append(conf[1])  # 신뢰구간 상방
    pred_lower.append(conf[0])  # 신뢰구간 하방

    model.update(i)

y_pred_list
```
```python
# 시각화
pred_dt = pd.DataFrame({"test" : y_test, "pred" : y_pred_list})
pred_dt.plot(figsize=(12,8))
```
### plotly를 활용한 시각화
```python
import plotly.graph_objects as go

fig = go.Figure([
    go.Scatter(x=y_train.index, y=y_train, name="Train", mode="lines", line=dict(color="royalblue")),
    # 테스트데이터
    go.Scatter(x=y_test.index, y=y_test, name="Test", mode="lines", line=dict(color="rgba(0,0,30,0.5)")),
    # 예측 데이터
    go.Scatter(x=y_test.index, y=y_pred_list, name="Pred", mode="lines", line=dict(color="red", dash="dot", width=3)),
    # 신뢰구간
    go.Scatter(x=y_test.index.tolist() + y_test.index[::-1].tolist(),
                y=pred_upper + pred_lower[::-1],
                fillcolor="rgba(0,0,30,0.2)",
                fill="toself",
                line={"color":"rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=True)
])

fig.update_layout(height=400, width=1000, title_text="ARIMA(0,1,0)")
fig.show()
```
```python
import plotly.express as px

fig = px.line(pred_dt)
fig.show()
```

</div>
</details>

</br>

데이터 전처리
===
(정규화, 로그 변환, 스케일러, 원-핫 인코딩)
---
* 선형회귀 모델을 위한 데이터 변환
    - 회귀 모델과 같은 선형 모델은 일반적으로 피처와 타깃값 간에 선형의 관계가 있다고 가정하고, 이러한 최적의 선형함수를 찾아내 결과를 예측함.
    - 또한 선형 회귀 모델은 피처값과 타깃값의 분포가 정규 분포 형태인 경우를 매우 선호함.
### 로그 변환, 스케일러, 다항 특성 적용

<table>
 <tr>
    <th>변환대상</th>
    <th>설명</th>
  </tr>
  <tr>
    <td>타깃값 변환</td>
    <td>회귀에서 타깃값은 반드시 정규 분포를 가져야 함.</br>
    이를 위해 주로 로그변환을 적용.</td>
  </tr>
  <tr>
    <td rowspan="3">피처값 변환</td>
    <td>StandardScaler : 평균이 0, 분산이 1인 표준정규분포를 가진 데이터 세트로 변환</br>
    MinMaxScaler : 최솟값이 0, 최댓값이 1인 값으로 정규화를 수행
    </td>
  </tr>
  <tr>
    <td>스케일링/정규화를 수행한 데이터 세트에 다시 다항 특성을 적용하여 변환.</br>
    보통 1번 방법을 통해 예측 성능에 향상이 없을 경우 이와 같은 방법을 적용.
    </td>  
  </tr>
  <tr>
    <td>원래 값에 log 함수를 적용하면 보다 정규 분포에 가까운 형태로 값이 분포됨. 로그 변환은 매우 유용한 변환이며, 실제로 선형 회귀에서는 앞의 1, 2번 방법보다 로그 변환이 훨씬 많이 사용되는 변환 방법.</br>
    왜냐하면 1번 방법의 경우 예측 성능 향상을 크게 기대하기 어려운 경우가 많으며, 2번 방법의 경우 피처의 개수가 매우 많을 경우에는 다항 변환으로 생성되는 피처의 개수가 기하급수적으로 늘어나 과적합 이슈가 발생할 수 있기 때문.
    </td>
  </tr>
</table>

### 인코딩
* 선형 회귀의 데이터 인코딩은 일반적으로 레이블 인코딩이 아닌 원-핫 인코딩을 적용함.
* 레이블 인코딩 : 카테고리 별로 1, 2, 3, 4, ...
* 원-핫 인코딩 : 0과 1로 구성된 행렬 형태

#### 실습 : 피처 데이터 변환에 따른 예측 성능 비교

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# 데이터 설정
house_price = load_boston()
X_data = house_price.data
y_target = house_price.target

print(y_target.shape)
plt.hist(y_target, bins=10)
```
```python
# method는 표준 정규 분포 변환(Standard), 최대값/최소값 정규화(MinMax), 로그변환(Log) 결정
# p_degree는 다항식 특성을 추가할 때 적용, p_degree는 2이상 부여하지 않음.
def get_scaled_data(method="None", p_degree=None, input_data=None) :
    if method == "Standard" :
        scaled_data = StandardScaler().fit_transform(input_data)
        # 정규분포화
    elif method == "MinMax" :
        scaled_data = MinMaxScaler().fit_transform(input_data)
        # 최대최소
    elif method == "Log" :
        scaled_data = np.log1p(input_data)
        # 로그변환
    else :
        scaled_data = input_data
        # 메소드 입력 안 했을 경우 그대로 저장.

    if p_degree != None :
        scaled_data = PolynomialFeatures(
            degree=p_degree, include_bias=False).fit_transform(scaled_data)
        # p_degree 있는 경우 Polynomial 진행
    
    return scaled_data
```
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd

data = []

def get_linear_reg_eval(method="Ridge", params=[], X_data_n=None, y_target_n=None, verbose=True) :
    sub_data = []
    for param in params :        
        ridge = Ridge(alpha=param)
        neg_mse_scores = cross_val_score(ridge, X_data_n, y_target_n, scoring="neg_mean_squared_error", cv=5, verbose=verbose)
        rmse_scores = np.sqrt(-1 * neg_mse_scores)
        avg_rmse = np.mean(rmse_scores)
        print(f"alpha {param}일 때 5 folds의 개별 평균 RMSE : {avg_rmse:.4f}")
        sub_data.append(np.round(avg_rmse,3))
    data.append(sub_data)
```
```python
# Ridge의 alpha값을 다르게 적용하고 다양한 데이터 변환방법에 따른 RMSE 추출.
alphas = [0.1, 1, 10, 100]

# 변환 방법은 모두 6개, 원본 그대로, 표준정규분포, 표준정규분포+다항식 특성
# 최대/최소 정규화, 최대/최소 정규화+다항식 특성, 로그변환
scale_methods=[(None, None), ("Standard", None), ("Standard", 2),
                ("MinMax", None), ("MinMax", 2), ("Log", None)]

for scale_method in scale_methods :
    X_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1],
                                    input_data=X_data)
    print(f"\n## 변환 유형:{scale_method[0]}, Polynomial Degree:{scale_method[1]}")

    # alpha 값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고,
    # 회귀 계수값들을 DataFrame으로 반환해주는 함수
    get_linear_reg_eval("Ridge", params=alphas, X_data_n=X_data_scaled,
                        y_target_n=y_target, verbose=False)
```
```python
# 결과 값 데이터 프레임 확인
result_df = pd.DataFrame(data, columns=[["alpha값", "alpha값", "alpha값", "alpha값"], ["alpha=0.1", "alpha=1", "alpha=10", "alpha=100"]])
result_df["변환유형"] = ["원본데이터", "표준정규분포", "표준정규분포 + 2차 다항식", "최솟값/최댓값 정규화", "최솟값/최댓값 정규화 + 2차 다항식", "로그변환"]
result_df.set_index("변환유형", inplace=True)
result_df
```

</div>
</details>

</br>

## 로지스틱 회귀
### 로지스틱 회귀 개요
* 로지스틱 회귀는 선형 회귀 방식을 분류에 적용한 알고리즘. 즉, 로지스틱 회귀는 분류에 사용됨.       
로지스틱 회귀가 선형 회귀와 다른 점은 학습을 통해 선형 함수의 회귀 최적선을 찾는 것이 아니라, 시그모이드(Sigmoid)함수 최적선을 찾고, 이 시그모이드 함수의 반환 값을 확률로 간주해 확률에 따라 분류를 결정함.
* 로지스틱 회귀는 주로 이진분류에 사용됨(다중 클래스 분류에도 적용 가능함).     
  로지스틱 회귀에서 예측 값은 예측 확률을 의미하며, 예측 확률이 0.5 이상이면 1로, 0.5 이하이면 0으로 예측함. 로지스틱 회귀의 예측 확률은 시그모이드 함수의 출력값으로 계산됨.   
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/functions_graph.png" width="40%">

#### 로지스틱 회귀 예측

* 시그모이드 함수 : $y = \frac{1}{1+e^{-x}}$
* 단순 선형회귀 : $y=w_1x + w_0$가 있다고 할 때,    
  로지스틱 회귀는 0과 1을 예측하기에 단순 회귀식은 의미가 없음.    
  하지만 Odds(성공확률/실패확률)을 통해 선형 회귀식에 확률을 적용할 수 있음.
  $$Odds(p) = \frac{p}{1-p}$$ 
  하지만 확률p의 범위가 (0,1)이므로 선형 회귀의 반환값인 $(-\infty, +\infty)$에 대응하기 위해 로그 변환을 수행하고 이 값에 대해 선형 회귀를 적용함.    
  $$\log{Odds(p)} = w_1x + w_0$$
  해당 식을 데이터 값 x의 확률 p로 정리하면 아래와 같음
  $$p(x)=\frac{1}{1+e^{-(w_1x+w_0)}}$$
  로지스틱 회귀는 학습을 통해 시그모이드 함수의 w를 최적화하여 예측하는 것.

#### 시그모이드를 이용한 로지스틱 회귀 예측

<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/Sigmoid.png" width="40%">

* 로지스틱 회귀는 가볍고 빠르지만, 이진 분류 예측 성능도 뛰어남. 이 때문에 로지스틱 회귀를 이진 분류의 기본 모델로 사용하는 경우가 많음. 또한 로지스틱 회귀는 희소한 데이터 세트 분류에도 뛰어난 성능을 보여서 텍스트 분류에서도 자주 사용됨.
* 사이킷런은 LogisticRegression 클래스로 로지스틱 회귀를 구현함. 주요 하이퍼 파라미터로 penalty와 C가 있음. Penalty는 Regularization의 유형을 설정함. 'l2'로 설정시 L2 규제 등. default는 'l2'. C는 규제 강도를 조절하는 alpha값의 역수. C값이 작을수록 규제 강도가 큼.   
$$C=\frac{1}{\alpha}$$


  #### 실습 : 로지스틱 회귀

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# 위스콘신 유방암 데이터 불러오기
cancer = load_breast_cancer()
```
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScaler()로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, cancer.target, test_size=0.3, random_state=0
)
```
```python
from sklearn.metrics import accuracy_score, roc_auc_score

# 로지스틱 회귀를 이용하여 학습 및 예측 수행

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

lr_preds = lr_clf.predict(X_test)

# accuracy와 roc_auc 측정
print(f"accuracy {accuracy_score(y_test, lr_preds):.3f}")
print(f"roc_auc {roc_auc_score(y_test, lr_preds):.3f}")
```
```python
from sklearn.model_selection import GridSearchCV

params = {
    "penalty" : ["l2", "l1"],
    "C" :[0.01, 0.1, 1, 1, 5, 10]
}

grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring="accuracy", cv=3)
grid_clf.fit(data_scaled, cancer.target)
print(f"최적 하이퍼 파라미터 : {grid_clf.best_params_}, 최적 평균 정확도 : {grid_clf.best_score_:.3f}")
```

</div>
</details>

</br>

## 회귀 트리
### 회귀 트리 개요
* 회귀 트리 : 트리 기반의 회귀 방식
* 사이킷런의 결정 트리 및 결정 트리 기반의 앙상블 알고리즘 분류 뿐만 아니라 회귀도 가능.
* 트리가 CART(Classification And Regression Trees)를 기반으로 만들어졌기 때문.   
 CART는 분류 뿐만 아니라 회귀도 가능한 트리 분할 알고리즘.
* CART 회귀 트리는 분류와 유사하게 분할을 하며, 분할 기준은 RSS(SSE)가 최소가 될 수 있는 기준을 찾아 분할됨.
* 최종 분할이 완료된 후에 각 분할 영역에 있는 데이터 결정값들의 평균 값으로 학습/예측함.   
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/CART.png" width="40%">

### 회귀 트리의 오버피팅(과대적합)
* 회귀 트리 역시 복잡한 트리 구조를 가질 경우 오버피팅하기 쉬움. 트리의 크기와 노드 개수의 제한 등의 방법을 통해 오버 피팅을 개선 할 수 있음.   
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/CART_overfitting.png" width="40%">

#### 실습 : 회귀 트리로 보스턴 집값 예측

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 데이터 불러오기
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# 보스턴 데이터 세트 로드
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)

# 랜덤 포레스트로 교차 검증
bostonDF["PRICE"] = boston.target
y_target = bostonDF["PRICE"]
X_data = bostonDF.drop(["PRICE"], axis=1, inplace=False)

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(f"5 교차 검증의 개별 Negative MSE scores : {np.round(neg_mse_scores,2)}")
print(f"5 교차 검증의 개별 RMSE scores : {np.round(rmse_scores,2)}")
print(f"5 교차 검증의 평균 RMSE : {np.round(avg_rmse,3)}")
```
```python
# RMSE 측정 함수
def get_model_cv_prediction(model=None, X_data=None, y_target=None) :
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    if model == dt_reg :
        print(f"### DecisionTreeRegressor ### \n5 교차 검증의 평균 RMSE : {np.round(avg_rmse,3)}")
    elif model == rf_reg :
        print(f"### RandomForestRegressor ### \n5 교차 검증의 평균 RMSE : {np.round(avg_rmse,3)}")
    else :
        print(f"### XGBRegressor ### \n5 교차 검증의 평균 RMSE : {np.round(avg_rmse,3)}")
```
```python
# 3개 회귀 트리 모델로 회귀 수행

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# max_depth=4짜리 디시전 트리
dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
# n_estimators=1000짜리 랜덤 포레스트
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
# n_estimators=1000짜리 XGBoost
xg_reg = XGBRegressor(num_round=1000, verbosity=0)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행
models = [dt_reg, rf_reg, xg_reg]
for model in models :
    get_model_cv_prediction(model, X_data, y_target)
```
##### 회귀 트리의 피처 중요도 파악
```python
# 회귀 트리는 feature_importances_로 피처 중요도를 파악한다.
# 선형회귀의 회귀 계수 역할을 함.

import seaborn as sns

# 랜덤 포레스트 Reg
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)

# 앞 예제에서 만들어진 X_data, y_target 데이터 셋을 적용하여 학습함.
rf_reg.fit(X_data, y_target)

# 주요 피처 시각화
feature_series = pd.Series(data=rf_reg.feature_importances_, index=X_data.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
```
```python
import seaborn as sns

# XGB Reg
xg_reg = XGBRegressor(num_round=1000, verbosity=0)

# 앞 예제에서 만들어진 X_data, y_target 데이터 셋을 적용하여 학습함.
xg_reg.fit(X_data, y_target)

# 주요 피처 시각화
feature_series = pd.Series(data=xg_reg.feature_importances_, index=X_data.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
```
```python
# 위에서 선별한 주요 피처만 불러온 데이터 생성
impt_ftrs_rf = ["RM", "LSTAT", "DIS", "CRIM", "NOX"]
impt_ftrs_xg = ["LSTAT", "RM", "NOX", "DIS", "PTRATIO"]

X_data_rf = X_data[impt_ftrs_rf]
X_data_xg = X_data[impt_ftrs_xg]
```
```python
# max_depth=4짜리 디시전 트리
dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
# n_estimators=1000짜리 랜덤 포레스트
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
# n_estimators=1000짜리 XGBoost
xg_reg = XGBRegressor(num_round=1000, verbosity=0)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행
# rf에서 뽑힌 주요 피처 상위 5개 항목 평가
models = [dt_reg, rf_reg, xg_reg]
for model in models :
    get_model_cv_prediction(model, X_data_rf, y_target)
```
```python
# xgb에서 뽑힌 주요 피처 상위 5개 항목 평가
models = [dt_reg, rf_reg, xg_reg]
for model in models :
    get_model_cv_prediction(model, X_data_xg, y_target)

# 예상과 달리 XGB에서 중요 피처의 RMSE 값이 떨어짐.
```
##### 회귀 트리의 오버 피팅 시각화
```python
import matplotlib.pyplot as plt

bostonDF_sample = bostonDF[["RM", "PRICE"]]
bostonDF_sample = bostonDF_sample.sample(n=100, random_state=0)
print(bostonDF_sample.shape)
plt.figure()
plt.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
```
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 선형 회귀와 결정 트리 기반의 Regressor 생성, DecisionTreeRegressor의 max_depth는 각각2, 7
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2)
rf_reg7 = DecisionTreeRegressor(max_depth=7)

# 실제 예측을 적용할 테스트 데이터 셋을 4.5 ~ 8.5까지 100개 데이터 셋 생성
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1)

# 보스턴 주택가격 데이터에서 시각화를 위해 피처는 RM만, 그리고 결정 데이터인 PRICE 추출
X_feature = bostonDF_sample["RM"].values.reshape(-1, 1)
y_target = bostonDF_sample["PRICE"].values.reshape(-1, 1)

# 학습과 예측 수행.
lr_reg.fit(X_feature, y_target)
rf_reg2.fit(X_feature, y_target)
rf_reg7.fit(X_feature, y_target)

pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
pred_rf7 = rf_reg7.predict(X_test)
```
```python
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 4), ncols=3)

# x축 값을 4.5 ~ 8.5로 변환하여 입력했을 때, 선형 회귀와 결정 트리 회귀 예측 선 시각화
# 선형 회귀로 학습된 모델 회귀 예측선
ax1.set_title("Linear Regression")
ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax1.plot(X_test, pred_lr, label="linear", linewidth=2)

# DecisionTreeRegressor의 max_depth를 2로 했을 때 회귀 예측선
ax2.set_title("DecisionTreeRegressor : \n max_depth=2")
ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax2.plot(X_test, pred_rf2, label="max_depth:2", linewidth=2)

# DecisionTreeRegressor의 max_depth를 7로 했을 때 회귀 예측선
ax3.set_title("DecisionTreeRegressor : \n max_depth=7")
ax3.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax3.plot(X_test, pred_rf7, label="max_depth:7", linewidth=2)

# 선형 회귀는 직선으로 예측 회귀선을 표현
# 회귀 트리는 분할되는 데이터 지점에 따라 계단 형태로 회귀선 표현
# DTR의 max_depth=2 인 경우 어느정도 분류가 잘 됨
# max_depth=7인 경우 학습 데이터 세트의 outlier 데이터도 학습하면서
# 복잡한 계단 형태의 회귀선을 만들어 과적합 모델을 만듦.
```


</div>
</details>

</br>

회귀 실습
===
자전거 대여 수요 예측
---

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

### 1. 전처리

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 데이터 로드
bike_df = pd.read_csv("./csv_data/train_bike.csv")

print(bike_df.shape)
bike_df.head()
```
```python
bike_df.info()  #null없음 확인
```
```python
# 문자열을 datetime 타입으로 변경
bike_df["datetime"] = bike_df.datetime.apply(pd.to_datetime)

# datetime 타입에서 년, 월, 일, 시간 추출
bike_df["year"] = bike_df.datetime.apply(lambda x : x.year)
bike_df["month"] = bike_df.datetime.apply(lambda x : x.month)
bike_df["day"] = bike_df.datetime.apply(lambda x : x.day)
bike_df["hour"] = bike_df.datetime.apply(lambda x : x.hour)
bike_df.head()
```
```python
bike_df.info()  # Data type 확인 
```
```python
# 확인 후 불필요한 목록 삭제
drop_columns = ["datetime", "casual", "registered"]
# casual, regi : 사전에 등록되었는지 여부에 따른 대여 횟수
bike_df.drop(drop_columns, axis=1, inplace=True)
```
### 2. 에러 함수들 정의 후 선형회귀 학습/예측
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# log 값 변환 시 NaN 등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSE 계싼
def rmsle(y, pred) :
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_squared_error() 를 이용하여 RMSE로 계산
def rmse(y, pred) :
    return np.sqrt(mean_squared_error(y, pred))

# MAE, RMSE, RMSLE 를 모두 계산
def evaluate_regr(y, pred) : 
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    # MAE는 scikit learn의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y, pred)
    print(f"RMSLE : {rmsle_val:.3f}, RMSE : {rmse_val:.3f}, MAE : {mae_val:.3f}")
```
```python
# 학습 데이터, 테스트 데이터 분리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = bike_df["count"]
X_features = bike_df.drop(["count"], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target,
                                        test_size=0.3, random_state=0)

# 선형회귀 적용 후 학습/예측/평가
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test, pred)
# RMSLE에 비해 RMSE값이 매우 크게 나왔음.
# 예측 에러가 매우 큰 값들이 섞여있기 때문.
```
### 3. 예측값과 실제값 오차 확인
```python
def get_top_error_data(y_test, pred, n_tops = 5) :
    # DataFrame의 컬럼들로 실제 대여횟수(count)와 예측 값을 서로 비교할 수 있도록 생성.
    result_df = pd.DataFrame(y_test.values, columns=["real_count"])
    result_df["predicted_count"] = np.round(pred)
    result_df["diff"] = np.abs(result_df["real_count"] - result_df["predicted_count"])
    # 예측값과 실제값이 가장 큰 데이터 순으로 출력.
    print(result_df.sort_values("diff", ascending=False)[:n_tops])

get_top_error_data(y_test, pred, n_tops=5)
```
### 4. 타겟값에 로그를 취해서 정규화
```python
# 타겟 컬럼인 count 값을 log1p 로 Log 변환
y_target_log = np.log1p(y_target)

# 로그 변환된 y_target_log 를 반영하여 학습/테스트 데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log,
                                    test_size=0.3, random_state=0)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

# 테스트 데이터 셋의 Target 값은 Log 변환되었으므로 다시 expm1을 이용하여 원래 scale로 변환
y_test_exp = np.expm1(y_test)

# 예측 값 역시 Log 변환된 타겟 기반으로 학습되어 예측되었으므로 다시 exmp1으로 scale변환
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)
# 아직도 RMSLE에 비해 RMSE값이 매우 크게 나옴
```
### 5. 피처 별 회귀 계수 확인
```python
coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)

# year(2011, 2012)가 영향력이 큰 것을 볼 수 있음
# → 해당 데이터는 2011년에 창업한 스타트업으로
# 2012년부터 더 성장해 대여 수요량이 늘어난 것.
```
### 6. 원-핫 인코딩 후 다시 학습/예측
```python
# "year", "month", "hour", "season", "weather" feature들을 One Hot Encoding
X_features_ohe = pd.get_dummies(X_features, columns=["year", "month", "hour", "holiday",
                                            "workingday", "season", "weather"])
```
```python
# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log,
                                        test_size=0.3, random_state=0)

# 모델과 학습/테스트 데이터 셋을 입력하면 성능 평가 수치를 반환
# 기본 선형회귀와 릿지, 라쏘 모델에 대해 성능 평가를 해주는 함수
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1 = False) :
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print("###", model.__class__.__name__, "###")
    evaluate_regr(y_test, pred)
# end of function get_model_predict
```
```python
# model 별로 평가 수행
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)

# 이전보다 RMSE가 많이 줄은 것을 볼 수 있음
```
### 7. 원-핫 인코딩 후 회귀 계수 확인
```python
coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:10]
sns.barplot(x=coef_sort.values, y=coef_sort.index)
```
### 8. 회귀 트리 사용
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg] :
    # XGBoost의 경우 DataFrame이 입력 될 경우 버전에 따라 오류 발생 가능, ndarray로 변환.
    get_model_predict(model, X_train.values, X_test.values, y_train.values, y_test.values, is_expm1=True)
```

</div>
</details>

</br>

Unsupervised Learnin
===
차원 축소
---
### 1. 차원의 저주
* 차원이 커질수록
  * 데이터 포인트들 간 거리가 크게 늘어남
  * 데이터가 희소화(Sparse) 됨
* 수백~수천개 이상의 피처로 구성된 포인트들간 거리에 기반한 ML 알고리즘이 무력화됨.
* 또한 피처가 많을 경우 개별 피처간에 상관관계가 높아 선형 회귀와 같은 모델에서는 다중 공선성 문제로 모델의 예측 성능이 저하될 가능성이 높음.
### 2. 차원 축소의 장점
* 수십 ~ 수백개의 피처들을 작은 수의 피처들로 축소한다면?
  * 학습 데이터 크기를 줄여서 학습 시간 절약
  * 불필요한 피처들을 줄여서 모델 성능 향상에 기여(주로 이미지 관련 데이터)
  * 다차원의 데이터를 3차원 이하의 차원 축소를 통해서 시각적으로 보다 쉽게 데이터 패턴 인지
* **어떻게 원본 데이터의 정보를 최대한으로 유지한 채로 차원 축소를 수행할 건인가?**
### 3. 피처 선택과 피처 추출
* 일반적으로 차원 축소는 피처선택(feature selection)과 피처추출(feature extraction)으로 나눌 수 있음.
  * Feature Selection : 특정 피처에 종속성이 강한 불필요한 피처는 아예 제거, 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 것.
  * Feature Extraction : 피처(특성) 추출은 기존 피처를 저차원의 중요 피처로 압축해서 추출하는 것. 이렇게 새롭게 추출된 중요 특성은 기존의 피처를 반영해 압축된 것이지만 새로운 피처로 추출하는 것.    
<img src="C:\Users\user\Desktop\Vocational_Training\FinTech\images\feature_S_E.png" width="40%">

### 4. Feature Extraction
* 피처 추출은 기존 피처를 단순 압축이 아닌, 피처를 함축적으로 더 잘 설명할 수 있는 또 다른 공간으로 매핑해 추출하는 것.     
<img src="C:\Users\user\Desktop\Vocational_Training\FinTech\images\feature_E.png" width="40%">

### 5. 차원 축소의 의미
* 차원 축소는 단순히 데이터의 압축을 의미하는 것이 아님. 더 중요한 의미는 차원 축소를 통해 좀 더 데이터를 잘 설명할 수 있는 잠재적(Latent)인 요소를 추출하는 데에 있음.
  * 추천 엔진
  * 이미지 분류 및 변환
  * 문서 토픽 모델링

  <img src="C:\Users\user\Desktop\Vocational_Training\FinTech\images\dimension_reduction.png" width="40%">

  ## PCA(Pincipal Component Analysis)의 이해
### PCA(주성분 분석, Principal Component Analysis)
* 고차원의 원본 데이터를 저차원의 부분 공간으로 투영하여 데이터를 축소하는 기법
* 예를 들어 10차원의 데이터를 2차원의 부분 공간으로 투영하여 데이터를 축소
* PCA는 원본 데이터가 가지는 데이터 변동성을 가장 중요한 정보로 간주하며 이 변동성에 기반한 원본 데이터 투영으로 차원 축소를 수행
### PCA 원리
* 키, 몸무게 2개의 축을 가지는 2차원 원본 데이터를 가정
* PCA는 **원본 데이터 변동성**이 가장 큰 방향으로 순차적으로 축들을 생성, 이렇게 생성된 축으로 데이터를 투영하는 방식임.
  * A. 데이터 변동성이 가장 큰 방향으로 축 생성
    * 키($X_1$) - 몸무게($X_2$) 평면에서 데이터의 변동성 방향으로 키-몸무게 축 생성.
  * B. 새로운 축으로 데이터 투영
    * 키-몸무게 축에 데이터를 투영함.
  * C. 새로운 축 기준으로 데이터 표현
    * 새롭게 생긴 키-몸무게 축에 매핑되는 1차원 데이터로 차원이 축소됨.
* PCA는 제일 먼저 원본 데이터에 가장 큰 데이터 변동성(Variance)을 기반으로 첫번째 벡터 축을 생성함.   
 두번째 축은 첫번째 축을 제외하고 그 다음으로 변동성이 큰 축을 설정하는데, 이는 첫번째 축에 직각이 되는 벡터(직교벡터)축임.      
 세번째 축은 다시 두번째 축과 직각이 되는 벡터를 설정하는 방식으로 축을 생성함.
 이렇게 생성된 벡터 축에 원본 데이터를 투영하면 벡터 축의 개수만큼의 차원으로 원본 데이터가 차원 축소됨.
* PCA, 즉 주성분 분석은 이처럼 원본 데이터의 피처 개수에 비해 매우 작은 주성분으로 원본 데이터의 총 변동성을 대부분 설명할 수 있는 분석법.
### PCA 프로세스
* PCA를 선형대수 관점에서 해석해 보면, 입력 데이터의 공분산 행렬(Covariance Matrix)을 고유값 분해하고, 이렇게 구한 고유벡터에 입력 데이터를 선형 변환하는 것.
  * 1. 원본데이터의 공분산 행렬 추출
  * 2. 공분산 행렬을 고유 벡터와 고유값 분해
  * 3. 원본 데이터를 고유 벡터로 선형변환
  * 4. PCA 변환 값 도출
* 고유벡터는 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타냄.
* 고윳값(eigenvalue)은 고유벡터의 크기를 나타내며, 동시에 입력 데이터의 분산을 나타냄.
### 공분산 행렬
* 보통 분산은 한개의 특정한 변수의 데이터 변동을 의미하나, 공분산은 두 변수 간의 변동을 의미함.   
  즉, 사람 키 변수를 $X$, 몸무게 변수를 $Y$라고 하면 공분산 $Cov(X,Y) > 0$은 $X$(키)가 증가할 때 $Y$(몸무게)도 증가한다는 의미.

* 공분산 행렬 예시
<center>
<table>
 <tr>
    <th>공분산</th>
    <th>X</th>
    <th>Y</th>
    <th>Z</th>
  </tr>
  <tr>
    <th>X</th>
    <td>3.0 (Var(X))</td>
    <td>-0.71</td>
    <td>-0.24</td>
  </tr>
  <tr>
    <th>Y</th>
    <td>-0.71</td>
    <td>4.5 (Var(Y))</td>
    <td>0.28</td>
  </tr>
  <tr>
    <th>Z</th>
    <td>-0.24</td>
    <td>0.28</td>
    <td>0.91 (Var(Z))</td> 
  </tr>
</table>
</center>

* __공분산 행렬은 여러 변수와 관련된 공분산을 포함하는 정방형 행렬이며 대칭 행렬__
* 정방행렬은 열과 행이 같은 행렬을 지칭.   
  정방행렬 중에서 대각 원소를 중심으로 원소 값이 대칭되는 행렬.   
  즉 $A^T=A$인 행렬을 대칭행렬이라고 부름.
### 선형 변환과 고유 벡터/고유값
* 일반적으로 선형 변환은 특정 벡터에 행렬 A를 곱해 새로운 벡터로 변환하는 것을 의미함.    
  이를 특정 벡터를 하나의 공간에서 다른 공간으로 투영하는 개념으로도 볼 수 있음. 이 경우 이 행렬을 바로 공간으로 가정하는 것.
* 고유벡터는 행렬 A를 곱하더라도 방향이 변하지 않고 그 크기만 변하는 벡터를 지칭함.   
  즉, $A\vec{v} = \lambda\vec{v}$ ($A$는 행렬, $\vec{v}$는 고유벡터, $\lambda$는 스칼라값).   
  이를 만족하는 고유벡터는 여러 개가 존재하며, 정방행렬은 최대 그 차원 수만큼의 고유벡터를 가질 수 있음.
  예를 들어 $2\times 2$행렬은 두 개의 고유벡터를, $3\times 3$ 행렬은 3개의 고유벡터를 가질 수 있음.
  고유벡터는 행렬이 작용하는 힘의 방향과 관계가 있어서 행렬을 분해하는 데 사용됨.
### 공분산 행렬의 고유값 분해
* 공분산 행렬은 정방행렬(Diagonal Matrix)이며 대칭행렬(Symmetric Matrix).   
  정방행렬은 열과 행이 같은 행렬을 지칭. 정방행렬 중에서 대각 원소를 중심으로 원소 값이 대칭되는 행렬, 즉 $A^T=A$인 행렬을 대칭행렬이라 부름.
* 대칭행렬은 고유값 분해와 관련해 매우 좋은 특성이 있음. 대칭행렬은 항상 고유벡터를 직교행렬(orthogonal matrix)로, 고유값을 정방 행렬로 대각화 할 수 있음.
* $C=P{\Lambda}P^T$ ($P$는 $n{\times}n$ 정방행렬, $\Lambda$는 $n{\times}n$ 정방행렬)
  $$C = \begin{bmatrix}e_1&\cdots&e_n\\ \end{bmatrix}\begin{bmatrix}\lambda_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&\lambda_n \end{bmatrix}\begin{bmatrix}e_1\\\vdots\\{e_n} \end{bmatrix}$$
* 공분산 $C$는 고유벡터 직교 행렬, 고유값 정방행렬 * 고유벡터 직교행렬의 전치 행렬로 분해됨.
* $e_1$는 첫번째 고유벡터를, $\lambda_i$는 i번째 고유벡터의 크기를 의미함.
  $e_2$는 $e_1$에 수직이면서 다음으로 가장 분산이 큰 방향을 가진 고유벡터임.

#### 실습

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
# 1. Covariance Matrix of Features
import pandas as pd

# Eating, exercise habbit and their body shape
df = pd.DataFrame(columns=['calory', 'breakfast', 'lunch', 'dinner', 'exercise', 'body_shape'])
```
```python
# 데이터 프레임 설정
df.loc[0] = [1200, 1, 0, 0, 2, 'Skinny']
df.loc[1] = [2800, 1, 1, 1, 1, 'Normal']
df.loc[2] = [3500, 2, 2, 1, 0, 'Fat']
df.loc[3] = [1400, 0, 1, 0, 3, 'Skinny']
df.loc[4] = [5000, 2, 2, 2, 0, 'Fat']
df.loc[5] = [1300, 0, 0, 1, 2, 'Skinny']
df.loc[6] = [3000, 1, 0, 1, 1, 'Normal']
df.loc[7] = [4000, 2, 2, 2, 0, 'Fat']
df.loc[8] = [2600, 0, 2, 0, 0, 'Normal']
df.loc[9] = [3000, 1, 2, 1, 1, 'Fat']
```
```python
# Data, label 설정
y = df["body_shape"]
X = df.drop("body_shape", axis=1, inplace=False)
X
```
```python
from sklearn.preprocessing import StandardScaler

# StandardScale
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std
```
```python
import numpy as np

# 공분산 행렬 생성
features = X_std.T
cov_mtrx = np.cov(features)
cov_mtrx
```
```python
# 고유 벡터와 고유값 with 공분산 행렬
eig_vals, eig_vecs = np.linalg.eig(cov_mtrx)

print(eig_vals)
print()
print(eig_vecs)
```
```python
eig_vals[0]/sum(eig_vals)   # 특정 칼럼의 영향력 측정
```
```python
# project data into selected eigen vector
projected_X = X_std.dot(eig_vecs.T[0])
```
```python
# PCA 결과
result = pd.DataFrame(projected_X, columns=["PC_1"])
result["label"] = y
result["y-axis"] = 0.0
result
```
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

# 원본 데이터의 디시전 트리 reg 검사
labels = LabelEncoder()
labels = labels.fit_transform(y)
# y_encd = OneHotEncoder()
# y_encd = y_encd.fit_transform(labels.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_std, labels,
                                        test_size=0.2, random_state=0)

lg_reg = DecisionTreeRegressor()
lg_reg.fit(X_train, y_train)
pred = lg_reg.predict(X_test)

score = accuracy_score(y_test, pred)
print(score)
```
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화
sns.lmplot("PC_1", "y-axis", data=result, scatter_kws={"s":50}, hue="label")
plt.title("PCA result")
```
```python
# 2. sklearn PCA 라이브러리 이용
from sklearn.decomposition import PCA

pca = PCA(n_components=1)

pca_x = pca.fit_transform(X_std)
pca_x
```
```python
# 라이브러리 이용한 PCA 결과 확인
result = pd.DataFrame(pca_x, columns=["PC_1"])
result["label"] = y
result["y-axis"] = 0.0
result
# 부호 반대
```
```python
# 시각화
sns.lmplot("PC_1", "y-axis", data=result, scatter_kws={"s":50}, hue="label")
plt.title("PCA result")
```

</div>
</details>

</br>

### PCA 요약
* PCA 변환
  * 입력 데이터의 공분산 행렬이 고유벡터와 고유값으로 분해될 수 있으며, 이렇게 분해된 고유벡터를 이용해 입력 데이터를 선형 변환하는 방식
* PCA 변환 수행 절차
  1. 입력 데이터 세트의 공분산 행렬을 생성함.
  2. 공분산 행렬의 고유벡터와 고유값을 계산함.
  3. 고유값이 가장 큰 순으로 K개(PCA 변환 차수)만큼 고유벡터를 추출함.
  4. 고유값이 가장 큰 순으로 추출된 교유벡터를 이용해 새롭게 입력 데이터를 변환.
### 사이킷런 PCA 클래스
* sklearn.decomposition.PCA
* class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

<details>
<summary>Parameters 펼치기/접기</summary>
<div markdown="1">

* **n_components** : int, float or ‘mle’, default=None

  >Number of components to keep. if n_components is not set all components are kept:
  >
  > n_components == min(n_samples, n_features)
  >
  >If  `n_components  ==  'mle'`  and  `svd_solver  ==  'full'`, Minka’s MLE is used to guess the dimension. Use of  `n_components  ==  'mle'`  will interpret  `svd_solver  ==  'auto'`  as  `svd_solver  ==  'full'`.
  >
  >If  `0  <  n_components  <  1`  and  `svd_solver  ==  'full'`, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
  >
  >If  `svd_solver  ==  'arpack'`, the number of components must be strictly less than the minimum of n_features and n_samples.
  >
  > Hence, the None case results in:
  >
  > n_components == min(n_samples, n_features) - 1


* **copy** : bool, default=True

  >If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.

* **whiten** : bool, default=False

  >When True (False by default) the  `components_`  vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
  >
  >Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.

* **svd_solver** : {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’

  >If auto :
  >
  >The solver is selected by a default policy based on  `X.shape`  and `n_components`: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards.

  >If full :
  >
  >run exact full SVD calling the standard LAPACK solver via  `scipy.linalg.svd`  and select the components by postprocessing

  >If arpack :
  >
  >run SVD truncated to n_components calling ARPACK solver via  `scipy.sparse.linalg.svds`. It requires strictly 0 < n_components < min(X.shape)

  >If randomized :
  >
  >run randomized SVD by the method of Halko et al.
  >


* **tol** : float, default=0.0

  >Tolerance for singular values computed by svd_solver == ‘arpack’. Must be of range [0.0, infinity).



* **iterated_power** : int or ‘auto’, default=’auto’

  >Number of iterations for the power method computed by svd_solver == ‘randomized’. Must be of range [0, infinity).



* **random_state** : int, RandomState instance or None, default=None

  >Used when the ‘arpack’ or ‘randomized’ solvers are used. Pass an int for reproducible results across multiple function calls. See  [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

</div>
</details>

</br>

<details>
<summary>Attributes 펼치기/접기</summary>
<div markdown="1">

* **components_** : ndarray of shape (n_components, n_features)
	
	> Principal axes in feature space, representing the directions of maximum variance in the data. Equivalently, the right singular vectors of the centered input data, parallel to its eigenvectors. The components are sorted by  `explained_variance_`.

* **explained_variance_** : ndarray of shape (n_components,)
	
	> The amount of variance explained by each of the selected components. The variance estimation uses  `n_samples  -  1`  degrees of freedom.
	>
	> Equal to n_components largest eigenvalues of the covariance matrix of X.


* **explained_variance_ratio_** : ndarray of shape (n_components,)
	
	> Percentage of variance explained by each of the selected components.
	> 
	> If  `n_components`  is not set then all components are stored and the sum of the ratios is equal to 1.0.
	>
* **singular_values_**: ndarray of shape (n_components,)
	
	>The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the  `n_components`  variables in the lower-dimensional space.


* **mean_** : ndarray of shape (n_features,)

	> Per-feature empirical mean, estimated from the training set.
	>
	>Equal to  `X.mean(axis=0)`.

* **n_components_** : int
	
	> The estimated number of components. When n_components is set to ‘mle’ or a number between 0 and 1 (with svd_solver == ‘full’) this number is estimated from input data. Otherwise it equals the parameter n_components, or the lesser value of n_features and n_samples if n_components is None.

* **n_features_** : int

	> Number of features in the training data.

* **n_samples_** : int

	>Number of samples in the training data.

* **noise_variance_** : float
	
	>The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999. See “Pattern Recognition and Machine Learning” by C. Bishop, 12.2.1 p. 574 or  [http://www.miketipping.com/papers/met-mppca.pdf](http://www.miketipping.com/papers/met-mppca.pdf). It is required to compute the estimated data covariance and score samples.
	>
	>Equal to the average of (min(n_features, n_samples) - n_components) smallest eigenvalues of the covariance matrix of X.

* **n_features_in_** : int

	> Number of features seen during  [fit](https://scikit-learn.org/stable/glossary.html#term-fit).


* **feature_names_in_** : ndarray of shape (`n_features_in_`,)
	
	> Names of features seen during  [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when  `X`  has feature names that are all strings.

</div>
</details>

</br>

# 과제
1. 붓꽃데이터를 데이터 전처리
2. train : test = 8:2
3. 차원 축소(components=2)
4. RamdomForest를 이용해서 학습
5. 차원축소 하지 않은 것과 비교를 accuracy로 평가

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
std = StandardScaler()
iris_data = iris.data
iris_label = iris.target

iris_data_std = std.fit_transform(iris_data)

pca = PCA(n_components=2)
pca_x = pca.fit_transform(iris_data_std)

X_train, X_test, y_train, y_test = train_test_split(iris_data_std, iris_label, test_size=0.2, random_state=0)
X_pca_train, X_pca_test, y_train, y_test = train_test_split(pca_x, iris_label, test_size=0.2, random_state=0)

rf_clf_orgn = RandomForestClassifier()
rf_clf_pca = RandomForestClassifier()

rf_clf_orgn.fit(X_train, y_train)
rf_clf_pca.fit(X_pca_train, y_train)

pred_orgn = rf_clf_orgn.predict(X_test)
pred_pca = rf_clf_pca.predict(X_pca_test)

print(f"일반 예측 결과 : {accuracy_score(y_test, pred_orgn)},\nPCA 예측 결과 {accuracy_score(y_test, pred_pca)}")
```
```python
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)
iris_df["target"] = iris_label
iris_df
```
```python
# PCA 전 원본 데이터 분류 시각화
import matplotlib.pyplot as plt
# setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=["^", "s", "o"] # 세모, 네모, 동그라미

# setosa의 target 값은 0, versicolor는 1, virginica는 2.
# 각 target별로 다른 shape로 scatter plot.
for i, marker in enumerate(markers) :
    x_axis_data = iris_df[iris_df["target"]==i]["sepal length (cm)"]
    y_axis_data = iris_df[iris_df["target"]==i]["sepal width (cm)"]
    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()
```
```python
# PCA
iris_df_pca = pd.DataFrame(pca_x, columns=["pca_component_1", "pca_component_2"])
iris_df_pca["target"] = iris_label
```
```python
# PCA 분포 시각화
# setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=["^", "s", "o"] # 세모, 네모, 동그라미

# setosa의 target 값은 0, versicolor는 1, virginica는 2.
# 각 target별로 다른 shape로 scatter plot.
for i, marker in enumerate(markers) :
    x_axis_data = iris_df_pca[iris_df_pca["target"]==i]["pca_component_1"]
    y_axis_data = iris_df_pca[iris_df_pca["target"]==i]["pca_component_2"]
    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel("pca_component_1")
plt.ylabel("pca_component_2")
plt.show()
# 원본보다 잘 분할 되어 있는 것 확인할 수 있음
```

</div>
</details>

</br>

실습
===
신용카드 연체 예측 데이터 PCA
---

<details>
<summary> 코드 펼치기/접기</summary>
<div markdown="1">

### 1. credit card 데이터 세트 변환
* 신용카드 연체 예측(UCI credit card default data)
* 데이터 전처리 : 컬럼명 변경, 속성/클래스 분류

```python
# 신용카드 연체 예측(UCI credit card default data)
# 예제 : credit card 데이터 세트 변환
import pandas as pd

df = pd.read_csv("./csv_data/UCI_Credit_Card.csv", encoding="CP949")
df
```
```python
df.info()   # null 등 있는지 확인
```
```python
df.describe()   # 데이터 정보
```
```python
# 컬럼명 변경
df = df.rename(columns={"PAY_0" : "PAY_1", "default.payment.next.month" : "default"})
df
```
```python
# 속성과 클래스로 데이터 분류
y_target = df["default"]
X_features = df.drop("default", axis=1)
```
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 공분산행렬
corr = X_features.corr()
corr
```
### 2. 피처 간 상관관계 살펴보기
* 히트맵으로 피처 간 상관관계 시각화
  * 과거 지불 금액간 상관관계 높음.
  * 과거 청구 금액간 상관관계는 더 높음.
  * 상관도가 높은 피처들 간에는 PCA 효율이 좋음.
```python
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True, fmt=".1g")
```
### 3. 일부 피처들 PCA 변환(n_components=2)
* 일부 상관도가 높은 피처들(BILL_AMT1~6)을 PCA(n_components=2) 변환 후 변동성 확인
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# BILL_AMT1 ~ BILL_AMT6 까지 6개의 속성명 생성
cols_bill = ["BILL_AMT" + str(i) for i in range(1, 7)]
print(f"대상 속성명 : {cols_bill}")

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_ 계산 위해 fit() 호출
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca_2 = PCA(n_components=2)
pca_2.fit(df_cols_scaled)

df_bill=pca_2.transform(df_cols_scaled)

print(f"PCA Component별 변동성 : {pca_2.explained_variance_ratio_}")

# 6개의 피처를 2개의 피처로 PCA 변환했을 때 첫번째 컴포넌트가 전체 변동성의 90%를 설명함.
```
```python
# PAY의 경우
cols_pay = ["PAY_AMT" + str(i) for i in range(1, 7)]
print(f"대상 속성명 : {cols_pay}")

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_ 계산 위해 fit() 호출
scaler = StandardScaler()
df_pay_cols_scaled = scaler.fit_transform(X_features[cols_pay])

pca_pay_2 = PCA(n_components=2)
pca_pay_2.fit(df_pay_cols_scaled)

df_pay = pca_pay_2.transform(df_pay_cols_scaled)

print(f"PCA Component별 변동성 : {pca_pay_2.explained_variance_ratio_}")

# PAY 피처는 6개를 2개로 PCA 변환했을 때 첫번째 컴포넌트가 전체 변동성의 32%를 설명함.
```
### 4. 전체 피처들 PCA 변환(n_components=7)
* 전체 원본 데이터와 PCA 변환된 데이터 간 랜덤 포레스트 예측 성능 비교
```python
# 1. 원본 데이터
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators=300, random_state=156)

# 원본 데이터일 때 랜덤 포레스트 예측 성능
scores = cross_val_score(rcf, X_features, y_target, scoring="accuracy", cv=3)

print(f"CV=3 인 경우의 개별 Fold세트별 정확도 : {scores}")
print(f"평균 정확도 : {np.mean(scores):.4f}")
```
```python
# 2. PCA 변환된 데이터
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 원본 데이터셋에 먼저 StandardScaler 적용
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X_features)

# PCA 변환을 수행하고 랜덤 포레스트 예측 성능
pca_7 = PCA(n_components=7)
df_pca_7 = pca_7.fit_transform(df_scaled)
scores_pca = cross_val_score(rcf, df_pca_7, y_target, scoring="accuracy", cv=3)

print(f"CV=3 인 경우의 PCA 변환된 개별 Fold세트별 정확도 : {scores_pca}")
print(f"PCA 변환 데이터 셋 평균 정확도 : {np.mean(scores_pca):.4f}")
```
### 5. PCA 수행한 새로운 데이터 프레임 분석
```python
cols = cols_bill+cols_pay
df_new = df.drop(cols, axis=1)

PCA_BILL = pd.DataFrame(df_bill, columns=["PCA_BILL_1", "PCA_BILL_2"])
PCA_PAY = pd.DataFrame(df_pay, columns=["PCA_PAY_1", "PCA_PAY_2"])

df_new = df_new.merge(PCA_BILL, left_index=True, right_index=True)
df_new = df_new.merge(PCA_PAY, left_index=True, right_index=True)
df_new
```
```python
X_PCA_features = df_new.drop(["default"], axis=1)
```
```python
rcf = RandomForestClassifier(n_estimators=300, random_state=156)

# pca 수행 후 성능 측정
scores_pca = cross_val_score(rcf, X_PCA_features, y_target, scoring="accuracy", cv=3)

print(f"CV=3 인 경우의 PCA 변환 후 개별 Fold세트별 정확도 : {scores_pca}")
print(f"PCA 변환 시 평균 정확도 : {np.mean(scores_pca):.4f}")

# 원본의 경우
# CV=3 인 경우의 개별 Fold세트별 정확도 : [0.257  0.8209 0.784 ]
# 평균 정확도 : 0.6206
```


</div>
</details>

</br>

군집화
===
Unsupervised Learning
---
### 군집화란
* 군집화(clustering)
  * 데이터 포인트들을 별개의 군집으로 그룹화 하는 것을 의미함
  * 유사성이 높은 데이터들을 동일한 그룹으로 분류하고 서로 다른 군집들이 상이성을 가지도록 그룹화함.
* 군집화 활용분야
  * 고객, 마켓, 브랜드, 사회 경제 활동 세분화(Segmentation)
  * Image 검출, 세분화, 트랙킹
  * 이상 검출(Abnomaly detection)
* 어떻게 유사성을 정의할 것인가?
### 군집화 알고리즘 종류
* K-Means : centroid(군집 중심점) 기반
* Mean Shift : centroid(군집 중심점) 기반
* Gaussian Mixture Model : 데이터 정규분포 기반
* DBSCAN : 데이터 밀도 기반

K-Means Clustering
---
### K-Means clustering
* 군집 중심점(Centroid) 기반 클러스터링
<center>
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/clustering.png">
</center>

### K-Means의 장점/단점
* 장점
  * 일반적인 군집화에서 가장 많이 활용되는 알고리즘.
  * 알고리즘이 쉽고 간결함.
  * 대용량 데이터에도 활용이 가능.
* 단점
  * 거리 기반 알고리즘으로 속성의 개수가 매우 많을 경우 군집화 정확도가 떨어짐(차원의 저주 문제 발생 가능).
  * 이를 위해 PCA로 차원 축소를 적용해야 할 수도 있음.
  * 반복을 수행하는데, 반복 횟수가 많을 경우 수행 시간이 느려짐.
  * 이상치(Outlier) 데이터에 취약함.
### 사이킷런 K-Means 클래스
* sklearn.cluster.KMeans

* >class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')

<details>
<summary>Parameters 펼치기/접기</summary>
<div markdown="1">

* **n_clusters** : int, default=8

	>The number of clusters to form as well as the number of centroids to generate.

* **init** : {‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’

	>Method for initialization:
	>
	>‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
	>
	> ‘random’: choose  `n_clusters`  observations (rows) at random from data for the initial centroids.
	>
	> If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
	>
	>If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.

* **n_init** : int, default=10

	>Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

* **max_iter** : int, default=300

	>Maximum number of iterations of the k-means algorithm for a single run.

* **tol** : float, default=1e-4

	>Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.

* **verbose** : int, default=0

	>Verbosity mode.

* **random_state** : int, RandomState instance or None, default=None

	>Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See  [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

* **copy_x** : bool, default=True

	>When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False.

* **algorithm** : {“auto”, “full”, “elkan”}, default=”auto”
	
	>K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
	>
	>For now “auto” (kept for backward compatibility) chooses “elkan” but it might change in the future for a better heuristic.

</div>
</details>

</br>

<details>
<summary>Attributes 펼치기/접기</summary>
<div markdown="1">

* **cluster_centers_** : ndarray of shape (n_clusters, n_features)

	>Coordinates of cluster centers. If the algorithm stops before fully converging (see  `tol`  and  `max_iter`), these will not be consistent with  `labels_`.

* **labels_** : ndarray of shape (n_samples,)

	> Labels of each point

* **inertia_** : float

	> Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.

* **n_iter_** : int

	> 	Number of iterations run.

* **n_features_in_** : int
	
	> Number of features seen during  [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

* **feature_names_in_**: ndarray of shape (`n_features_in_`,)

	> Names of features seen during  [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when  `X`  has feature names that are all strings.

</div>
</details>

</br>

#### 실습

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

##### 붓꽃 데이터 K-Means Clustering
```python
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
columns = []
for i in iris.feature_names :
    i = i.replace(" (cm)", "")
    i = i.replace(" ","_")
    columns.append(i)

# 피처 데이터만 별도로 저장
irisDF = pd.DataFrame(data=iris.data, columns=columns)
```
##### 붓꽃 데이터에 K-Means 군집화 수행
* 피처 데이터에 K-Means 적용
```python
# kmeans 객체 생성
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, random_state=0)

# 붓꽃 데이터에 군집화 수행
kmeans.fit(irisDF)
```
```python
# 각 데이터들마다 centroid(군집 중심점) 할당됨
print(kmeans.labels_)

irisDF["cluster"]=kmeans.labels_
```
```python
# 타겟 별 군집 중심점 확인
irisDF["target"] = iris.target

iris_result = irisDF.groupby(["target", "cluster"])["sepal_length"].count()
print(iris_result)
```
##### 군집화 결과 시각화(PCA 2차원 변환)
* 2차원 평면에 데이터 군집화된 결과 나타내기 위해 2차원 PCA로 차원 축소
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

irisDF["pca_x"] = pca_transformed[:,0] # 전체 행(:), 0번째 열
irisDF["pca_y"] = pca_transformed[:,1]
```
```python
# cluster 값이 0, 1, 2 인 경우마다 별도의 Index로 추출
marker0_ind = irisDF[irisDF["cluster"]==0].index
marker1_ind = irisDF[irisDF["cluster"]==1].index
marker2_ind = irisDF[irisDF["cluster"]==2].index

# cluster 값이 0, 1, 2 에 해당하는 Index로 각 cluster 레벨의 pca_x, pca_y 값 추출.
# o, s, ^ 로 marker 표시
plt.scatter(x=irisDF.loc[marker0_ind,"pca_x"], y=irisDF.loc[marker0_ind, "pca_y"], marker="o")
plt.scatter(x=irisDF.loc[marker1_ind,"pca_x"], y=irisDF.loc[marker1_ind, "pca_y"], marker="s")
plt.scatter(x=irisDF.loc[marker2_ind,"pca_x"], y=irisDF.loc[marker2_ind, "pca_y"], marker="^")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("3 Clusters Visualization by 2 PCA Components")
plt.show()
```
##### K-Means 수행 후 개별 클러스터의 군집 중심 시각화
* Clustering 알고리즘 테스트를 위한 데이터 생성
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 클러스터링 할 데이터 생성 - make_blobs (생성할 데이터 200개, 데이터 피처 갯수 2개,
#                           군집 갯수3개, 데이터 표준편차 0.8)
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
print(X.shape, y.shape)
```
```python
# y target 값의 분포를 확인
unique, counts = np.unique(y, return_counts=True)
print(unique, counts)

# 타겟값 0,1,2 / 총 200개 데이터
```
```python
import pandas as pd

# 데이터 프레임화
clusterDF = pd.DataFrame(data=X, columns=["ftr1", "ftr2"])
clusterDF["target"] = y

print(clusterDF.shape)
clusterDF
```
```python
# make_blobs로 만들어진 데이터 시각화
target_list = np.unique(y)

# 각 target별 scatter plot의 marker 값들.
markers = ["o", "s", "^", "P", "D", "H", "x"]

# 3개의 cluster 영역으로 구분한 데이터 셋을 생성했으므로 target_list는 [0, 1, 2]
# target == 0, target == 1, target == 2 로 scatter plot을 marker별로 생성.
for target in target_list :
    target_cluster = clusterDF[clusterDF["target"]==target]
    plt.scatter(x=target_cluster["ftr1"], y=target_cluster["ftr2"],
                edgecolors="k", marker=markers[target])
plt.show()
```
```python
# K-Means 군집화 수행하고 개별 클러스터의 군집 중심 시각화

# KMeans 객체를 이용하여 X 데이터를 K-Means 클러스터링 수행
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=200, random_state=0)
cluster_labels = kmeans.fit_predict(X)
clusterDF["kmeans_label"] = cluster_labels

# cluster_centers는 개별 클러스터의 중심 위치 좌표 시각화를 위해 추출
centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ["o", "s", "^", "P", "D", "H", "x"]

# 군집화 label 유형별로 iteration하면서 marker 별로 scatter plot 수행.
for label in unique_labels :
    label_cluster = clusterDF[clusterDF["kmeans_label"]==label]
    center_x_y = centers[label]
    plt.scatter(x=label_cluster["ftr1"], y=label_cluster["ftr2"], edgecolors="k",
                marker=markers[label])

    # 군집별 중심 위치 좌표 시각화
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color="white",
                alpha=0.9, edgecolors="k", marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color="k", edgecolors="k",
                marker=f"${label}$")

print(kmeans.cluster_centers_)
plt.show()
```
```python
# 분포 확인
print(clusterDF.groupby("target")["kmeans_label"].value_counts())
```



</div>
</details>

</br>

실습
===
신용카드 연체 예측 데이터 PCA
---

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

### 1. credit card 데이터 세트 변환
* 신용카드 연체 예측(UCI credit card default data)
* 데이터 전처리 : 컬럼명 변경, 속성/클래스 분류

```python
# 신용카드 연체 예측(UCI credit card default data)
# 예제 : credit card 데이터 세트 변환
# 데이터 로드
import pandas as pd

df = pd.read_csv("./csv_data/UCI_Credit_Card.csv", encoding="CP949")
```
```python
# 데이터 정보 확인
df.info()
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# X2: Gender (1 = male; 2 = female).
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# X4: Marital status (1 = married; 2 = single; 3 = others).
# X5: Age (year).
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
```
```python
# 컬럼명 변경
df = df.rename(columns={"PAY_0" : "PAY_1", "default.payment.next.month" : "default"})
```
```python
# 속성과 클래스로 데이터 분류
y_target = df["default"]
X_features = df.drop("default", axis=1)
```
```python
# 분산 공분산 행렬
import seaborn as sns
import matplotlib.pyplot as plt

corr = X_features.corr()
corr
```

### 2. 피처 간 상관관계 살펴보기
* 히트맵으로 피처 간 상관관계 시각화
  * 과거 지불 금액간 상관관계 높음.
  * 과거 청구 금액간 상관관계는 더 높음.
  * 상관도가 높은 피처들 간에는 PCA 효율이 좋음.

```python
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True, fmt=".1g")
```

### 3. 일부 피처들 PCA 변환(n_components=2)
* 일부 상관도가 높은 피처들(BILL_AMT1~6)을 PCA(n_components=2) 변환 후 변동성 확인

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# BILL_AMT1 ~ BILL_AMT6 까지 6개의 속성명 생성
cols_bill = ["BILL_AMT" + str(i) for i in range(1, 7)]
print(f"대상 속성명 : {cols_bill}")

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_ 계산 위해 fit() 호출
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca_2 = PCA(n_components=2)
pca_2.fit(df_cols_scaled)

df_bill=pca_2.transform(df_cols_scaled)

print(f"PCA Component별 변동성 : {pca_2.explained_variance_ratio_}")

# 6개의 피처를 2개의 피처로 PCA 변환했을 때 첫번째 컴포넌트가 전체 변동성의 90%를 설명함.
```

```python
# PAY의 경우
cols_pay = ["PAY_AMT" + str(i) for i in range(1, 7)]
print(f"대상 속성명 : {cols_pay}")

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_ 계산 위해 fit() 호출
scaler = StandardScaler()
df_pay_cols_scaled = scaler.fit_transform(X_features[cols_pay])

pca_pay_2 = PCA(n_components=2)
pca_pay_2.fit(df_pay_cols_scaled)

df_pay = pca_pay_2.transform(df_pay_cols_scaled)

print(f"PCA Component별 변동성 : {pca_pay_2.explained_variance_ratio_}")

# PAY 피처는 6개를 2개로 PCA 변환했을 때 첫번째 컴포넌트가 전체 변동성의 32%를 설명함.
```

### 4. 전체 피처들 PCA 변환(n_components=7)
* 전체 원본 데이터와 PCA 변환된 데이터 간 랜덤 포레스트 예측 성능 비교

```python
# 1. 원본 데이터
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators=300, random_state=156)

# 원본 데이터일 때 랜덤 포레스트 예측 성능
scores = cross_val_score(rcf, X_features, y_target, scoring="accuracy", cv=3)

print(f"CV=3 인 경우의 개별 Fold세트별 정확도 : {scores}")
print(f"평균 정확도 : {np.mean(scores):.4f}")
```
### 5. PCA 수행한 새로운 데이터 프레임 분석

```python
# PCA 결과 확인
cols = cols_bill+cols_pay
df_new = df.drop(cols, axis=1)

PCA_BILL = pd.DataFrame(df_bill, columns=["PCA_BILL_1", "PCA_BILL_2"])
PCA_PAY = pd.DataFrame(df_pay, columns=["PCA_PAY_1", "PCA_PAY_2"])

df_new = df_new.merge(PCA_BILL, left_index=True, right_index=True)
df_new = df_new.merge(PCA_PAY, left_index=True, right_index=True)
df_new
```
```python
# 불필요한 컬럼 삭제
X_PCA_features = df_new.drop(["default"], axis=1)
```
```python
rcf = RandomForestClassifier(n_estimators=300, random_state=156)

# pca 수행 후 성능 측정
scores_pca = cross_val_score(rcf, X_PCA_features, y_target, scoring="accuracy", cv=3)

print(f"CV=3 인 경우의 PCA 변환 후 개별 Fold세트별 정확도 : {scores_pca}")
print(f"PCA 변환 시 평균 정확도 : {np.mean(scores_pca):.4f}")

# 원본의 경우
# CV=3 인 경우의 개별 Fold세트별 정확도 : [0.257  0.8209 0.784 ]
# 평균 정확도 : 0.6206
```

</div>
</details>

</br>

LDA(Linear Discriminant Analysis)
===
* LDA는 선형 판별 분석법으로 불리며, PCA와 매우 유사함.
* LDA는 PCA와 유사하게 입력 데이터 세트를 저차원 공간에 투영해 차원을 축소하는 기법.   
  중요한 차이는 LDA는 지도학습의 분류(Classification)에서 사용하기 쉽도록 개별 클래스를 분별할 수 있는 기준을 최대한 유지하면서 차원을 축소함.   
  PCA는 입력 데이터의 변동성이 가장 큰 축을 찾지만, LDA는 입력 데이터의 결정 값 클래스를 최대한으로 분리할 수 있는 축을 찾음
* LDA는 같은 클래스의 데이터는 최대한 근접해서, 다른 클래스의 데이터는 최대한 떨어뜨리는 축을 매핑함.
<center>
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/LDA.png">
</center>

## LDA 차원 축소 방식
* LDA는 특정 공간상에서 클래스 분리를 최대화하는 축을 찾기 위해 클래스 간 분산(between-class scatter)과 클래스 내부 분산(within-calss scatter)의 비율을 최대화하는 방식으로 차원을 축소함.
* 즉, 클래스 간 분산은 최대한 크게 가져가고, 클래스 내부의 분산은 최대한 작게 가져가는 방식.
<center>
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/LDA_variance.png">
</center>

## LDA 절차
* 일반적으로 LDA를 구하는 스탭은 PCA와 유사하나, 가장 큰 차이점은 공분산 행렬이 아니라 클래스 간 분산과 클래스 내부 분산 행렬을 생성한 뒤, 이 행렬에 기반해 고유벡터를 구하고 입력 데이터를 투영한다는 점임.
1. 클래스 내부와 클래스 간 분산 행렬을 구함.   
  이 두 개의 행렬은 입력 데이터의 결정 값 클래스별로 개별 피처의 평균 벡터(mean vector)를 기반으로 구함.
2. 클래스 내부 분산 행렬을 $S_W$, 클래스 간 분산 행렬을 $S_B$라고 하면 다음 식으로 두 행렬을 고유벡터로 분해할 수 있음.
$$S_W^T S_B = \begin{bmatrix}e_1&\cdots&e_n\\ \end{bmatrix}\begin{bmatrix}\lambda_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&\lambda_n \end{bmatrix}\begin{bmatrix}e_1\\\vdots\\{e_n} \end{bmatrix}$$
3. 고유값이 가장 큰 순으로 K개(LDA변환 차수만큼) 추출함.
4. 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환함.

### 실습 LDA

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data)
```
```python
# LDA를 이용해 2개 축을 추출
lda = LinearDiscriminantAnalysis(n_components=2)

# fit 호출시 target값 입력
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)
print(iris_lda.shape)
```
```python
import pandas as pd
import matplotlib.pyplot as plt

lda_columns = ["lda_component_1", "lda_component_2"]
iris_df_lda = pd.DataFrame(iris_lda, columns=lda_columns)
iris_df_lda["target"] = iris.target

# setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=["^", "s", "o"]

# setosa의 target 값은 0, versicolor는 1, virginica는 2, 각 target 별로 다른 shape로 scatter plot
for i, marker in enumerate(markers) :
    x_axis_data = iris_df_lda[iris_df_lda["target"]==i]["lda_component_1"]
    y_axis_data = iris_df_lda[iris_df_lda["target"]==i]["lda_component_2"]

    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])

plt.legend(loc="upper right")
plt.xlabel("lda_component_1")
plt.ylabel("lda_component_2")
plt.show()
```

</div>
</details>

</br>

특이값 분해 SVD
===
## 대표적인 행렬 분해 방법
### 고유값 분해(Eigen-Decomposition)
$$C=P{\Lambda}P^T$$
$$C = \begin{bmatrix}e_1&\cdots&e_n\\ \end{bmatrix}\begin{bmatrix}\lambda_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&\lambda_n \end{bmatrix}\begin{bmatrix}e_1\\\vdots\\{e_n} \end{bmatrix}$$
* 정방행렬만을 고유벡터로 분해.
* PCA는 분해된 고유벡터에 원본 데이터를 투영하여 차원 축소.
### 특이값 분해(Singular Value Decomposition)
$$A=U{\Sigma}V^T$$
* SVD는 정방행렬뿐만 아니라 행과 열의 크기가 다른 $m{\times}n$행렬도 분해 가능.
* $U$ : 왼쪽 직교행렬, ${\Sigma}$ : 대각 행렬, $V^T$ : 오른쪽 직교행렬
* 행렬 $U$와 $V$에 속한 벡터는 특이벡터(singular vector)이며, 모든 특이벡터는 서로 직교하는 성질을 가짐.
$$U^TU=I$$
$$V^TV=I$$
* ${\Sigma}$는 대각행렬이며, 행렬의 대각에 위치한 값만 0이 아니고 나머지 위치의 값은 모두 0.
* ${\Sigma}$이 위치한 0이 아닌 값이 바로 행렬 $A$의 특이값.
## SVD 유형
<center>
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/SVD_classes.png">
</center>

### Truncated SVD 행렬 분해의 의미
$$A_{(m{\times}n)}{\approx}\begin{bmatrix}m{\times}r\end{bmatrix} \begin{bmatrix}r{\times}r\end{bmatrix} \begin{bmatrix}r{\times}n\end{bmatrix} = A'_{(m{\times}n)}$$
* SVD는 차원 축소를 위한 행렬 분해를 통해 Latent Factor(잠재 요인)을 찾을 수 있는데, 이렇게 찾아진 Latent Factor는 많은 분야에 활용(추천 엔진, 문서의 잠재 의미 분석 등).
* SVD로 차원 축소 행렬 분해된 후 다시 분해된 행렬을 이용하여 원복된 데이터 셋은 잡음(Noise)이 제거된 형태로 재구성 될 수 있음.
* 사이킷런에서는 TruncatedSVD로 차원을 축소할 때 원본 데이터에 $U{\Sigma}$를 적용하여 차원 축소.
## SVD 활용
* 이미지 압축/변환
* 추천엔진
* 문서 잠재 의미 분석
* 의사 역행렬을 통한 모델 예측
### 실습

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
# numpy의 svd 모듈 import
import numpy as np
from numpy.linalg import svd

# 4X4 Random 행렬 a 생성
np.random.seed(121)
a = np.random.randn(4,4)
print(np.round(a,3))
```
```python
# 4X4 행렬을 특이값 분해
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print()
print(f"U matrix :\n{np.round(U,3)}")
print()
print(f"Sigma Value :\n{np.round(Sigma,3)}")
print()
print(f"V transpose matrix :\n{np.round(Vt,3)}")
```
```python
# Sigma를 다시 0을 포함한 대칭행렬로 변환
Sigma_mat = np.diag(Sigma)
a_ = np.dot(np.dot(U, Sigma_mat), Vt)
print(np.round(Sigma_mat,3))
print()
print(np.round(a_,3))
```
* 데이터 의존도가 높은 원본 원본 데이터 행렬 생성 for compact SVD
```python
print(np.round(a,3))
print()
# 행렬내의 값 사이에 공선성 형성
a[2] = a[0] + a[1]
a[3] = a[0]
print(np.round(a,3))
```
```python
# 다시 SVD를 수행하여 Sigma 값 확인
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print(f"Sigma Value :\n{np.round(Sigma,3)}")

# 원본에서는 [3.423 2.023 0.463 0.079]
```
```python
# U 행렬의 경우는 Sigma와 내적을 수행하므로 Sigma의 앞 2행에 대응되는 앞 2열만 추출
U_ = U[:, :2]
Sigma_ = np.diag(Sigma[:2])

# V 전치 행렬의 경우는 앞 2행만 추출
Vt_ = Vt[:2]
print(U_.shape, Sigma_.shape, Vt_.shape)
print()

# U, Sigma, Vt의 내적을 수행하며, 다시 원본 행렬 복원
a_ = np.dot(np.dot(U_,Sigma_), Vt_)
print(np.round(a_,3))
```
* Truncated SVD를 이용한 행렬 분해
```python
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals

# 원본 행렬을 출력하고, SVD를 적용할 경우 U, Sigma, Vt의 차원 확인
np.random.seed(121)
matrix = np.random.random((6,6))
print(f"원본 행렬 :\n{matrix}")
U, Sigma, Vt = svd(matrix, full_matrices=False)
print(f"\n분해 행렬 차원 : {U.shape, Sigma.shape, Vt.shape}")
print(f"\nSigma값 행렬 : {Sigma}")

# Truncated SVD로 Sigma 행렬의 특이값을 4개로 하여 Truncated SVD 수행.
num_components = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix, k=num_components)
print(f"\nTruncated SVD 분해 행렬 차원 : {U_tr.shape, Sigma_tr.shape, Vt_tr.shape}")
print(f"\nTruncated SVD Sigma값 행렬 : {Sigma_tr}")
matrix_tr = np.dot(np.dot(U_tr, np.diag(Sigma_tr)), Vt_tr)  # output of TuncatedSVD

print(f"\nTruncated SVD로 분해 후 복원 행렬 : \n{matrix_tr}")
```
* 사이킷런 TruncatedSVD 클래스를 이용한 변환
```python
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_ftrs = iris.data

# 2개의 주요 component로 TruncatedSVD 변환
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)

# Scatter plot 2차원으로 TruncatedSVD 변환 된 데이터 표현. 품종은 색깔로 구분
plt.scatter(x=iris_tsvd[:,0], y=iris_tsvd[:,1], c=iris.target)
plt.xlabel("TruncatedSVD Component 1")
plt.ylabel("TruncatedSVD Component 2")
plt.show()
```
```python
from sklearn.preprocessing import StandardScaler

# iris 데이터를 StandardScaler로 변환
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_ftrs)

# 스케일링 된 데이터를 기반으로 TruncatedSVD 변환 수행
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_scaled)
iris_tsvd = tsvd.transform(iris_scaled)

# 스케일링 된 데이터를 기반으로 PCA 변환 수행
pca = PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

# TruncatedSVD 변환 데이터를 왼쪽에, PCA변환 데이터를 오른쪽에 표현
fig, (ax1, ax2) = plt.subplots(figsize=(9,4), ncols=2)
ax1.scatter(x=iris_tsvd[:,0], y=iris_tsvd[:,1], c=iris.target)
ax2.scatter(x=iris_pca[:,0], y=iris_pca[:,1], c=iris.target)
ax1.set_title("Truncated SVD Transformed")
ax2.set_title("PCA Transformed")
plt.show()
```
</div>
</details>

</br>


# 참고
## NMF(Non Negative Matrix Factorization)
* NMF는 원본 행렬 내의 모든 원소 값이 모두 양수(0 이상)라는 게 보장되면 다음과 같이 좀 더 간단하게 두 개의 기반 양수 행렬로 분해될 수 있는 기법을 지칭함
$$V_{4\times6} \approx W_{4\times2} \times H_{2\times6}$$
## 행렬분해(Matrix Factorization)
* 행렬분해는 일반적으로 SVD와 같은 행렬 분해 기법을 통칭하는 것.
* 행렬분해를 하게 되면 $W$행렬과 $H$행렬은 일반적으로   
  * 길고 가는 행렬 $W$(즉, 원본 행렬의 행 크기와 같고, 열 크기보다 작은 행렬)와   
  * 작고 넓은 행렬 $H$(즉, 원본 행렬의 행 크기보다 작고, 열 크기와 같은 행렬)로   
분해됨.
* 이렇게 분해된 행렬은 Latent Factor(잠재 요소)를 특성으로 가지게 됨.
* 분해 행렬 $W$는 원본 행에 대해서 이 잠재 요소의 값이 얼마나 되는지에 대응하며, 분해 행렬 $H$는 이 잠재 요소가 원본 열(즉, 원본 속성)로 어떻게 구성됐는지를 나타내는 행렬.
<center>
<img src="C:/Users/user/Desktop/Vocational_Training/FinTech/images/matrix_fact.png">
</center>

### 실습

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_ftrs = iris.data
nmf = NMF(n_components=2, init="nndsvda")
nmf.fit(iris_ftrs)
iris_nmf = nmf.transform(iris_ftrs)
plt.scatter(x=iris_nmf[:,0], y=iris_nmf[:,1], c=iris.target)
plt.xlabel("NMF Component 1")
plt.ylabel("NMF Component 2")
plt.show()
```

</div>
</details>

</br>

군집평가
===
실루엣 분석
---
### 군집평가 - 실루엣 분석
* 실루엣 분석은 각 군집 간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타냄.
* 실루엣 분석은 개별 데이터가 가지는 군집화 지표인 실루엣 계수(silhouette coefficient)를 기반으로 함.
* 개별 데이터가 가지는 실루엣 계수는 해당 데이터가 같은 군집 내의 데이터와 얼마나 가깝게 군집화돼 있고, 다른 군집에 있는 데이터와는 얼마나 멀리 분리돼 있는지를 나타내는 지표임.
### 실루엣 계수
* silhouette coefficient : 개별 데이터가 가지는 군집화 지표
$$s(i) = \frac{b_i-a_i}{Max(a_i, b_i)}$$
* $a_{ij}$는 $i$번째 데이터에서 자신이 속한 클러스터 내의 다른 데이터 포인트 까지의 거리. 즉, $a_{12}는 1번 데이터에서 2번 데이터까지의 거리.
* $a_i$는 $i$번째 데이터에서 **_자신이 속한_** 클러스터 내의 다른 데이터 포인트들의 거리 평균. 즉, $a_i = mean(a_{12}, a_{13}, a_{14})$
* $b_i$는 $i$번째 데이터에서 가장 가까운 **_타_** 클러스터 내의 다른 데이터 포인트들의 거리 평균. 즉, $b_i = mean(b_{15}, b_{16}, b_{17}, b_{18})$
* 두 군집 간의 거리가 얼마나 떨어져 있는가의 값은 $(b_i-a_i)$이며, 이 값을 정규화 하기 위해 $Max(a_i,b_i)$ 값으로 나눔.
* 실루엣 계수는 $-1<s_i<1$ 의 값을 가지며, $1$에 가까워질수록 근처의 군집과 더 멀리 떨어져 있다는 것이고, $0$에 가까울 수록 근처의 군집과 가까워진다는 것. $(-)$ 값은 아예 다른 군집에 데이터 포인트가 할당됐음을 뜻함.
* 즉, 실루엣 계수가 1에 가까울수록 군집화가 잘 되었다는 의미이며, 실루엣 계수가 음수$(-)$인 경우는 해당 데이터의 군집화가 잘못 되었다는 뜻임.

### 사이킷런 실루엣분석 API
* 사이킷런 실루엣 분석 API
  * sklearn.metrics.silhouette_samples(X, labels, *, metric='euclidean', **kwds)
    * X : 개별 데이터, labels : 군집, metric : 거리 계산법.
    * **각 데이터 포인트의 실루엣 계수**를 계산해 반환함.
  * sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
    * 인자로 X feature 데이터 세트와 각 피처 데이터 세트가 속한 군집 레이블 값인 labels 데이터를 입력해주면 **전체 데이터의 실루엣계수 값을 평균**해 반환함. 즉, np.mean(silhouette_samples())임.   
    * 일반적으로 이 값이 높을수록 군집화가 어느정도 잘 됐다고 판단할 수 있음. 하지만 무조건 이 값이 높다고 해서 군집화가 잘 됐다고 판단할 수 는 없음.
* 실루엣 분석에 기반한 좋은 군집 기준
  * 전체 실루엣 계수의 평균 값, 즉 사이킷런의 silhouette_score() 값은 $0~1$ 사이의 값을 가지며, 1에 가까울수록 좋음.
  * 하지만 전체 실루엣 계수의 평균값과 더불어 개별 군집의 평균값의 편차가 크지 않아야 함. 즉, 개별 군집의 실루엣 계수 평균값이 전체 실루엣 계수의 평균값에서 크게 벗어나지 않는 것이 중요함.   
  * 만약 전체 실루엣 계수의 평균값은 높지만, 특정 군집의 실루엣 계수 평균값만 유난히 높고 다른 군집들의 실루엣 계수 평균값이 낮으면 좋은 군집화 조건이 아님.

#### 실습 : 붓꽃 데이터에서 실루엣 계수 계산

<details>
<summary>코드 펼치기/접기</summary>
<div markdown="1">

```python
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 실루엣 분석 metric 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_df = pd.DataFrame(data=iris.data, columns=feature_names)

# KMeans 군집화 수행
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, random_state=0).fit(iris_df)
# 데이터당 클러스터 값 할당
iris_df["cluster"] = kmeans.labels_

print(iris_df.shape)
iris_df
```
```python
# iris의 모든 개별 데이터의 실루엣 계수값을 구함
scroe_samples = silhouette_samples(iris.data, iris_df["cluster"])
print(f"silhouette_samples() return 값의 shape : {scroe_samples.shape}")

# iris_df에 실루엣 계수 컬럼 추가
iris_df["silhouette_coeff"] = scroe_samples

# 모든 데이터의 평균 실루엣 계수값을 구함
average_score = silhouette_score(iris.data, iris_df["cluster"])
print(f"붓꽃 데이터셋 Silhouette Analysis Score : {average_score:.3f}")

iris_df

# 데이터 별 실루엣 계수
# 클러스터가 1인 데이터들은 0.8 정도의 실루엣 계쑤를 가지므로 군집화가 어느정도 잘 된 듯 함.
# 하지만 실루엣 계수 평균 값이 0.553인 이유는 
# 다른 클러스터에 할당된 데이터들의 실루엣 계수값이 작아서임.
```
```python
iris_df.groupby("cluster")["silhouette_coeff"].mean()

# 다른 클러스터의 실루엣 계수 평균이 상대적으로 작은 점 확인 가능함.
```
```python
iris_df["silhouette_coeff"].hist()
# setosa는 군집화가 잘 되었지만, verginica와 virsicolor는 잘 되지 않음.
# 가장 오른쪽이 setosa.
```
#### 실루엣 계수 시각화를 통해 최적의 클러스터 수 찾기
* 데이터들의 실루엣 계수를 계산해서 시각화 해주는 함수
```python
# 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features) :

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 계수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()로 리스트에 기재된 클러스터링 수만큼의 sub_figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 계수 시각화
    for ind, n_cluster in enumerate(cluster_lists) :

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title(f"Number of Cluster : {n_cluster} \n\
                            Silhouette Score : {round(sil_avg,3)}")
        axs[ind].set_xlabel("The silhouette coefficeint values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([]) # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx() 형태의 막대 그래프 표현
        for i in range(n_cluster) :
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i)/n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
```
* 클러스터 수 변화시키면서 random 데이터 실루엣 계수 분포 시각화
```python
# make_blobs를 통해 clustering을 위한 4개의 클러스터 중심의 500개 2차원 데이터 셋 생성
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, \
                center_box=(-10.0, 10.0), shuffle=True, random_state=1)

# cluster 개수를 2개, 3개, 4개, 5개 일때의 클러스터별 실루엣 계수 평균값을 시각화
visualize_silhouette([2,3,4,5], X)

# 클러스터의 개수가 2일 때 실루엣 스코어가 가장 높지만,
# 실제 분포를 살펴보면 한쪽으로 치우쳐 있음 → 좋은 분류라 하기 어려움.
# 임의 데이터를 생성할 때 클러스터를 4개로 나누었으므로,
# 실제 데이터에 적합한 클러스터 개수는 실루엣 스코어가 상대적으로 낮은 4개.
# 분류를 적용할 때 주의해야 함.
```
* 클러스터 수 변화시키면서 붓꽃 데이터 실루엣 계수 분포 시각화
```python
from sklearn.datasets import load_iris

iris=load_iris()
visualize_silhouette([2,3,4,5], iris.data)

# 붓꽃 데이터 역시 본 데이터가 3개의 레이블을 가지고 있으므로,
# 가장 적합한 클러스터 개수는 상대적으로 실루엣 점수가 낮더은 3개임을 알 수 있음. 
```

</div>
</details>

</br>
