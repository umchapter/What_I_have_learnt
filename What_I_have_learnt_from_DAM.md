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
