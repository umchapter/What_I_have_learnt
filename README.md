# What I have learnt

데이터 분석을 위한 SQL, MongoDB, Python의 활용을 연습합니다.   
기본적 분석모형의 연습 경험은 아래를 통해 보실 수 있습니다.

<details>
<summary>(스압주의) 펼치기/접기</summary>
<div markdown="1">



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

</div>
</details>

</br>
