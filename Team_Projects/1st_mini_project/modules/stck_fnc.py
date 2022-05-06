import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import ndiffs
import pmdarima as pm
from datetime import datetime

# 폰트 설정
plt.rc("font", family="D2Coding")

# 주식 종목명과 티커 데이터
stock_list = pd.read_csv("./data/stock_list.csv", encoding="CP949")

# 날짜 설정
# 영업일 1년 251일, 3년 753일, 대한민국 법정공휴일 15일 반영.
date = pd.date_range(end=datetime.now().date(), periods=900, freq="B").strftime("%Y-%m-%d").to_list()

# 검색 함수
def search(input_) :
    if stock_list.loc[stock_list["Symbol"] == input_]["Symbol"].notnull().values :
        df = fdr.DataReader(symbol=input_, start=date[0], end=date[-1])["Close"]
        name_ = stock_list.loc[stock_list["Symbol"] == input_]["Name"].values[0]
        result = [df, name_]
        return result
    elif stock_list.loc[stock_list["Name"] == input_]["Name"].notnull().values :
        symbol = stock_list.loc[stock_list["Name"]==input_]["Symbol"].values[0]
        df = fdr.DataReader(symbol=symbol, start=date[0], end=date[-1] )["Close"]
        name_ = input_
        result = [df, name_]
        return result
    else :
        return None

# 없을 경우 예시 보여주는 함수
def re_search(input_) :
    examp = stock_list.loc[stock_list["s_Name"].str.contains(input_)][["Symbol", "Name"]]
    return examp

# 로그화 함수
def log_func(df) :
    df_log = np.log1p(df)

    return df_log

# best_diffs 뽑아주는 함수
def best_diffs(df, name, trace=False) :
    best_diffs = {}
    y_train = df[:int(0.7*len(df))]
    y_test = df[int(0.7*len(df)):]

    kpss_diffs = ndiffs(y_train, alpha=0.05, test="kpss", max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test="adf", max_d=6)
    n_diffs = max(kpss_diffs, adf_diffs)
    best_diffs = n_diffs
    if trace == True :
        expln = (f"{name}의 추정된 최적 차수 d = {n_diffs}")        
    else :
        expln = ""

    return best_diffs, expln

# auto arima 적용해주는 함수
def auto_arima_process(df, name, diff, trace=False) : 
    y_train = df[:int(0.7*len(df))]
    y_test = df[int(0.7*len(df)):]

    if trace == True :
        print(f"##### model_{name} #####")
        model = pm.auto_arima(y=y_train, d=diff, start_p=0, max_p=5, start_q=0,
                max_q=5, m=1, seasonal=False, stepwise=True, trace=True)
        model_name = f"model_{name}"
        print()
    else :
        model = pm.auto_arima(y=y_train, d=diff, start_p=0, max_p=5, start_q=0,
                max_q=5, m=1, seasonal=False, stepwise=True, trace=False)
        model_name = f"model_{name}"
    
    return model, model_name


# 1 step forecast 함수
def forecast_one_step(model) :
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)

    return (fc.tolist()[0], np.asarray(conf_int).tolist()[0])

def model_update(df, model) :
    y_train = df[:int(0.7*len(df))]
    y_test = df[int(0.7*len(df)):]
    
    y_pred_list = []
    pred_upper = []
    pred_lower = []

    for i in y_test :
        fc, conf = forecast_one_step(model)
        y_pred_list.append(fc)      # 예측치
        pred_upper.append(conf[1])  # 신뢰구간 상방
        pred_lower.append(conf[0])  # 신뢰구간 하방

        model.update(i)

    pred_df = pd.DataFrame({"pred" : y_pred_list, "upper" : pred_upper, "lower" : pred_lower}, index=y_test.index)

    return pred_df


# 정상화 함수
def re_exp(df, pred_df) :
    df = np.expm1(df)
    df_pred = np.expm1(pred_df)

    return df, df_pred

# 평가(MAPE) 함수
def MAPE(y_test, y_pred, name) :

    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mape_df = pd.DataFrame(mape, index=[f"{name}"], columns=["MAPE Score(%)"])

    return mape_df

# 그래프 함수
def draw_plots(df, df_pred, name, model) :
    y_train = df[:int(0.7*len(df))]
    y_test = df[int(0.7*len(df)):]
    
    mape_df = MAPE(y_test=y_test, y_pred=df_pred["pred"], name=name)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.95, 0.05], subplot_titles=[f"{str(model)[:13]} 학습 내용", "학습 결과(MAPE)"])

    fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, name=f"Train_{name}", mode="lines", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name=f"Test_{name}", mode="lines", line=dict(color="rgba(0,0,30,0.5)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test.index, y=df_pred["pred"], name=f"Pred_{name}", mode="lines", line=dict(color="red", dash="dot", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test.index.tolist() + y_test.index[::-1].tolist(),
                    y=df_pred["upper"].tolist() + df_pred["lower"][::-1].tolist(),
                    name="Confidence Interval",
                    fillcolor="rgba(0,0,30,0.2)",
                    fill="toself",
                    line={"color":"rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=True), row=1, col=1)

    fig.add_trace(go.Bar(x=mape_df.index, y=mape_df["MAPE Score(%)"], name="MAPE Score(%)", text=mape_df["MAPE Score(%)"].round(2)), row=1, col=2)
    fig.layout.annotations[0].update(x=0.1, y=1.1, font={"size":22})
    fig.layout.annotations[1].update(y=1.08, font={"size" : 14})
    fig.write_html(f"./templates/show_{name}.html", include_plotlyjs="cdn", full_html=False)
    html_name = f"show_{name}.html"
    name_ = name
    return html_name, name_


def draw_predict(name, model) :
    df_origins = pd.DataFrame(data=model.predict(5), columns=["5days_prediction"])
    df_origins = df_origins.set_index([pd.date_range(start=datetime.now().date(), periods=5, freq="B")])
    df_pred = np.expm1(df_origins)
    fig = go.Figure()
    

    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred["5days_prediction"], name=f"{str(model)[:13]} 5days Prediction", marker={"color" : '#9467bd'}))
    fig.update_layout(
        title = f'{name} 5일 종가 예측',
        xaxis_tickformat = '%B %d <br>%Y',
        xaxis=dict(dtick="%d"),
        title_font=dict(size=22),
        title_y=0.9
        )

    fig.write_html(f"./templates/pred_{name}.html", include_plotlyjs="cdn", full_html=False)
    pred_html = f"pred_{name}.html"
    return pred_html


def show_forecast(df, name, trace=False) :
    df = log_func(df)
    best_diff, _ = best_diffs(df, name)
    model, model_name = auto_arima_process(df=df, name=name, diff=best_diff, trace=trace)
    forecast_df = model_update(df, model=model)
    df, forecast_df = re_exp(df, forecast_df)
    html_name, name_ = draw_plots(df, forecast_df, name=name, model=model)
    pred_html = draw_predict(name, model)
    
    return html_name, name_, pred_html

# acf_pacf 그리는 함수
def draw_acf_pacf(df, name) :

    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    plot_acf(df, ax=ax[0], title=(name+ "_ACF"))
    plot_pacf(df, method="ywm", ax=ax[1], title=(name + "_PACF"))

    fig.tight_layout()
    
    return plt.show()

# arima_summary 보여주는 함수
def model_summary(model) :
 
    return model.summary()

# model_diagnostics 보여주는 함수
def model_diagnostics(model) : 

    diag = model.plot_diagnostics
    
    return diag