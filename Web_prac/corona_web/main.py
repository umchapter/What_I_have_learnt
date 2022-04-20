from cProfile import label
import mimetypes
from flask import Flask, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

corona_df = pd.read_csv("corona_trmd.csv")



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/corona/")
def corona() :
    return corona_df.to_html()

@app.route("/corona_dict/")
def corona_dict() :
    corona_dict = corona_df.head(10).to_dict()
    return render_template("corona_dict.html", result = corona_dict)

@app.route("/img/")
def img() :  
    decide_cnt = corona_df.tail(10)["일일확진자"].values.tolist()
    state_dt = corona_df.tail(10)["등록일시"].values.tolist()
    x = []
    for i in state_dt :
        i = i.replace(i[10:], "")
        i = i.replace(i[:5], "") 
        x.append(i)
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="w", top=False, bottom=False,
                left=False, right=False)
    ax1.plot(x, decide_cnt)
    ax2.bar(x, decide_cnt)

    ax.set_xlabel("2022-02", fontsize=18)
    ax.set_ylabel("Daily Decided", fontsize=16)
   
    img_1 = BytesIO()
    plt.savefig(img_1, format="png", dpi=200)
    img_1.seek(0)
        
    return send_file(img_1, mimetype="image/png")


app.run()