from flask import Flask, render_template, request, redirect, url_for, session
from modules import mod_sql, stck_fnc
from functools import wraps

# 플라스크 서버
app = Flask(__name__)

def is_loged_in(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        if 'is_loged' in session:
            return func(*args, **kwargs)
        else:
            return redirect('/')
    return wrap

@app.route("/")
def index() :
    return render_template("index.html")

@app.route("/test/")
def test():
    return render_template("chart_page(2).html")

@app.route("/signup/", methods=["GET"])
def singup():
    return render_template("signup.html")

@app.route("/signup/", methods=["POST"])
def signup_2():
 
    db = mod_sql.Database()

    _id = request.form["_id"]
    _pw = request.form["_pw"]
    _name = request.form["_name"]

    insert_sql = """
                    INSERT INTO user_info
                    VALUES (%s, %s, %s)
                 """
    _values = [_id, _pw, _name]
    
    db.execute(insert_sql, _values)
    db.commit()
    
    return redirect(url_for("index"))

@app.route("/sign_in/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = mod_sql.Database()
        _id = request.form["_id"]
        _pw = request.form["_pw"]

        select_sql = f"""
                        SELECT * FROM user_info WHERE ID = "{_id}"
                    """

        q = db.executeAll(select_sql)

        if q:
            if q[0].get("PW") == _pw :
                session['is_loged'] = True
                session['name'] = q[0].get("name")
                return render_template("welcome.html",
                                        name=q[0].get("name"),
                                        _id=q[0].get("ID"))
            else :
                return redirect(url_for("index"))
        else :
            return redirect(url_for("index"))
    else:
        if session["is_loged"] == True :
            return render_template("welcome.html", name=session["name"])
        else :
            return redirect(url_for("index"))
    
@app.route("/task/", methods=["GET", "POST"])
def task() :
    if request.method == "POST":
        input_ = request.form["input_"]
        result = stck_fnc.search(input_)

        if result == None :
            result = stck_fnc.re_search(input_)
            return render_template("select.html", df_= result , df_t = result.empty)

        elif result != None :
            df_1 = result[0]
            name_1 = result[1]
            html_name_, name_, pred_html_=  stck_fnc.show_forecast(df_1, name_1)
            return render_template("chart_page.html", html_name = html_name_,
                                                    name = name_,
                                                    pred_html = pred_html_)
    else:
        if session["is_loged"] == True :
            return render_template("welcome.html", name=session["name"])
        else :
            return redirect(url_for("index"))

@app.route("/expl/")
@is_loged_in
def expl() :
    return render_template("ARIMA.html")

@app.route("/intrprt/")
@is_loged_in
def intrprt() :
    return render_template("model_intrprt.html")

app.config['SECRET_KEY'] = "secret"
app.run(debug=True)