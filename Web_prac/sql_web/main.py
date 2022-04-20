from flask import Flask, redirect, render_template, request, url_for
from modules import mod_sql  #main.py 파일과 같은 폴더에 있는 modules

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup/", methods=["GET"])
def singup():
    return render_template("signup.html")

@app.route("/signup/", methods=["POST"])
def signup_2():
#      db = pymysql.connect(
#         user = "root",
#         passwd = "bornin1992`",
#         host = "localhost",
#         db = "AN"
#     )

#     cursor = db.cursor(pymysql.cursors.DictCursor)
#    
    
    db = mod_sql.Database()

    _id = request.form["_id"]
    _pw = request.form["_pw"]
    _name = request.form["_name"]
    _phone = request.form["_phone"]
    _adrs = request.form["_adrs"]
    _gender = request.form["_gender"]
    _age = request.form["_age"]
    _regstdate = request.form["_regstdate"]


    insert_sql = """
                    INSERT INTO user_info
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                 """
    _values = [_id, _pw, _name, _phone, _adrs, _gender, _age, _regstdate]
    
    db.execute(insert_sql, _values)
    db.commit()
    # db.close()
    
    return redirect(url_for("index"))

@app.route("/login/", methods=["POST"])
def login():
    #html에서 form태그 수정, action, method 입력

    # db = pymysql.connect(
    #     user = "root",
    #     passwd = "bornin1992`",
    #     host = "localhost",
    #     db = "AN"
    # )

    # cursor = db.cursor(pymysql.cursors.DictCursor)  모듈이용으로 인해
    #                                                   주석처리

    db = mod_sql.Database()
    _id = request.form["_id"]
    _pw = request.form["_pw"]


    select_sql = f"""
                    SELECT * FROM user_info WHERE ID = "{_id}"
                 """

    q = db.executeAll(select_sql)
    # result = cursor.fetchall()
    # db.close()

    if q:
        if q[0].get("PW") == _pw :
            return render_template("welcome.html",
                                    name=q[0].get("name"),
                                    _id=q[0].get("ID"))
        else :
            return redirect(url_for("index"))
    else :
        return redirect(url_for("index"))
    


    #cursor = db.cursor() 딕셔너리 커서로 해도 됨.(pymysql.cursors.DictCursor)
    #select_sql = """
    #                SELECT * FROM user_info WHERE ID = "{_id}" AND PW = "{_pw}"
    #             """       #AND문으로 조건 2개 동시에 걸 수 있음.
    #cursor.execute(select_sql)
    #result = cursor.fetchall()
    #db.close()
    #if result :
    #   return "login"
    #else :
    #   return "Fail"

@app.route("/update/")
def update() :
    db = mod_sql.Database()
    id = request.args["_id"]
    SELECT_sql = f"""
                    SELECT * FROM user_info WHERE ID = "{id}"
                """
    result = db.executeAll(SELECT_sql)
    return render_template("update.html",
                            info = result[0])

@app.route("/update/", methods=["POST"])
def update2() :
    db = mod_sql.Database()

    _id = request.form["_id"]
    _pw = request.form["_pw"]
    _name = request.form["_name"]
    _phone = request.form["_phone"]
    _adrs = request.form["_adrs"]
    _gender = request.form["_gender"]
    _age = request.form["_age"]

    update_sql = f"""
                    UPDATE user_info SET 
                    PW = "{_pw}", name = "{_name}",
                    phone = "{_phone}", adrs = "{_adrs}",
                    gender = "{_gender}", age = "{_age}" 
                    WHERE ID = "{_id}"
                 """
    db.execute(update_sql)
    db.commit()
    return redirect(url_for("index"))

@app.route("/delete/", methods=["GET"])
def delete() :
    _id = request.args["_id"]
    return render_template("delete.html",
                            _id = _id)

@app.route("/delete/", methods=["POST"])
def delete2() :
    db = mod_sql.Database()
    _id = request.form["_id"]
    _pw = request.form["_pw"]

    select_sql = f"""
                    SELECT * FROM user_info WHERE ID = "{_id}"
                  """

    delete_sql = f"""
                    DELETE FROM user_info WHERE ID = "{_id}"
                  """

    q = db.executeAll(select_sql)

    if q[0].get("PW") == _pw :
        db.execute(delete_sql)
        db.commit()
        return redirect(url_for("index"))
    else :
        return "비밀번호 오류"

@app.route("/view/", methods=["GET"])
def _view() :
    db = mod_sql.Database()
    
    join_sql = """
                    SELECT user_info.name, user_info.adrs, user_info.age,
                    adrs_info.regidnt_count
                    FROM user_info LEFT JOIN adrs_info
                    ON user_info.adrs = adrs_info.adrs
                """

    result = db.executeAll(join_sql)
    key = list(result[0].keys())
    _len = range(len(result))
    return render_template("view2.html", info = result, keys = key,
                                        len = _len)

app.run(port=80, debug=True)