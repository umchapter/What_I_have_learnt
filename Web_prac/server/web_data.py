from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index() :
    return render_template('index.html')

@app.route('/second/')
def second() :
    _id = request.args.get("id")
    _pw = request.args.get("pw")
    print(_id, _pw)
    return render_template('second.html', id = _id, pw = _pw)
    '''
    if (_id == "asd" and _pw == "qwe") :
        return render_template('second.html')
    else :
        return "Login Failed"
    '''

@app.route('/third/', methods=["POST"])
def third() :
    _id = request.form["id"]
    _pw = request.form["pw"]
    print(_id, _pw)
    if _id == "asd" and _pw == "qwe" :
        return "Hello World"
    else :
        return "Login Failed"

app.run