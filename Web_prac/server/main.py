from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index() :
    return "Hello, World"

@app.route('/second/')
def second_page() :
    return "Second Page"

@app.route('/table/')
def third_page() :
    return render_template("table.html")

app.run