from flask import Flask, abort, request, redirect, url_for
from flask import render_template
from data_cleaner import DataCleaner

app = Flask(__name__)
@app.route('/')
def index():
    name = request.args.get('name')
    if not name:
        name = '<unknown>'
    return render_template('home.html', name=name)
@app.route('/hello/<name>')
@app.route('/hello/')
def hello(name=None):
    if name is None:
    # If no name is specified in the URL, attempt to retrieve it
    # from the query string.
        name = request.args.get('name')
    if name:
        return 'Hello, %s' % name
    else:
    # No name was specified in the URL or the query string.
        abort(404)
@app.route('/submit', methods=['POST'])
def intent_classify():
    content = request.form['content']
    print (content)
    # if not content:
    #     content = '<unknown>'
    return redirect(url_for('/'))
    # return redirect("/#about", content = content)
if __name__ == '__main__':
    app.run(debug=True)