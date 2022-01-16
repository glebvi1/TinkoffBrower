from flask import Flask, render_template, request
from time import time

app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    scored = retrieve(query)
    results = [doc.format()+['%.2f' % scr] for doc, scr in scored]

    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Tinkoff',
        results=results
    )


if __name__ == '__main__':
    from data.search import retrieve, build_index
    build_index()
    app.run(host='0.0.0.0', port=80)
