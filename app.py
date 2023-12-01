from flask import Flask, render_template, request, jsonify
from bot import export_answer

app = Flask(__name__, template_folder='templates')

@app.get("/")
def index_get():
    return render_template('index.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("msg")
    response = export_answer(text)
    msg = {"answer": response}
    return jsonify(msg)

if __name__ == "__main__":
    app.run(debug=True)