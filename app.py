from flask import Flask

from src.model import ContactReasonPredictionModel

app = Flask(__name__)

model = ContactReasonPredictionModel()
model.load("data/models/model.joblib")


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/post/<int:account_id>/<string:email_sentence_embeddings>")
def post(account_id, email_sentence_embeddings):
    prediction = model.forward(account_id, email_sentence_embeddings)
    return f'Predicted contact reason: "{prediction}"'
