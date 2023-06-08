"""
Simple app for model testing. To be ran after `preporcessing.py` and `main.py`
The app provides a simple way to test the trained model.
[!] If path_to_save was changed, MODEL_PATH variable should be updated.

How to run:
> flask --app app run

How to use:
In web-browser open:
http://127.0.0.1:5000/post/<account_id>/<email_sentence_embeddings>
where:
    - account_id is account id from the processed dataset,
    - email_sentence_embeddings is string in the format from inital dataset.
The resulting page will show the following message:
    Predicted contact reason: <contact_reason>
"""

from flask import Flask

from src.model import ContactReasonPredictionModel

MODEL_PATH = "data/models/model.joblib"

app = Flask(__name__)

model = ContactReasonPredictionModel()
model.load(MODEL_PATH)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/post/<int:account_id>/<string:email_sentence_embeddings>")
def post(account_id, email_sentence_embeddings):
    prediction = model.forward(account_id, email_sentence_embeddings)
    return f'Predicted contact reason: "{prediction}"'
