from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ✅ This will open your website
@app.route('/')
def home():
    return render_template('index.html')

# ✅ This handles prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)[0]

    return jsonify({"stress": int(result)})

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)