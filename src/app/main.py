from flask import Flask, request, jsonify, render_template
from src.app.companion_service import get_ai_response
import os

app = Flask(__name__, template_folder=os.path.join(
    os.path.dirname(__file__),
    '..',
    'templates'
))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    user_relationship = data.get("relationship")

    result = get_ai_response(user_input, user_relationship)
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
