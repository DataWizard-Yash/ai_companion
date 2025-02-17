from flask import Flask, request, jsonify, render_template
from src.app.companion_service import get_ai_response
import os

# Set template and static folder paths relative to the current file location.
# Since main.py is in src/app, templates are one directory up in src/templates.
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates')
)

@app.route('/')
def index():
    return render_template('index.html')  # just the template name

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    result = get_ai_response(user_input)
    return jsonify(result)

if __name__ == "__main__":
    # Running Flask on port 8000
    app.run(port=8000, debug=True)
