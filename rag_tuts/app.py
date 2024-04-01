import os
import sys
from flask import Flask, request, jsonify

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from rag_tuts.chatbot.rag_chatbot import start_chatbot

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    if data is None or 'query' not in data:
        return jsonify({'error': 'Missing or invalid JSON data'}), 400

    user_input = data['query']
    response, is_from_cache, time_taken = start_chatbot(user_input)
    return jsonify({'response': response, 'is_from_cache': is_from_cache, 'time_taken': time_taken})


if __name__ == '__main__':
    app.run(debug=True)
