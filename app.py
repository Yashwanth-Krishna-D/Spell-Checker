"""
Routes
------
  GET  /           → serves the terminal UI (index.html)
  POST /check      → accepts JSON { "word": "..." }
                     returns JSON with candidates + best correction
"""

import os
from flask import Flask, request, jsonify, render_template
import sys
import os
# print("RUNNING FROM:", os.getcwd())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spellchecker.main import spell_check

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(force=True, silent=True) or {}
    word = data.get("word", "").strip()
    top  = int(data.get("top", 10))

    if not word:
        return jsonify({"error": "No word provided"}), 400

    result = spell_check(word, top_n=top, verbose=True)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
