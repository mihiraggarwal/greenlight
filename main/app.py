from flask import Flask, request
from flask_cors import CORS

from main import main

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Welcome to Greenlight!"

@app.route("/solve", methods=["POST"])
def solve():
    data = request.json
    company = data.get("company", "").lower()
    link1 = data.get("link1", "").lower()
    link2 = data.get("link2", "").lower()

    if not company or not link1:
        return {"error": "Company and link1 are required."}, 400

    try:
        main(company, link1, link2)
    except Exception as e:
        print(f"Error: {e}")
    
    return "Done", 200