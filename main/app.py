import os
from flask import Flask, request
from flask_cors import CORS
from threading import Thread
from pathlib import Path

from main import main

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Welcome to Greenlight!"

@app.route("/submit", methods=["POST"])
def solve():
    data = request.form
    files = request.files
    company = data.getlist("company")[0].lower()
    link1 = data.getlist("url1")[0].lower()
    link2 = data.getlist("url2")[0].lower()
    annual_report = files.getlist("annual")[0]
    esg_report = files.getlist("esg")[0] if "esg" in files else None

    if not company or not annual_report or not link1:
        return {"error": "Company and one URL are required."}, 400
    
    cdir = Path(__file__).parent
    pdir = cdir.parent
    os.makedirs(f"{pdir}/data/{company}", exist_ok=True)

    annual_report.save(f"{pdir}/data/{company}/annual_report.pdf")

    if esg_report:
        esg_report.save(f"{pdir}/data/{company}/esg_report.pdf")

    try:
        Thread(target=main, args=(company, link1, link2)).start()
    except Exception as e:
        print(f"Error: {e}")
    
    return "Done", 200