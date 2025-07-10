import os
import json
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to Greenlight Chat API!"

@app.route("/", methods=["POST"])
def index():
    query = request.json.get("query")
    company = request.json.get("company")

    vectorstore = FAISS.load_local(f"../data/{company}/final/vectorstore", GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key), allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=gemini_api_key)
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    if not qa_chain:
        return jsonify({"error": "Company not found"}), 404
    
    result = qa_chain(query)

    return jsonify({
        "answer": result["result"],
    })

@app.route("/master", methods=["POST"])
def master():
    company = request.json.get("company")
    json_path = f"../data/{company}/final/master.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

@app.route("/logfile", methods=["POST"])
def logfile():
    company = request.json.get("company")
    log_path = f"../data/{company}/final/logfile"
    with open(log_path, "r", encoding="utf-8") as f:
        data = f.read()

    return jsonify(data)