import os
import nltk
import json
import lxml
import spacy
import torch
import PyPDF2
import pathlib
import logging
import requests
import textstat

from google import genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.genai import types
from collections import Counter
from nltk.corpus import stopwords
from urllib3.util.retry import Retry
from nltk.tokenize import sent_tokenize
from requests.adapters import HTTPAdapter
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, BertTokenizer, BertForSequenceClassification, RobertaTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification

load_dotenv()

company = "adani green"
link = "adanigreenenergy.com"
# link2 = "unilever.com"

merger = PyPDF2.PdfMerger()

pdf1 = f"data/{company}/annual_report.pdf"
pdf2 = f"data/{company}/esg_report.pdf"

pdfs = [pdf1, pdf2]

print("Merging PDFs")
for pdf in pdfs:
    merger.append(pdf)

merger.write(f"data/{company}/final_doc.pdf")
merger.close()
print("PDFs merged successfully.\n")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
filepath = pathlib.Path(f'data/{company}/final_doc.pdf')

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

if not os.path.exists(f"data/{company}/final"):
    os.makedirs(f"data/{company}/final")

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, filename=f"data/{company}/final/logfile", encoding="utf-8")

master = {}

def extract(pdf_path):

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    nltk.download("punkt_tab")

    print("Extracting text from PDF...\n")
    logging.info("Extracting text from PDF...\n")

    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        p = 0
        for page in pdf_reader.pages:
            text += page.extract_text().replace("\n", " ") + " "
            p += 1

            if p % 20 == 0:
                print(f"Extracted {p} pages")
        
    print("Extraction complete.\n")
    logging.info("Extraction complete.\n")
    
    sentences = sent_tokenize(text)
    return sentences

def get_gemini():
    
    prompt = "Extract only the sentences from this report where sustainability claims may exhibit greenwashing—such as vague language, lack of supporting data, selective reporting, or misleading comparisons. Do not include any explanations, summaries, or additional text—only return the extracted passages. False positives are acceptable so return as many as you can. Also return the broad section of their work this sentence relates to. Return the results in JSON with each array containing the sentence and the section it relates to, with keys 'text' and 'section' respectively. The section should be one out of the following 12: ['cloud computing', 'artificial intelligence', 'software and services', 'semiconductors and hardware', 'supply chain and logistics', 'digital transformation', 'corporate ESG reporting', 'e-waste and recycling', 'blockchain and web3', smart devices and iot, green energy and carbon offsets, 'employee and office sustainability']."

    print("Querying Gemini...")
    logging.info("Querying Gemini...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",

        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),

            prompt
        ]
    )

    print("Received content from Gemini.\n")
    logging.info("Received content from Gemini.\n")

    # future: instead of sections from gemini, use a proper classifier

    for i in range(len(response.text)-1, -1, -1):
        if response.text[i] == '}':
            rt = response.text[8:i+1] + ']'
            break

    with open(f"data/{company}/sentences.json", "w", encoding="utf-8") as f:
        f.write(rt)

    rj = json.loads(rt, strict=False)
    return rj

def classify_esg(texts):
    tokenizer = BertTokenizer.from_pretrained('esg_bert_binary/tokenizer')
    model = BertForSequenceClassification.from_pretrained('esg_bert_binary/model')

    c = 0
    new_texts = []

    print("Classifying text into ESG and non-ESG...")
    logging.info("Classifying text into ESG and non-ESG...")
    for i in range(len(texts)):
        inputs = tokenizer(texts[i]['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        
        output = model(**inputs)
        logits = output.logits
        pred = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(pred, dim=-1).item()

        if pred_id == 0:
            new_texts.append(texts[i])

        c += 1
        if c % 5 == 0:
            print(c, end=' ')

    print("\nSuccessfully classified text into ESG and non-ESG.\n")
    logging.info("Successfully classified text into ESG and non-ESG.")
    return new_texts

def claims(texts):

    c = 0

    tokenizer = RobertaTokenizer.from_pretrained('climatebert/environmental-claims')
    model = RobertaForSequenceClassification.from_pretrained('climatebert/environmental-claims')

    preds = []

    texts_0 = []

    print("Classifying text into claims and non-claims...")
    logging.info("Classifying text into claims and non-claims...")
    for i in range(len(texts)):
        inputs = tokenizer(texts[i]['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        
        output = model(**inputs)
        logits = output.logits
        pred = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(pred, dim=-1).item()

        if pred_id == 0:
            texts_0.append(texts[i])
        
        c += 1
        if c % 5 == 0:
            print(c, end=' ')

        preds.append(pred_id)
    
    texts = [texts[i] for i in range(len(texts)) if preds[i] == 1]

    with open(f"data/{company}/claims.json", "w") as f:
        f.write(json.dumps(texts))
    
    print("\nSuccessfully classified text into claims and non-claims.\n")
    logging.info("Successfully classified text into claims and non-claims.")
    return texts_0, texts

def actions(texts):
    c = 0
    
    tokenizer = AutoTokenizer.from_pretrained("ESGBERT/EnvironmentalBERT-action")
    model = AutoModelForSequenceClassification.from_pretrained("ESGBERT/EnvironmentalBERT-action")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    non_action = []
    action = []

    print("Classifying text into actions and non-actions...")
    logging.info("Classifying text into actions and non-actions...")
    for i in range(len(texts)):
        pred_id = pipe(texts[i]['text'], padding=True, truncation=True)[0]['label']

        if pred_id == "action":
            action.append(texts[i])
        else:
            non_action.append(texts[i])

        c += 1
        if c % 5 == 0:
            print(c, end=' ')

    with open(f"data/{company}/actions.json", "w") as f:
        f.write(json.dumps(action))

    with open(f"data/{company}/non_action.json", "w") as f:
        f.write(json.dumps(non_action))
    
    print("\nSuccessfully classified text into actions and non-actions.\n")
    logging.info("Successfully classified text into actions and non-actions.")
    return non_action, action

def finbert_9(texts):
    c = 0

    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels=9)

    device = 0 if torch.cuda.is_available() else 1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    scores = {}

    print("Classifying text into 9 categories...")
    logging.info("Classifying text into 9 categories...")
    for i in range(len(texts)):
        try:
            result = pipe(texts[i]['text'], padding=True, truncation=True)
        except:
            continue

        if result[0]['label'] in scores.keys():
            scores[result[0]['label']] += result[0]['score']
        else:
            scores[result[0]['label']] = result[0]['score']
        
        c += 1
        if c % 5 == 0:
            print(c, end=' ')

    print("\nSuccessfully classified text into 9 categories.\n")
    logging.info("Successfully classified text into 9 categories.")
    return scores

### NEWS ###

def extract_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    keywords = []
    action_words = []
    numbers = []
    
    print("Extracting keywords from text...")
    logging.info("Extracting keywords from text...")
    for token in doc:
        if token.text.lower().strip() == company.lower():
            continue
        if token.pos_ in ["NOUN", "PROPN"]:  # Key ESG nouns or proper nouns
            keywords.append(token.text)
        if token.pos_ == "VERB":  # Action words (e.g., reduce, invest, eliminate)
            action_words.append(token.lemma_)
        if token.like_num:  # Numbers (e.g., 50% reduction, 2030 target)
            numbers.append(token.text)
    
    print("Keywords extracted.\n")
    logging.info("Keywords extracted.")
    return {
        "keywords": Counter(keywords).most_common(10),
        "action_words": Counter(action_words).most_common(5),
        "numbers": Counter(numbers).most_common(5)
    }

def web(field, kw, cter):
    query = f"{company} {' '.join([k[0] for k in kw["keywords"][:2]])} -site:{link}" # -site:{link2}"

    if cter < 100:
        apikey = os.getenv("NEWS_API_KEY")
        cx = os.getenv("NEWS_CX")
    elif cter < 200:
        apikey = os.getenv("NEWS_API_KEY2")
        cx = os.getenv("NEWS_CX2")
    else:
        apikey = os.getenv("NEWS_API_KEY3")
        cx = os.getenv("NEWS_CX3")

    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={apikey}&cx={cx}&num=2&siteSearch={link}&siteSearchFilter=e&hl=en"

    print("Fetching news URLs...")
    logging.info("Fetching news URLs...")
    response = requests.get(url)

    print("URLs fetched.\n")
    logging.info("URLs fetched.")

    return response.json().get("items", [])

def full_article(url):

    try:
        print("Fetching full article...")
        logging.info("Fetching full article: %s", url)
        response = session.get(url, timeout=5)

        if response.status_code == 200:
            response = response.text.encode("utf-8")

            if len(response) > 3000000:
                return ""

            soup = BeautifulSoup(response, "lxml")
            paragraphs = soup.find_all("p")

            full_text = " ".join([p.get_text() for p in paragraphs])

            print("Full article fetched.\n")
            logging.info("Full article fetched.")

            return full_text
    except Exception as e:
        print(f"Error fetching article: {e}")
        return ""
    return ""

def truncate_text(text, max_tokens=600):

    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

    tokens = tokenizer.tokenize(text)
    truncated_text = tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return truncated_text

def validate(news_articles, report_text):

    device = 1 if torch.cuda.is_available() else 0
    entailment_model = pipeline("text-classification", model="roberta-large-mnli", device=device)
    
    print("Validating news articles against the report...")
    logging.info("Validating news articles against the report...")
    validated_articles = []
    
    for article in news_articles:
        article_text = full_article(article["link"])
        if article_text == "":
            continue

        label = entailment_model(f"Hypothesis: {report_text['text']} Premise: {article_text}", truncation=True)[0]["label"]
        
        if label.upper() == "CONTRADICTION":
            validated_articles.append(report_text)

        break

    print("Validation complete.\n")
    logging.info("Validation complete.")
    return validated_articles

### SCORES ###

def relative():
    env = ['adaptation', 'agricultural', 'air quality', 'animal', 'atmospher', 'biodiversity', ' biomass', 'capture', 'ch4', 'climat', 'co2', 'coastal', 'concentration', 'conservation', 'consumption', ' degree', 'depletion', 'dioxide', 'diversity', 'drought ', 'ecolog', 'ecosystem', 'ecosystems', 'emission', ' emissions', 'energy', 'environment', 'environmental', ' flood', 'footprint', 'forest', 'fossil', 'fuel', 'fuels ', 'gas', 'gases', 'ghg', 'global warming', 'green', ' greenhouse', 'hydrogen', 'impacts', 'land use', ' methane', 'mitigation', 'n2o', 'nature', 'nitrogen', ' ocean', 'ozone', 'plant', 'pollution', 'rainfall', ' renewable', 'resource', 'seasonal', 'sediment', 'snow', 'soil', 'solar', 'sources', 'sustainab', 'temperature ', 'thermal', 'trees', 'tropical', 'waste', 'water']
    soc = ['age', 'cultur', 'rac', 'access to', ' accessibility', 'accident', 'accountability', ' awareness', 'behaviour', 'charit', 'civil', 'code of conduct', 'communit', 'community', 'consumer protection ', 'cyber security', 'data privacy', 'data protection', 'data security', 'demographic', 'disability', 'disable ', 'discrimination', 'divers', 'donation', 'education', 'emotion', 'employee benefit', 'employee development', 'employment benefit', 'empower', 'equal', 'esg', ' ethic', 'ethnic', 'fairness', 'family', 'female', ' financial protectio', 'gap', 'gender', 'health', 'human ', 'inclus', 'information security', 'injury', 'leave ', 'lgbt', 'mental well-being', 'parity', 'pay equity', 'peace', 'pension benefit', 'philanthrop', 'poverty', 'privacy', 'product quality', 'product safety', ' promotion', 'quality of life', 'religion', 'respectful ', 'respecting', 'retirement benefit', 'safety', ' salary', 'social', 'society', 'supply chain transparency', 'supportive', 'talent', 'volunteer', ' wage', 'welfare', 'well-being', 'wellbeing', 'wellness ', 'women', 'workforce', 'working conditions']
    gov = ['audit', 'authority', 'practice', ' bribery', 'code', 'compensation', 'competition', ' competitive', 'compliance', 'conflict of interest', ' control', 'corporate', 'corruption', 'crisis', 'culture ', 'decision', 'due diligence', 'duty', 'ethic', ' governance', 'framework', 'issue', 'structure', ' guideline', 'integrity', 'internal', 'lead', 'legal', ' lobby', 'oversight', 'policy', 'politic', 'procedure', 'regulat', 'reporting', 'responsib', 'right', ' management', 'sanction', 'stake', 'standard', ' transparen', ' vot', 'whistleblower', 'accounting', ' accountable', 'accountant', 'accounted']

    tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english', decode_error='ignore')
    tfidf_vectorizer.fit_transform([filepath])
    total = sum(list(tfidf_vectorizer.vocabulary_.values()))

    env_score = 0
    for word in env:
        try:
            env_score += (tfidf_vectorizer.vocabulary_[word] / total)
        except KeyError:
            pass

    env_score = env_score / len(env)
    while env_score < 0.1:
        if env_score != 0:
            env_score *= 10
        else:
            break
            
    soc_score = 0
    for word in soc:
        try:
            soc_score += (tfidf_vectorizer.vocabulary_[word] / total)
        except KeyError:
            pass

    soc_score = soc_score / len(soc)
    while soc_score < 0.1:
        if soc_score != 0:
            soc_score *= 10
        else:
            break
    
    gov_score = 0
    for word in gov:
        try:
            gov_score += (tfidf_vectorizer.vocabulary_[word] / total)
        except KeyError:
            pass
    
    gov_score = gov_score / len(gov)
    while gov_score < 0.1:
        if gov_score != 0:
            gov_score *= 10
        else:
            break

    return env_score, soc_score, gov_score

def ecotrix_score(sentences, esg_texts):
    h1 = len(esg_texts) / len(sentences) # already b/w 0 and 1
    
    text = filepath.read_bytes().decode('utf-8', errors='ignore')
    h2 = textstat.flesch_reading_ease(text) / 100 # to make it b/w 0 and 1

    master["flesch_reading_ease"] = {
        "explanation": "The Flesch Reading Ease score of the text divided by 100",
        "score": h2
    }

    nltk.download('vader_lexicon')

    print("Calculating sentiment score...")
    logging.info("Calculating sentiment score...")
    sia = SentimentIntensityAnalyzer()
    h3 = sia.polarity_scores(json.dumps(esg_texts))['compound']
    h3 = (h3 + 1) / 2 # convert to [0, 1] range
    print("Sentiment score calculated.\n")
    logging.info("Sentiment score calculated.\n")

    master["sentiment_score"] = {
        "explanation": "The normalized sentiment score of the text",
        "score": h3
    }
    
    prompt = "From this document, extract the number of board members and the number of female board members. Return the results in JSON, with keys being 'num_board' and 'num_female' respectively. If you cannot find the answer, don't add the key in the JSON."

    print("Querying Gemini for board details...")
    logging.info("Querying Gemini for board details...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",

        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),

            prompt
        ]
    )
    print("Received content from Gemini.\n")
    logging.info("Received content from Gemini.")

    for i in range(len(response.text)-1, -1, -1):
        if response.text[i] == '}':
            rt = response.text[8:i+1]
            break

    rj = json.loads(rt, strict=False)
    
    h4 = 0
    h5 = 0

    if 'num_board' in rj.keys():
        h4 = int(rj['num_board'])
        h4 = h4 / 15 # max board size is 15 so this brings it to [0, 1]

    if 'num_female' in rj.keys():
        h5 = int(rj['num_female'])
        h5 = h5 / 15 # max board size is 15 so this brings it to [0, 1]

    master["board_members"] = {
        "explanation": "The score of board members",
        "score": h4
    }

    master["female_board_members"] = {
        "explanation": "The score of female board members",
        "score": h5
    }

    # coefficients taken from the paper, normalized to add to 1
    a = 3.33
    b = -0.33
    c = 0.67
    d = -0.67
    e = -2

    return a * h1 + b * h2 + c * h3 + d * h4 + e * h5


def llm_score(esg_texts, texts_0, texts_1, na, non_action, action, action_contradicted, not_action_contradicted):
    # things to consider:
    # - esg texts that were non claims - little impact
    # - esg texts that were actions and contradicted by news - high impact
    # - esg texts that were non actions and contradicted by news - high impact
    # - esg texts that were non actions and not contradicted by news - medium impact
    # - esg texts that were claims but not actions - little impact
    
    # these coefficients are decided heuristically
    a = 0.05
    b = 0.4
    c = 0.3
    d = 0.2
    e = 0.05

    l1 = 0
    l2 = 0
    len2 = len(not_action_contradicted)

    tbl2 = 0
    tbl2a = []
    while l2 < len2:
        if non_action[l1] == not_action_contradicted[l2]:
            l2 += 1
        else:
            tbl2 += 1
            tbl2a.append(non_action[l1])
        l1 += 1

    texts_0_sep = finbert_9(texts_0)
    action_contradicted_sep = finbert_9(action_contradicted)
    not_action_contradicted_sep = finbert_9(not_action_contradicted)
    tbl2_sep = finbert_9(tbl2a)
    na_sep = finbert_9(na)

    a_mul = 0
    b_mul = 0
    c_mul = 0
    d_mul = 0
    e_mul = 0
    
    if len(esg_texts) != 0:
        a_mul = len(texts_0) / len(esg_texts)
    if len(action) != 0:
        b_mul = len(action_contradicted) / len(action)
    if len(non_action) != 0:
        c_mul = len(not_action_contradicted) / len(non_action)
        d_mul = tbl2 / len(non_action)
    if len(texts_1) != 0:
        e_mul = len(na) / len(texts_1)

    final_score = a * a_mul + b * b_mul + c * c_mul + d * d_mul + e * e_mul

    sec_scores = {}
    for i in range(len(texts_0)):
        if texts_0[i]['section'] in sec_scores.keys():
            sec_scores[texts_0[i]['section']] += a
        else:
            sec_scores[texts_0[i]['section']] = a
    
    for i in range(len(action_contradicted)):
        if action_contradicted[i]['section'] in sec_scores.keys():
            sec_scores[action_contradicted[i]['section']] += b
        else:
            sec_scores[action_contradicted[i]['section']] = b
    
    for i in range(len(not_action_contradicted)):
        if not_action_contradicted[i]['section'] in sec_scores.keys():
            sec_scores[not_action_contradicted[i]['section']] += c
        else:
            sec_scores[not_action_contradicted[i]['section']] = c

    for i in range(len(tbl2a)):
        if tbl2a[i]['section'] in sec_scores.keys():
            sec_scores[tbl2a[i]['section']] += d
        else:
            sec_scores[tbl2a[i]['section']] = d
    
    for i in range(len(na)):
        if na[i]['section'] in sec_scores.keys():
            sec_scores[na[i]['section']] += e
        else:
            sec_scores[na[i]['section']] = e
    
    total_sec = sum(sec_scores.values())
    for key in sec_scores.keys():
        sec_scores[key] = sec_scores[key] / total_sec

    sep_classes = list(map(lambda x: x.strip(), "Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, Non-ESG".split(',')))

    sep_scores = {}

    for sep_class in sep_classes:

        asep_mul = 0
        bsep_mul = 0
        csep_mul = 0
        dsep_mul = 0
        esep_mul = 0

        if sep_class == "Non-ESG":
            continue
        if sep_class in texts_0_sep.keys():
            asep_mul = texts_0_sep[sep_class]
        if sep_class in action_contradicted_sep.keys():
            bsep_mul = action_contradicted_sep[sep_class]
        if sep_class in not_action_contradicted_sep.keys():
            csep_mul = not_action_contradicted_sep[sep_class]
        if sep_class in tbl2_sep.keys():
            dsep_mul = tbl2_sep[sep_class]
        if sep_class in na_sep.keys():
            esep_mul = na_sep[sep_class]

        class_score = a * asep_mul + b * bsep_mul + c * csep_mul + d * dsep_mul + e * esep_mul
        sep_scores[sep_class] = class_score
    
    total_sep = sum(sep_scores.values())
    for key in sep_scores.keys():
        sep_scores[key] = sep_scores[key] / total_sep

    return final_score, sep_scores, sec_scores

def main():

    # Take input files

    init_gwashes = get_gemini()
    # init_gwashes = json.loads(open(f"data/{company}/sentences.json", "r").read())

    esg_texts = classify_esg(init_gwashes)

    llm_esg = {
        "explanation": "Possible greenwashing texts returned by the LLM, filtered to remove non-ESG texts.",
        "texts": esg_texts
    }

    master["llm_esg"] = llm_esg

    texts_0, texts_1 = claims(esg_texts)

    llm_claims0 = {
        "explanation": "The non-claims in the possible greenwashing texts returned by the LLM",
        "texts": texts_0
    }

    master["llm_claims0"] = llm_claims0

    llm_claims1 = {
        "explanation": "The claims in the possible greenwashing texts returned by the LLM",
        "texts": texts_1
    }

    master["llm_claims1"] = llm_claims1

    na, aa = actions(texts_1)

    llm_na = {
        "explanation": "The non-actions in the claims in the possible greenwashing texts returned by the LLM",
        "texts": na
    }

    master["llm_na"] = llm_na

    non_action, action = actions(esg_texts)

    llm_non_action = {
        "explanation": "The non-actions in the possible greenwashing texts returned by the LLM",
        "texts": non_action
    }

    master["llm_non_action"] = llm_non_action

    llm_action = {
        "explanation": "The actions in the possible greenwashing texts returned by the LLM",
        "texts": action
    }

    master["llm_action"] = llm_action

    action_contradicted = []
    not_action_contradicted = []

    print("Total actions: " + str(len(action)))
    print("Total non-actions: " + str(len(non_action)))
    print("")

    json_object = json.dumps(master, indent=4)
    with open(f"data/{company}/final/master.json", "w") as f:
        f.write(json_object)

    cter = 0
    a_i = 0
    for act in action:
        print("Validation pipeline for action: " + str(a_i + 1))
        kw = extract_keywords(act['text'])
        news_val = web(act['section'], kw, cter)

        action_contradicted.extend(validate(news_val, act))
        with open(f"data/{company}/action_contradicted.json", "w") as f:
            f.write(json.dumps(action_contradicted))
        
        cter += 1
        a_i += 1
    
    llm_action_contradicted = {
        "explanation": "The actions in the possible greenwashing texts returned by the LLM that were contradicted by news articles",
        "texts": action_contradicted
    }

    master["llm_action_contradicted"] = llm_action_contradicted

    json_object = json.dumps(master, indent=4)
    with open(f"data/{company}/final/master.json", "w") as f:
        f.write(json_object)

    na_i = 0
    for act in non_action:
        print("Validation pipeline for non-action: " + str(na_i + 1))
        kw = extract_keywords(act['text'])
        news_val = web(act['section'], kw, cter)

        not_action_contradicted.extend(validate(news_val, act))
        with open(f"data/{company}/not_action_contradicted.json", "w") as f:
            f.write(json.dumps(not_action_contradicted))
        na_i += 1
        cter += 1

    llm_not_action_contradicted = {
        "explanation": "The non-actions in the possible greenwashing texts returned by the LLM that were contradicted by news articles",
        "texts": not_action_contradicted
    }

    master["llm_not_action_contradicted"] = llm_not_action_contradicted

    json_object = json.dumps(master, indent=4)
    with open(f"data/{company}/final/master.json", "w") as f:
        f.write(json_object)

    final_score, sep_scores, sec_scores = llm_score(esg_texts, texts_0, texts_1, na, non_action, action, action_contradicted, not_action_contradicted)

    master["llm_score"] = {
        "explanation": "The final score calculated from the first pipeline",
        "score": final_score
    }

    master["llm_separate_scores"] = {
        "explanation": "The separate scores calculated from the first pipeline",
        "scores": sep_scores
    }

    master["llm_section_scores"] = {
        "explanation": "The section scores calculated from the first pipeline",
        "scores": sec_scores
    }

    #### SENTENCE PIPELINE ####

    sentences = extract(filepath)
    sentences = [{"text": i} for i in sentences]

    master["sentences"] = {
        "explanation": "The sentences extracted from the PDF",
        "sentences": sentences
    }

    texts = classify_esg(sentences)

    master["sentences_esg"] = {
        "explanation": "The sentences that are classified as ESG",
        "sentences": texts
    }

    texts_0, texts_1 = claims(texts)

    master["sentences_claims0"] = {
        "explanation": "The sentences that are classified as non-claims",
        "sentences": texts_0
    }

    master["sentences_claims1"] = {
        "explanation": "The sentences that are classified as claims",
        "sentences": texts_1
    }

    non_action, action = actions(texts)

    master["sentences_non_action"] = {
        "explanation": "The sentences that are classified as non-actions",
        "sentences": non_action
    }

    master["sentences_action"] = {
        "explanation": "The sentences that are classified as actions",
        "sentences": action
    }

    total_sentences = len(sentences)
    claims_score = (len(texts_1) / len(sentences))
    action_score = (len(action) / len(sentences))

    json_object = json.dumps(master, indent=4)
    with open(f"data/{company}/final/master.json", "w") as f:
        f.write(json_object)

    env, soc, gov = relative()
    net_action = action_score - claims_score
    relative_score = env + soc + gov / 3

    master["sentences_relative"] = {
        "explanation": "The relative scores calculated from the sentences",
        "scores": {
            "environment": env,
            "social": soc,
            "governance": gov
        },
        "relative_score": relative_score
    }

    master["sentences_net_action"] = {
        "explanation": "The net action score calculated from the sentences",
        "score": net_action
    }

    metrics_score = ecotrix_score(sentences, texts)

    master["metrics_score"] = {
        "explanation": "The metrics score calculated from the sentences",
        "score": metrics_score
    }

    a = -0.15 # relative score -ve
    b = 0.25 # net action -ve
    c = 0.15 # metrics score +ve
    d = 0.75 # llm/final score +ve

    greenwashing_score = a * relative_score + b * net_action + c * metrics_score + d * final_score

    master["greenwashing_score"] = {
        "explanation": "The final greenwashing score",
        "score": greenwashing_score
    }

    json_object = json.dumps(master, indent=4)
    with open(f"data/{company}/final/master.json", "w") as f:
        f.write(json_object)

    print("Greenwashing Score: " + str(greenwashing_score))
    print("Final Score: " + str(final_score))
    print("Relative Score: " + str(relative_score))
    print("Metrics Score: " + str(metrics_score))
    print("Claims Score: " + str(claims_score))
    print("Action Score: " + str(action_score))
    print("Net Action Score: " + str(net_action))

    logging.info("Greenwashing Score: %s", str(greenwashing_score))
    logging.info("Final Score: %s", str(final_score))
    logging.info("Relative Score: %s", str(relative_score))
    logging.info("Metrics Score: %s", str(metrics_score))
    logging.info("Claims Score: %s", str(claims_score))
    logging.info("Action Score: %s", str(action_score))
    logging.info("Net Action Score: %s", str(net_action))
    logging.info("Separate Scores: %s", str(sep_scores))
    logging.info("Section Scores: %s", str(sec_scores))
    # print("Separate Scores:" + sep_scores)
    # print("Section Scores:" + sec_scores)

main()
