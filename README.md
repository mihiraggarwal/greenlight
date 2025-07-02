# Greenlight

Providing greenwashing scores and insights for Indian companies

## What

Greenwashing, the act of misleading stakeholders about a company’s environmental, social, and governance (ESG) practices, is an increasingly critical problem in sustainability reporting. This project proposes an end-to-end NLP pipeline that attempts to detect potential greenwashing in company reports.  

This was created as part of my final project for CS-4420 (Natural Language Processing) at Ashoka University.

## Why

In recent years, sustainability has become a key corporate concern, driven by growing environmental
awareness and tightening regulatory frameworks. This has led to an increase in the instances of greenwashing, which is difficult to detect because it often involves vague or selective language rather than
outright lies. This project explores whether modern NLP tools can be used to identify potentially greenwashed claims in ESG reports in an automated, interpretable, and ethically responsible manner.

## How to Use

## Architecture

The architecture is built around three major pipelines:

### 1. Main Pipeline (Core Scoring)

This pipeline contributes 75-80% of the final greenwashing score. It consists of the following stages:

**Step 1: LLM-Based Sentence Extraction**

- Annual and sustainability reports (in PDF format) are input into Google Gemini LLM using a custom prompt.
- The prompt instructs the model to extract sentences where sustainability claims may exhibit greenwashing — particularly those with vague language, lack of data, or selective reporting.
- Manual evaluation showed false positives were more common than false negatives. Hence, the prompt explicitly prioritizes recall, trusting that downstream filtering will handle precision.

**Step 2: ESG Relevance Classification**

- Sentences returned by the LLM are filtered using ESG-BERT, a fine-tuned classifier that determines whether a sentence pertains to ESG topics.
- This step removes irrelevant but plausible-looking sentences returned by the LLM, in an attempt to clean the data.

**Step 3: Environmental Claim Detection**

- Each ESG-relevant sentence is passed into ClimateBERT, a RoBERTa-based classifier finetuned to detect whether the sentence constitutes an environmental claim.
- This model was trained on a dataset annotated by multiple domain experts. Thus, as outputs we get those sentences that are environmental claims and those that aren’t.

**Step 4: Action Detection**

- The sentences and the claims are further classified using environmentalBERT/action, another fine-tuned BERT model that determines whether a sentence is an environmental action or not.
- Both, the actions and non-actions will be used in further validations. 
  
**Step 5: Keyword Extraction for News Validation**

- Using spaCy POS tagging, keywords such as nouns and verbs are extracted from claim and action sentences.
- These are used to construct queries that include the company name, sector, and top keywords.
- These queries are sent to the Google Custom Search API, configured to return only news articles and exclude the company’s own domain.

**Step 6: News Scraping and Contradiction Detection**

- The top 2-3 news results for each query are scraped using BeautifulSoup and truncated to a reasonable length.
- For each original sentence and its corresponding news article, Facebook’s RoBERTa-largeMNLI model checks for contradiction.
- This validation step is the core step of the pipeline, as sentences contradicted by news sources contribute strongly to the greenwashing score.

**Step 7: Scoring and ESG Sector Categorization**

- A weighted sum is calculated over:
    - ESG sentences that were not claims
    - Claims that were not actions
    - Actions contradicted by news
    - Non-actions contradicted by news
- FinBERT is used to classify each sentence into one of nine ESG sectors (e.g., renewable energy, digital infrastructure) to allow detailed breakdowns with sector-specific scores.

### 2. Robustness Pipelines

**Pipeline 2: Baseline ESG Signal Analysis**

- Extracts all sentences from the PDF and classifies them into ESG or not.
- Then classifies ESG texts into environmental claims and actions.
- The imbalance between claims and actions serves as a secondary greenwashing signal.
- TF-IDF analysis is used to determine focus on E, S, or G aspects, mainly for reporting purposes.
- The whole point of this pipeline is to flag any major discrepancies between the manual sentence extraction, and LLM querying for relevant sentences.

**Pipeline 3: Structural and Sentiment Indicators**

Computes:
- ESG-to-total sentence ratio
- Readability using Flesch reading ease
- Overall report sentiment via NLTK
- Board diversity (size and number of women directors)

### Deliverable

A Next.js dashboard was developed to display:

- Greenwashing scores + Rudimentary Insights
- Multiple sector-wise score breakdowns
- Transparent reports – master and log files

A Flask + LangChain chatbot was built to allow users to query a master JSON index of intermediate
outputs recorded throughout the pipelines, powered by RAG-based semantic retrieval.

## Limitations

- Lack of a benchmark dataset meant no true evaluation metric. Moreover, the coefficients for the weighted sum couldn’t be learnt and had to be heuristic based.
- Long latency (news validation is slow) which makes real-time results infeasible on the platform currently. News quality is also inconsistent across companies, and news fetching may return noisy/sparse results.
- Subjectivity and interpretability is a standard challenge in language focused problems like these, especially when even human experts may disagree on what constitutes greenwashing.
- While the BERT datasets are built taking context into account, whenever the LLM returns sentences it’s still possible that sentence-level scope misses inter-sentence context

Thus, while results are definitely imperfect, the project lays a foundation for future work in this space.

## References

1. https://www.scitepress.org/Papers/2023/121554/121554.pdf
2. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5024113
3. https://dl.acm.org/doi/pdf/10.1145/3503162.3503163
4. https://uis.brage.unit.no/uis-xmlui/bitstream/handle/11250/3090545/no.uis%3ainspera%3a129718883%3a50794433.pdf
5. https://onlinelibrary.wiley.com/doi/epdf/10.1111/eufm.12509
6. https://link.springer.com/article/10.1007/s11846-023-00718-w
7. https://www.sciencedirect.com/science/article/pii/S1544612324000096
8. https://www.researchcollection.ethz.ch/bitstream/20.500.11850/568978/1/CLE_WP_2022_07.pdf
9. https://www.allenhuang.org/uploads/2/6/5/5/26555246/esg_9-class_descriptions.pdf
10. https://arxiv.org/abs/2209.00507
11. https://aclanthology.org/2024.climatenlp-1.9.pdf
12. https://arxiv.org/abs/2203.16788
13. https://aclanthology.org/2024.finnlp-2.1.pdf