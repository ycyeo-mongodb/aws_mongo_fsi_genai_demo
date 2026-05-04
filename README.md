# MongoDB GenAI for Financial Services Workshop

A hands-on workshop for financial institutions embarking on GenAI projects. Build three production-relevant AI use cases on **MongoDB Atlas**: an internal FAQ chatbot (RAG), AI-enhanced credit scoring, and KYC document verification.

## Architecture

```
Browser (FSI Web App)  →  FastAPI Backend (Python)  →  MongoDB Atlas
                                    ↓                        ↕
                          API Gateway / Lambda       Atlas Vector Search
                                    ↓                  Aggregation Pipelines
                          Amazon Bedrock (Claude)
```

- **MongoDB Atlas** — operational documents, vector embeddings, and Atlas Search indexes in a single cluster
- **Voyage AI** (`voyage-4-large`) — multilingual embeddings for semantic retrieval
- **Amazon Bedrock** (Claude via API Gateway) — LLM generation for FAQ answers and credit explanations

## The Web Application

The frontend is a realistic **internet banking web application** with two portals:

| Portal | Audience | Features |
|---|---|---|
| **Internet Banking** | Bank customers | Account dashboard, transactions, transfers — with an AI-powered **FAQ Chatbot** popup (bottom-right) for instant policy Q&A |
| **Employee Portal** | Bank staff | **Credit Scoring** — select a customer and loan, get AI-enhanced risk assessment with explainable decisions. **KYC Verification** — select a document, run vector similarity search for duplicate detection |

## Three AI Modules

### Module 1: Internal FAQ Chatbot (RAG)
Retrieval-augmented generation over bank policy documents. Supports **multilingual** queries (English + Khmer). Uses Atlas Vector Search for retrieval and an LLM for grounded answer generation.

### Module 2: AI-Enhanced Credit Scoring
Aggregation pipelines (`$lookup`, `$addFields`) join customer profiles with loan applications. A deterministic scoring function computes risk, then the LLM generates a human-readable explanation of the decision — supporting compliance and audit requirements.

### Module 3: KYC Document Verification
Embeds document descriptions with Voyage AI and uses Atlas Vector Search to flag potential duplicates above a similarity threshold. Includes expiry checks and risk flags.

## Project Structure

```
├── app.py                     # Full FastAPI app (Easy track)
├── app_starter.py             # Skeleton app with TODOs (Hard track)
├── static/
│   └── index.html             # FSI web application (internet banking + employee portal)
├── data/
│   ├── faq_policies.json      # Bilingual FAQ/policy documents
│   ├── customers.json         # Synthetic customer profiles
│   ├── loan_applications.json # Loan application data
│   └── kyc_documents.json     # KYC document records
├── 01_load_faq_data.py        # Load & embed FAQ policies → faq_chunks
├── 02_create_indexes.py       # Create vector + text search indexes
├── 03_faq_chatbot.py          # Test the RAG chatbot (terminal)
├── 04_load_customers.py       # Load customer & loan data
├── 05_credit_scoring.py       # Test credit scoring pipeline (terminal)
├── 06_product_recommendation.py  # Test product recommendations
├── 07_load_kyc_data.py        # Load & embed KYC documents
├── 08_kyc_verification.py     # Test KYC verification (terminal)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/ycyeo-mongodb/MongoDB_FSI_GenAI_Workshop.git
cd MongoDB_FSI_GenAI_Workshop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your MongoDB URI, Voyage AI key, and LLM API URL

# 3. Load data and create indexes
python 01_load_faq_data.py
python 04_load_customers.py
python 07_load_kyc_data.py
python 02_create_indexes.py

# 4. Launch the application
uvicorn app:app --reload

# 5. Open http://localhost:8000
```

## Environment Variables

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `VOYAGE_API_KEY` | Voyage AI API key (via Atlas AI integrations) |
| `LLM_API_URL` | API Gateway endpoint for LLM generation (Bedrock/Claude) |

## Workshop Tracks

- **Easy** — Run pre-built scripts and use `app.py` (everything works out of the box)
- **Hard** — Implement the AI logic yourself in `app_starter.py` (4 TODOs to complete)
