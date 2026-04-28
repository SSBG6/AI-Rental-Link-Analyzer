## 🚀 AI Rental Link Analyzer

- End-to-end rental deal analysis system from a single listing URL  
- Scrapes real estate webpages and extracts raw listing data  
- Uses **Qwen3-VL LLM (via LM Studio)** to convert unstructured text into structured features  
- Applies feature engineering (price per sqm, deposit ratio, value score)  
- Uses a **Random Forest Classifier** to predict deal quality (good / average / bad)  
- Provides confidence scoring using prediction probabilities  
- Stores high-quality benchmark deals in **ChromaDB (vector database)**  
- Retrieves similar listings to compare pricing within the same area  
- Performs market-based evaluation using historical best deals  
- Combines ML + vector search + LLM reasoning for hybrid intelligence  
- Uses LLM to generate **clear, human-readable final verdicts**  
- Ensures clean UX by hiding missing data and always giving confident outputs  
- Built with **FastAPI** using an MVC-style architecture (controllers, services, models)  
- Interactive frontend with animated loading states and smooth UX  
- Modular and extensible design for scaling to new regions or datasets  
- Focused on real-world usability, not just model accuracy  

---

### 🧠 Description

AI-powered rental deal analyzer that scrapes listing URLs, extracts structured data using the Qwen3-VL LLM via LM Studio, and evaluates deal quality with a Random Forest model. Uses ChromaDB for market comparison and generates clear, human-readable verdicts for smarter rental decisions.
