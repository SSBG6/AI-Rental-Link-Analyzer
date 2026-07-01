# AI Rental Link Analyzer

An AI-powered tool that analyzes rental listings from a URL and tells you whether it's actually a good deal.

Give it a listing link, and it will scrape the page, use a vision-language LLM to pull out structured details (price, size, location, amenities, etc.), compare it against similar listings in a local vector database, and score the deal with a trained Random Forest model — returning a clear, human-readable verdict instead of just raw numbers.

## How It Works

1. **Scrape** — The listing URL is fetched and parsed with `requests` + `BeautifulSoup`.
2. **Extract** — The scraped content (including images, via vision input) is passed to **Qwen3-VL**, run locally through **LM Studio**, which extracts structured rental data as JSON.
3. **Compare** — Extracted listings are embedded and stored in **ChromaDB**, allowing the current listing to be compared against similar rentals already seen.
4. **Score** — A **Random Forest Classifier** (scikit-learn), trained on rental data and serialized with **Joblib**, evaluates deal quality.
5. **Rank & Verdict** — Results are ranked and converted into a clear, plain-language verdict shown in the web UI.

## Tech Stack

| Layer            | Technology                                  |
|-------------------|-----------------------------------------------|
| Backend           | FastAPI (Python), Uvicorn                     |
| Frontend          | HTML, CSS, JavaScript (FastAPI/Jinja templates) |
| Web scraping      | Requests, BeautifulSoup                       |
| LLM               | Qwen3-VL via LM Studio (local inference)      |
| Data processing   | Pandas, NumPy                                 |
| Machine learning  | Scikit-learn (Random Forest Classifier)       |
| Model storage     | Joblib (`.pkl`)                               |
| Vector database   | ChromaDB (default embedding function)         |
| Architecture      | MVC — Controllers, Services, Models           |

## Project Structure

```
AI-Rental-Link-Analyzer/
├── controllers/       # Request handling / route logic
├── models/             # Data models and the trained Random Forest (.pkl) artifacts
├── services/            # Core logic: scraping, LLM extraction, scoring, ChromaDB access
├── static/               # CSS, JS, and other static frontend assets
├── templates/             # HTML templates rendered by FastAPI
├── config.py               # App/environment configuration
├── main.py                  # FastAPI application entry point
├── predict.py                 # Runs deal-quality prediction on a listing
├── rank.py                     # Ranks/compares listings using ChromaDB similarity
├── train_model.py               # Trains the Random Forest model on rental data
├── jsonconvert.py                 # Converts extracted/raw data into structured JSON
└── README.md
```

> `TODO`: Confirm exact responsibilities of `predict.py` vs. `rank.py` vs. `jsonconvert.py` if they differ from the above, and note the training data source used by `train_model.py`.

## Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) installed locally, with the **Qwen3-VL** model downloaded and its local server running
- pip (or a virtual environment tool of your choice)

## Installation

```bash
git clone https://github.com/SSBG6/AI-Rental-Link-Analyzer.git
cd AI-Rental-Link-Analyzer
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt   # TODO: add a requirements.txt if not already present
```

## Configuration

Set up any required values in `config.py` (or a `.env` file, if supported). At minimum you'll likely need:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1   # LM Studio's local OpenAI-compatible endpoint
CHROMADB_PATH=./chroma_data                    # Local ChromaDB storage path
MODEL_PATH=./models/rental_model.pkl           # Path to the trained Random Forest model
```

> `TODO`: Verify these variable names and defaults against `config.py`.

## Usage

1. Start LM Studio and load the **Qwen3-VL** model so its local server is running.
2. Launch the app:

   ```bash
   uvicorn main:app --reload
   ```

3. Open the app in your browser (default: `http://localhost:8000`).
4. Paste in a rental listing URL and submit it for analysis.
5. Review the extracted listing details, comparable market data, and the generated deal-quality verdict.

### Training the model

If you want to retrain the Random Forest model on your own dataset:

```bash
python train_model.py
```

> `TODO`: Document the expected format/location of training data for `train_model.py`.

## Roadmap / Ideas

- `TODO`: Support for additional rental listing sites
- `TODO`: Batch analysis of multiple listings at once
- `TODO`: Historical price tracking per listing

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change.

## License

`TODO`: No license is currently specified. Add a `LICENSE` file (e.g. MIT) if you intend this to be open source.
 
<img width="448" height="901" alt="image" src="https://github.com/user-attachments/assets/342a1c06-cb85-4907-aef9-274dd0729333" />
<img width="456" height="751" alt="image" src="https://github.com/user-attachments/assets/3294d4e0-1d33-49b9-beec-85a8525b6cb6" />
