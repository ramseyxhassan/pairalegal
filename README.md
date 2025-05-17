# pAIralegal - AI-Powered Insurance Regulation Assistant

![pAIralegal Interface](images/Screenshot%202025-05-05%20101159.png)

pAIralegal is an advanced AI system designed to help insurance professionals efficiently navigate and understand complex insurance regulations across different states. By leveraging state-of-the-art natural language processing and vector search technologies, pAIralegal makes it easy to find accurate, state-specific regulatory information without manually searching through lengthy documents.

## System Architecture

![System Architecture](images/Screenshot%202025-05-05%20101654.png)

The pAIralegal system consists of several integrated components that work together to deliver accurate regulatory information:

### 1. SERFF Web Scraper
- Automates collection of regulatory filings from state-specific SERFF (System for Electronic Rates & Forms Filing) endpoints
- Utilizes parallel scraping techniques to simultaneously gather data from different insurance types across multiple states
- Organizes documents by categories such as insurance type, business name, and regulatory state

### 2. Document Processing
- Implements document chunking to split large insurance filings into smaller, manageable sections
- Uses embedding overlap to preserve critical context across document sections
- Applies importance scoring to prioritize sections containing policy numbers, coverage details, and regulations

### 3. Vector Database
- Converts insurance text into numerical format (embeddings) using SentenceTransformer models
- Stores embeddings in Qdrant, a high-performance vector database
- Utilizes cosine distance and HNSW (Hierarchical Navigable Small World) algorithm for efficient similarity searches

### 4. LLaMA-Powered Response Generation
- Matches user queries against stored document embeddings
- Retrieves relevant context using vector similarity search
- Generates detailed, contextually accurate responses using a fine-tuned LLaMA 3.2 model

## Setup Instructions

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)
- Tesseract OCR installed at `C:\Developer\Tools\Tesseract-OCR\tesseract.exe`
- Chrome browser and ChromeDriver
- Docker (for running Qdrant)

### Installation

1. Clone the repository
```bash
git clone https://github.com/ramseyxhassan/pairalegal
cd pairalegal
```

2. Install required Python packages
```bash
cd llama3.2
pip install -r requirements.txt
```

3. Start Qdrant vector database
```bash
cd ../Qdrant/Qdrant
docker-compose up -d
```

4. Setup NLTK data (if not already done)
```bash
python setup_nltk.py
```

5. Download LLaMA model
   - Download LLaMA 3.2 3B Instruct model and place it at `C:/Developer/Models/Llama-3.2-3B-Instruct`
   - Or update the model path in `chat.py` to your custom location

### Data Collection

1. Run the web scraper to collect insurance filings:
```bash
python webscraper.py
```

2. Process and embed documents:
```bash
python embeddings.py
```

3. Upload embeddings to Qdrant:
```bash
python qdrant_loader.py
```

## Usage

Start the conversational interface:
```bash
python chat.py
```

![Chat Interface](images/Screenshot%202025-05-05%20101722.png)

The system will:
1. Load the embedding model and LLaMA 3.2 model
2. Connect to the Qdrant database
3. Present a chat interface where you can ask insurance regulation questions
4. Retrieve relevant document sections and generate detailed responses

## Example Use Cases

- Finding state-specific coverage requirements
- Understanding policy exclusions and limitations
- Comparing regulatory differences between states
- Researching specific insurance company filings
- Identifying compliance requirements for new insurance products

## Technical Details

- **Embedding Model**: intfloat/e5-large-v2
- **LLM**: LLaMA 3.2 3B Instruct
- **Vector Database**: Qdrant
- **Web Scraping**: Selenium with ChromeDriver
- **Document Processing**: NLTK, PyTesseract, PDFPlumber

## Project Structure

- `webscraper.py` - Scrapes insurance filings from SERFF
- `embeddings.py` - Processes documents and generates embeddings
- `qdrant_loader.py` - Loads embeddings into Qdrant
- `chat.py` - Provides conversational interface for querying regulations
- `requirements.txt` - Required Python packages
- `Qdrant/` - Contains Qdrant database configuration and data