# Natural Language Processing Course 2024/25: `Conversational Agent with Retrieval-Augmented Generation`

## Table of Contents
1. [About](#about)
2. [Team: Piškotki](#team-piškotki)
3. [Repository Structure](#repository-structure)
4. [Running the Project](#running-the-project)
5. [References](#references)

## About
This is <b>Project 1</b> which is part of the Natural Language Processing course for the academic year 2024/2025, where we aim to explore advanced techniques in conversational AI. Specifically, we will focus on Retrieval-Augmented Generation (RAG) to develop a chatbot capable of retrieving and integrating external information in real-time.

## Team: Piškotki
- Ana Poklukar
- Kristjan Sever
- Blaž Grilj

## Repository Structure  
- **`data/`**: Contains:
  - Datasets and documents
  - FAISS vector index files
  - `testing_questions.txt` - Evaluation queries

- **`report/`**: Project report and documentation

- **`src/`**: Source code:
  - `crawler.py` - Web crawling functionality
  - `main.py` - Command-line interface
  - `query.py` - Document retrieval system
  - `rag_model.py` - Core RAG implementation
  - `scraper.py` - Web scraping utilities

## Running the Project

### Prerequisites
- Python 3.8+
- pip package manager
- CUDA-enabled GPU (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-piskotki.git
   cd ul-fri-nlp-course-project-2024-2025-piskotki
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage Options

#### Single Question Mode
```bash
python src/main.py "Your question in quotes"
```

#### Batch Mode (Console Output)
```bash
python src/main.py questions.txt
```

#### Option 3: Batch Mode (File Output)
```bash
python src/main.py input_questions.txt output_answers.txt
```

## References
- [Gao, Y. et al. (2024) Retrieval-augmented generation for large language models: A survey](https://arxiv.org/abs/2312.10997)
- [Chen, J., Lin, H., Han, X., & Sun, L. (2024). Benchmarking Large Language Models in Retrieval-Augmented Generation](https://ojs.aaai.org/index.php/AAAI/article/view/29728)