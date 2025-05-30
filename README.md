# Natural Language Processing Course 2024/25: `Conversational Agent with Retrieval-Augmented Generation`

## Table of Contents
1. [About](#about)
2. [Team: Piškotki](#team-piškotki)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Running the Project](#running-the-project)
6. [Output](#output)
7. [Archive](#archive)
8. [References](#references)



## About
This is <b>Project 1</b> which is part of the Natural Language Processing course for the academic year 2024/2025, where we aim to explore advanced techniques in conversational AI. Specifically, we will focus on Retrieval-Augmented Generation (RAG) to develop a chatbot capable of retrieving and integrating external information in real-time.



## Team: Piškotki
- Ana Poklukar
- Kristjan Sever
- Blaž Grilj



## System Architecture
Our RAG system consists of three main components:
 - Retriever: Dynamically fetches Wikipedia content related to the user's query
    * Tries to find relevant Wikipedia pages
    * Scrapes and preprocesses the page content
    * Splits content into overlapping chunks for better context preservation
 - Embedding Engine: Transforms text into vector representations
    * Uses SentenceTransformer models for semantic encoding
    * Builds a FAISS index for efficient similarity search
    * Ranks chunks by relevance to the query and chooses best matches
 - Generator: Produces human-readable answers from retrieved content
    * Uses the DeepSeek LLM model (7B parameters)
    * Follows a structured prompt template that includes the question along with additional relevant context from Wikipedia
    * Constrains generation to use only retrieved information with prompt



## Repository Structure
- **`archive/`**: Old baseline code  
- **`data/`**: Contains:
  - `testing_questions.txt` - Evaluation queries 
  - `chatgpt_answers.txt` - ChatGPT's with internet search answers to `testing_questions.txt`
  - `model_answers.txt` - (Created after running `main.py`)
- **`report/`**: Project report and documentation
- **`src/`**: Source code:
  - `crawler.py` - Web crawling functionality
  - `main.py` - Command-line interface
  - `rag_model.py` - Core RAG implementation
  - `rag_evaluator.py` - Evaluation metrics and comparison
  - `wiki_scraper.py` - Web scraping utilities



## Running the Project

### Prerequisites
- Python 3.8+
- pip package manager
- CUDA-enabled GPU, 8GB+ GPU memory

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-piskotki.git
   cd ul-fri-nlp-course-project-2024-2025-piskotki
   ```

2. (Optional: recommended for isolating dependencies) Create a virtual environment:
   ```bash
    python -m venv rag_env
    source rag_env/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
#### SLURM (Recommended)
```bash
singularity build --ignore-fakeroot-command containers/container-rag-piskotki.sif ul-fri-nlp-course-project-2024-2025-piskotki/Singularity.def
```
### Usage Options
#### Single Question Mode
To ask a single question, you pass a question string as an input argument to the `main.py` script.
```bash
python src/main.py "When is GTA VI coming out?"
```

#### Batch Mode (Console Output)
You can also run a bunch of questions at the same time, write each question on its own row and pass the file path to the `main.py`. This will output answers in terminal.
```bash
python src/main.py testing_questions.txt
```

#### Batch Mode (File Output)
If you want outputs to be instead written in a file, provide additional path argument to the output_file:
```bash
python src/main.py testing_questions.txt model_answers.txt
```

#### SLURM
To run RAG model, change the question/python script arguments in the `run_model.sh` (see [local options](#usage-options-local)).
Then run: 
```bash
sbatch run_model.sh
```

### Evaluate
#### Locally
Call the `rag_evaluator.py` with your questions, chatgpt (or other source of answers) and model answers as arguments:
```bash
./src/rag_evaluator.py --questions ./data/testing_questions.txt --chatgpt ./data/chatgpt_answers.txt --model ./data/model_answers.txt
```

#### SLURM
To run the evaluation, run:
```bash
sbatch eval_model.sh
```

(Make sure `chatgpt_answers.txt`, `testing_questions.txt` and `model_answers.txt` exist before running)

`chatgpt_answers.txt` is a comparison file with ground truth answers. This file can be created by either manually answering yourselves, or simply ask some LLM.


### How evaluation works
Data files (`chatgpt_answers.txt`, `testing_questions.txt` and `model_answers.txt`) contain one question or answer per row. Our program uses row index to determine which answer belongs to which question.

`testing_questions.txt`:
```txt
When is GTA VI coming out?
Which upcoming game has Lashana Lynch acting in it?
...
```

`model_answers.txt`:
```txt
Grand Theft Auto VI is scheduled for release on May 26, 2026, for PlayStation 5 and Xbox Series X/S.
Lashana Lynch stars in Directive 8020, a sci-fi horror game where she plays astronaut Brianna Young.
...
```

`rag_evaluator` compares semantic similarity between the baseline (ChatGPT's) and our model's answers using sentence embeddings and cosine similarity. It draws a distribution across all answers in a `semantic_score_distribution_<timestamp>.png` as well as outputs the results in `semantic_results_<timestamp>.csv`. Below you can see the header.

```
Question,ChatGPT Answer,Model Answer,Semantic Similarity
```

## Output
Running `main.py` creates a text file `model_answers.txt`:
```txt
Grand Theft Auto VI is scheduled for release on May 26, 2026, for PlayStation 5 and Xbox Series X/S.
Lashana Lynch stars in Directive 8020, a sci-fi horror game where she plays astronaut Brianna Young.
...
```

## Archive
Contains old project where we initially used web scraping (search engines like Google search and DuckDuckGo) to get to the Wikipedia pages, but they were sadly too unreliable and rate limited.

## References
- [Gao, Y. et al. (2024) Retrieval-augmented generation for large language models: A survey](https://arxiv.org/abs/2312.10997)
- [Chen, J., Lin, H., Han, X., & Sun, L. (2024). Benchmarking Large Language Models in Retrieval-Augmented Generation](https://ojs.aaai.org/index.php/AAAI/article/view/29728)