# ✨ Chrisis Summarization ✨

Our python package `crisis_summary` uses a rank --> rerank --> summarization pipeline to summarize what happened in a given day in a specific event to create visibility on important information and provide answers and sources.

## Features 

### Summary Package

#### Key Variables
- **`eventNo`**: 
- **`days`**: 
- **`model`**: 
- **`rerank`**: 
- **`summarize`**: 
- **`save`**: 

#### Algorithm Steps
1. **Read in Dataset / Queries**: Store in separate dictionary
2. **Ranking**: Break each document into overlapping word sequences (k-shingles) to capture local structure.
3. **Reranking**: Apply multiple hash functions to each document’s shingles, keeping only the minimum hash per function to create a signature matrix where columns are documents and rows are hash functions.
4. **Summarization**: Split each signature into band(groups of rows) ➔ Within each band, the sequence of hash values is grouped and treated as a single entity ➔ If two documents have the same band hash they are considered to be in the same "bucket" for that band.

## Requirements
**These will be the technical requirements to run the code**
- python = "^3.11"
- pandas = "^2.2.3"
- requests = "^2.32.3"
- ir-datasets = {git = "https://github.com/allenai/ir_datasets.git", rev = "crisisfacts"}
- pyterrier-t5 = {git = "https://github.com/terrierteam/pyterrier_t5.git"}
- rerankers = "^0.6.0"
- openai = "^1.55.1"
- sphinx = "^8.1.3"
- myst-nb = "^1.1.2"
- autoapi = "^2.0.1"
- sphinx-autoapi = "^3.3.3"
- sphinx-rtd-theme = "^3.0.2"
- pytest = "^8.3.3"
- python-dotenv = "^1.0.1"

## Installation

**We describe the bash comments to run each file**

```bash
pip install crisis_summary
```

## Usage

Explore [Usage Explanation Notebook](./example.ipynb) for a more detailed explanation of the package

### How to run package in script or notebook
#### Running BM25 --> ColBERT --> GPT Summary

```{python}
from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta
from dotenv import load_dotenv
import os

load_dotenv('.env')
api = os.getenv("OPENAI_API_KEY")
os.environ['IR_DATASETS_HOME'] = './' # set to where crisisfacts repository is

eventsMeta = get_eventsMeta(eventNoList='001', days=2)

model = crisis(events = eventsMeta)

final_df, time_taken, memory_used = model.rank_rerank_colbert(model = 'BM25')
result_df = model.group_doc(final_df)
summary_df, time_summary, memory-summary = model.gpt_summary(result_df, api)
```


### How to run package from a terminal

Arguments:
- "-e", "--eventNo", - Required, "Event numbers to retrieve"
- "-d", "--days", Optional, "Number of days to retrieve"
- "-m", "--model", Optional, "Ranking model"
- "-r", "--rerank", Optional, "reranking model"
- "-u", "--summarize", Optional, "y/n for gpt summarization"
- "-s", "--save", Optional, "y/n to save output"

Example Terminal Code:
- python -m crisis_summary -e '001'
- python -m crisis_summary -e '001' -d 20 -m 'BM25' -r 'COLBERT'
- PYTHON -m crisis_summary -e '001' -d 20 -m 'BM25' -r 'COLBERT' -u 'y' -s 'y'

## Structure

Below is a brief overview of our file structure. We have also added a key to denote additional information about each file

🚫 - Part of .gitignore and not included in the repo 

🛠️ - Part of sphinx or package build, decription of file is not required 

🎨 - Images used in discussion.md, please see this file for further explanation

📚 - Please see sphinx documentation for a detailed description of this file

```
├── 📁 data (🚫)
├── 📁 docs 
│   ├── _build (🛠️)
│   ├── Makefile (🛠️)
│   ├── conf.py (🛠️)
│   ├── example.ipynb → demonstrates an example of usage of this package
│   ├── index.md (🛠️)
│   ├── make.bat (🛠️)
│   └── requirements.txt → required dependencies for this package
├── 📁 images
│   ├── LSH_Band_Row.png (🎨)
│   ├── LSH_Shingle_Length.png (🎨)
│   ├── LSH_number_trees.png (🎨)
│   ├── Viz1.png (🎨)
│   ├── Viz2.png (🎨)
│   ├── bloom_filter_results.png (🎨)
│   ├── hash_functions_FP_rate.png (🎨)
│   ├── lsh_improved_params.png (🎨)
│   └── lsh_s_curve.png (🎨)
├── 📁 notebooks
│   ├── EDA.ipynb → this notebook shows the EDA conducted on our text data
│   ├── exercise.ipynb → solves BloomFilter textbook problems
│   ├── lsh.ipynb → Used to test our implementation of LSH prior to the final version
│   ├── test.ipynb → Used to test our implementation of multi probe LSH prior to the final version
│   ├── visualization_lsh.ipynb → code for LSH graphs 
│   └── visualizations.ipynb → code for LSH graphs 
├── 📁 output → the files within this directory are the desired output for each data file for each near-deduplication algorithm
│   ├── hundredk-LSH_mp.txt
│   ├── hundredk-baseline.txt
│   ├── hundredk-lsh.txt
│   ├── onek-LSH.txt
│   ├── onek-LSH_mp.txt
│   ├── onek-baseline.txt
│   ├── tenk-LSH_mp.txt
│   ├── tenk-baseline.txt
│   ├── tenk-lsh.txt
│   ├── threehundred-LSH_mp.txt
│   ├── threehundred-baseline.txt
│   └── threehundred-lsh.txt
├── 📁 src
│   ├── 📁 deduplication
│   │   ├── LSH.py (📚)
│   │   ├── LSHForest.py (📚)
│   │   ├── LSHImproved.py (📚)
│   │   ├── __init__.py (📚)
│   │   ├── __main__.py (📚)
│   │   ├── bloom_filter.py (📚)
│   │   ├── dedup.py (📚)
│   └── 📁 utils
│       ├── use_cases.py (📚)
│       ├── utils.py (📚)
│       ├── visualization_lsh.py (📚)
│       └── visualizations.py (📚)
└── 📁 tests
│   ├── test_deduplication.py → test cases for each near-deduplication algorithm
├── discussion.md → describes algorithms in greater detail and answers project questions
├── poetry.lock (🛠️)
├── pyproject.toml (🛠️)
```


## Contributing

Clone and set up the repository with

```bash
git clone TODO && cd deduplication
pip install -e ".[dev]"
```

Install pre-commit hooks with

```bash
pre-commit install
```

Run tests using

```
pytest -v tests
```