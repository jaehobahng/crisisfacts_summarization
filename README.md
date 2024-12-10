# ✨ Crisis Summarization ✨

Our Python package, crisis_summary, implements a streamlined rank ➔ rerank ➔ summarization pipeline to distill key events from a specific day within a larger event. This process enhances visibility into critical information, delivering concise answers alongside the sources.

## Preparation
### Data
For the package to work seemlessly, you must downlad data from the following google drive and place the contents in the crisisfacts folder.
Data can be downloaded [here](https://drive.google.com/drive/folders/12rDWt6NVdTkMI6MXMiYFTLlFj9bgR6rX?usp=sharing). A subset of the data is in crisisfacts to help your understanding.

### API keys
Check .env.example file to create an .env file to hold OpenAI API keys. These keys will be used during summarization but will not be required for ranking and re-ranking.

### Dataset Credentials
Run the following code in example.ipynb to fill in credentials for running the pyterrier package.
```python
credentials = {
    "institution": "", # University, Company or Public Agency Name
    "contactname": "", # Your Name
    "email": "", # A contact email address
    "institutiontype": "" # Either 'Research', 'Industry', or 'Public Sector'
}

# Write this to a file so it can be read when needed
import json
import os

with open('./auth/crisisfacts.json', 'w') as f:
    json.dump(credentials, f)
```

## Features 

### Summary Package

#### Key Variables
- **`eventNo`**: Which event number to choose form the dataset `(001 ~ 018)`
- **`days`**: How many days to retrieve from each event
- **`model`**: Which ranking model to choose `["BM25", "TF_IDF", "PL2", "InL2", "DPH", "DirichletLM", "Hiemstra_LM", "DFRee"]`
- **`rerank`**: Which re-ranking models to choose `["COLBERT", "T5"]`
- **`summarize`**: Whether to utilize summarization model `['y','n']`
- **`save`**: Whether to save final output as a csv file `['y','n']`

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

```python
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
CRISIS_SUMMARY
│   .env
│   .gitignore
│   crisis_log.txt
│   discussion.qmd
│   Dockerfile
│   example.ipynb
│   LICENSE (🛠️)
│   poetry.lock (🛠️)
│   pyproject.toml (🛠️)
│   README.md
│   requirements.txt
│   upload.py
│   
├───📁.github
│   ├───PULL_REQUEST_TEMPLATE.md
│   │   
│   ├───📁ISSUE_TEMPLATE
│   │    ├───bug_report.md
│   │    ├───documentation.md
│   │    └───feature_request.md
│   │       
│   └───📁workflows
│        └───ci-cd.yml
│               
├───📁assets
│    ├───jaeho.ipynb
│       
├───📁auth (🚫)
│       crisisfacts.json
│       
├───📁crisisfacts (🚫)
│   │   001.csv
│   │   
│   └───📁001
│        └───2017-12-07
|                   
├───📁docs
│   │           
│   └───📁_build             
│       └───📁html
│           ├───changelog.html
│           ├───example.html
│           ├───genindex.html
│           ├───index.html
│           ├───objects.inv
│           ├───py-modindex.html
│           ├───search.html
│           └───searchindex.js
│               
├───📁output (🚫)
│    ├───original.csv
│    └───summary.csv
│       
├───📁src
│    └───📁crisis_summary
│        ├───summary.py (📚)
│        ├───__init__.py (📚)
│        ├───__main__.py (📚)
│        └───📁utils
│             └───util.py (📚)
└───📁tests
    └───test_crisis.py
```

## Contributing

Clone and set up the repository with

```bash
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