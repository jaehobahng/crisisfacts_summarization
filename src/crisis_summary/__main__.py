from crisis_summary.summary import crisis
from utils.util import get_eventsMeta

from dotenv import load_dotenv
import os

# User input

# eventnumber '001,002'
# days : Any number. Number of days you want form the event

# ranking model : 'BM25'
# Reranking model : Colbert, T5. If statement to run either model not a parameter in the function

# I think it's a good idea to have a y/n input for whether the user wants to have a summary because the summary needs an api key.

# Save to csv y/n      This isn't a pre-made function maybe we can just put it as part of the main.py file

 

load_dotenv('../../.env')
os.environ['IR_DATASETS_HOME'] = '../../'
api = os.getenv("OPENAI_API_KEY")