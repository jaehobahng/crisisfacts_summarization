from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta
from rerankers import Reranker

import json
import os

# Create the 'auth' directory
os.makedirs('auth', exist_ok=True)

credentials = {
    "institution": "Georgetown University", # University, Company or Public Agency Name
    "contactname": "JaeHo Bahng", # Your Name
    "email": "jaheo127@gmail.com", # A contact email address
    "institutiontype": "Academic" # Either 'Research', 'Industry', or 'Public Sector'
}

# Write this to a file so it can be read when needed

with open('./auth/crisisfacts.json', 'w') as f:
    json.dump(credentials, f)


import os
os.environ['IR_DATASETS_HOME'] = './'

def test_columns():

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_colbert(model = 'BM25')

    final_df.shape[1]
    
    assert final_df.shape[1] == 20