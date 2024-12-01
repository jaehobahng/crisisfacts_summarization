from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta
from rerankers import Reranker
import ir_datasets

import json
import os

os.environ['IR_DATASETS_HOME'] = './'

# Ensure the dataset directory exists
if not os.path.exists(os.environ['IR_DATASETS_HOME']):
    os.makedirs(os.environ['IR_DATASETS_HOME'])

def test_columns():

    os.environ['IR_DATASETS_HOME'] = './'

    os.makedirs('auth', exist_ok=True)

    credentials = {
        "institution": "Georgetown University", # University, Company or Public Agency Name
        "contactname": "JaeHo Bahng", # Your Name
        "email": "jaheo127@gmail.com", # A contact email address
        "institutiontype": "Academic" # Either 'Research', 'Industry', or 'Public Sector'
    }

    with open('./auth/crisisfacts.json', 'w') as f:
        json.dump(credentials, f)

    os.environ['IR_DATASETS_HOME'] = './'

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    for eventId,dailyInfo in eventsMeta.items():

        for thisDay in dailyInfo:
            
            requestID = thisDay["requestID"]
            ir_dataset_id = "crisisfacts/%s/%s" % (eventId, thisDay["dateString"])        
            
            dataset = ir_datasets.load(ir_dataset_id)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_colbert(model = 'BM25')

    final_df.shape[1]
    
    assert final_df.shape[1] == 20