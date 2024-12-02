from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta
from rerankers import Reranker
import ir_datasets

import json
import os


def test_colbert_columns():

    os.environ['IR_DATASETS_HOME'] = './'

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_colbert(model = 'BM25')

    final_df.shape[1]
    
    assert final_df.shape[1] == 22
    assert final_df['Event'].unique() == '001'

def test_T5_columns():

    os.environ['IR_DATASETS_HOME'] = './'

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_T5(model = 'BM25')

    final_df.shape[1]
    
    assert final_df.shape[1] == 20
    assert final_df['Event'].unique() == '001'