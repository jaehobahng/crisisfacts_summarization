from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta
from rerankers import Reranker

import json
import os

os.environ['IR_DATASETS_HOME'] = './'


def test_columns():

    os.environ['IR_DATASETS_HOME'] = './'

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_colbert(model = 'BM25')

    final_df.shape[1]
    
    assert final_df.shape[1] == 20