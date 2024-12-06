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
    
    assert final_df.shape[1] == 23
    assert final_df['Event'].unique() == '001'

def test_T5_columns():

    os.environ['IR_DATASETS_HOME'] = './'

    eventsMeta = get_eventsMeta(eventNoList='001', days=1)

    mine = crisis(events = eventsMeta)

    final_df, time_taken, memory_used = mine.rank_rerank_T5(model = 'BM25')

    final_df.shape[1]
    
<<<<<<< HEAD
    assert final_df.shape[1] == 20
    assert final_df['Event'].unique() == '001'

def test_group_doc():
    """
    Test the group_doc function for proper aggregation and formatting.
    """
    os.environ['IR_DATASETS_HOME'] = './'
    
    eventsMeta = get_eventsMeta(eventNoList='001', days=1)
    mine = crisis(events=eventsMeta)
    
    final_df, _, _ = mine.rank_rerank_colbert(model='BM25')
    grouped_df = mine.group_doc(final_df)
    
    # Assertions for grouped DataFrame
    assert not grouped_df.empty, "Grouped DataFrame should not be empty"
    assert 'texts' in grouped_df.columns, "'texts' column missing in grouped DataFrame"
    assert 'avg_importance' in grouped_df.columns, "'avg_importance' column missing in grouped DataFrame"
    assert grouped_df['avg_importance'].notnull().all(), "Importance column contains null values"
    assert 'docno_list' in grouped_df.columns, "'docno_list' column missing in grouped DataFrame"
=======
    assert final_df.shape[1] == 21
    assert final_df['Event'].unique() == '001'
>>>>>>> a5f6811 (#1 pytest change of column numbers)
