from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta

from dotenv import load_dotenv
import os

import logging
import psutil
import sys
import argparse
import time

load_dotenv('.env')
os.environ['IR_DATASETS_HOME'] = './'
api = os.getenv("OPENAI_API_KEY")


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("crisis_log.txt"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to make corpus for assignment 2"
    )
    
    start_time = time.time()

    def log_memory_usage(message="Memory usage"):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"{message}: {mem_info.rss / (1024 * 1024):.2f} MB")


    parser.add_argument("-e", "--eventNo", required=True, help="Event numbers to retrieve")
    parser.add_argument("-d", "--days", required=False, help="Number of days to retrieve")
    parser.add_argument("-m", "--model", required=False, help="Ranking model")
    parser.add_argument("-r", "--rerank", required=False, help="reranking model")
    parser.add_argument("-u", "--summarize", required=False, help="y/n for gpt summarization")
    parser.add_argument("-s", "--save", required=False, help="y/n to save output")

    args = parser.parse_args()

    default_days = 1
    default_model = 'BM25'
    default_rerank = 'COLBERT'

    # Assign either the provided values or the default ones
    days = int(args.days) if args.days is not None else default_days
    model = args.model if args.model is not None else default_model
    rerank = args.rerank if args.rerank is not None else default_rerank


    eventsMeta = get_eventsMeta(eventNoList=args.eventNo, days=days)
    
    mine = crisis(events = eventsMeta)
    

    if rerank == 'COLBERT':
        logging.info(f"Using Ranking / Re-ranking method:{model}/{rerank}")
        rank_df, time_taken, memory_used = mine.rank_rerank_colbert(model = model)
        final_df = mine.group_doc(rank_df)
        if args.summarize == 'y':
            final_df, time_gpt, memory_gpt = mine.gpt_summary(final_df, api)
    elif rerank == 'T5':
        logging.info(f"Using Ranking / Re-ranking method:{model}/{rerank}")
        rank_df, time_taken, memory_used = mine.rank_rerank_T5(model = model)
        final_df = mine.group_doc(rank_df)
        if args.summarize == 'y':
            final_df, time_gpt, memory_gpt = mine.gpt_summary(final_df, api)
    else:
        print("Choose either 'COLBERT' or 'T5'")

    if args.summarize == 'y':
        summary = 'summary'
    else:
        summary = 'nosum'
    
    if args.save == 'y':
        final_df.to_csv(f"./output/{model}_{rerank}_{summary}.csv", encoding='utf-8')
    
    logging.info(f"Dataframe shape : {final_df.shape}")

    end_time_total = time.time()  # End timing the entire process
    logging.info("Total deduplication process took %.2f seconds.", end_time_total - start_time)

    log_memory_usage("Final memory usage")