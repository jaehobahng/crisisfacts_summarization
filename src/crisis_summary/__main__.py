from crisis_summary.summary import crisis
from crisis_summary.utils.util import get_eventsMeta

from dotenv import load_dotenv
import os

import logging
import psutil
import sys
import argparse
import time
"""
Main execution script for processing crisis-related events. This script retrieves 
event metadata, processes and ranks documents using specified ranking and re-ranking models, 
and optionally summarizes and saves the results.

Command-line Arguments:
    -e, --eventNo: (Required) Event numbers to retrieve. Input should be a list of event numbers.
    -d, --days: (Optional) Number of days of data to retrieve. Defaults to 1 if not provided.
    -m, --model: (Optional) Ranking model to use. Defaults to 'BM25' if not provided.
    -r, --rerank: (Optional) Re-ranking model to use. Choose between 'COLBERT' or 'T5'. Defaults to 'COLBERT'.
    -u, --summarize: (Optional) 'y' or 'n' indicating whether to use GPT-based summarization. Defaults to 'n'.
    -s, --save: (Optional) 'y' or 'n' indicating whether to save the output to a CSV file. Defaults to 'n'.

Workflow:
    1. Parse command-line arguments and assign default values for optional parameters.
    2. Retrieve metadata for the specified event numbers and days using `get_eventsMeta`.
    3. Initialize the `crisis` object with retrieved event metadata.
    4. Based on the chosen re-ranking model ('COLBERT' or 'T5'):
        - Perform ranking and re-ranking.
        - Group the documents into meaningful clusters.
        - If summarization is requested, summarize the grouped documents using GPT.
    5. Save the final results to a CSV file if requested.
    6. Log memory usage and process timing details at various stages.
    7. Display logging information to both the console and a log file.

Logging:
    - Logs information about memory usage, processing steps, and errors.
    - Outputs are written to 'crisis_log.txt' and displayed on the console.

Output:
    - If the `--save` option is enabled, the processed results are saved in the `output/` directory 
        with a filename pattern `{model}_{rerank}_{summary}.csv`.

Note:
    - Ensure that the `.env` file contains a valid OpenAI API key for summarization.
    - This script depends on the `psutil`, `dotenv`, and `argparse` libraries for its functionality.
"""
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
    parser.add_argument("-r", "--rerank", required=False, help="Reranking model")
    parser.add_argument("-u", "--summarize", required=False, help="Summarization model")
    parser.add_argument("-s", "--save", required=False, help="y/n to save output")
    parser.add_argument("-t", "--eval", required=False, help="y/n to evaluation")

    args = parser.parse_args()

    default_days = 1
    default_model = 'BM25'
    default_rerank = 'COLBERT'
    default_summary = 'GPT'

    # Assign either the provided values or the default ones
    days = int(args.days) if args.days is not None else default_days
    model = args.model if args.model is not None else default_model
    rerank = args.rerank if args.rerank is not None else default_rerank
    sum = args.summarize if args.summarize is not None else default_summary

    eventsMeta = get_eventsMeta(eventNoList=args.eventNo, days=days)
    
    mine = crisis(events = eventsMeta)
    
    # Ranking and Re-ranking
    if rerank == 'COLBERT':
        logging.info(f"Using Ranking / Re-ranking method:{model}/{rerank}")
        rank_df, time_taken, memory_used = mine.rank_rerank_colbert(model = model)
        final_df = mine.group_doc(rank_df)

    elif rerank == 'T5':
        logging.info(f"Using Ranking / Re-ranking method:{model}/{rerank}")
        rank_df, time_taken, memory_used = mine.rank_rerank_T5(model = model)
        final_df = mine.group_doc(rank_df)

    else:
        print("Choose either 'COLBERT' or 'T5'")

    # Summarization
    if sum == 'GPT':
        final_df, time_gpt, memory_gpt = mine.gpt_summary(final_df, api)
    elif sum == 'bart':
        final_df, time_gpt, memory_gpt = mine.bart_summary(final_df)

    # Save
    if args.save == 'y':
        final_df.to_csv(f"./output/{model}_{rerank}_{summary}.csv", encoding='utf-8')
    
    logging.info(f"Dataframe shape : {final_df.shape}")

    end_time_total = time.time()  # End timing the entire process
    logging.info("Total deduplication process took %.2f seconds.", end_time_total - start_time)

    log_memory_usage("Final memory usage")