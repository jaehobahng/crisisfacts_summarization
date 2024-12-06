import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import os
import pandas as pd
import psutil
import time
from rerankers import Reranker
import openai

# $env:PYTHONUTF8 = "1"

import pandas as pd

data = [
    ("001", "Lilac Wildfire 2017"),
    ("002", "Cranston Wildfire 2018"),
    ("003", "Holy Wildfire 2018"),
    ("004", "Hurricane Florence 2018"),
    ("005", "2018 Maryland Flood"),
    ("006", "Saddleridge Wildfire 2019"),
    ("007", "Hurricane Laura 2020"),
    ("008", "Hurricane Sally 2020"),
    ("009", "Beirut Explosion 2020"),
    ("010", "Houston Explosion 2020"),
    ("011", "Rutherford TN Floods 2020"),
    ("012", "TN Derecho 2020"),
    ("013", "Edenville Dam Fail 2020"),
    ("014", "Hurricane Dorian 2019"),
    ("015", "Kincade Wildfire 2019"),
    ("016", "Easter Tornado Outbreak 2020"),
    ("017", "Tornado Outbreak April 2020"),
    ("018", "Tornado Outbreak March 2020"),
]

# Create DataFrame
event_df = pd.DataFrame(data, columns=["ID", "EventName"])


class crisis:
    """
    A class to manage and analyze crisis events using ranking, reranking, and summarization techniques.

    Attributes:
        eventsMeta (dict): Metadata about events, including their IDs and daily information.
    """
    global event_df
    def __init__(self, events):
        """
        Initialize the crisis class with events metadata.

        Args:
            events (dict): Dictionary containing metadata for various events.
        """
        self.eventsMeta = events

    def rank_rerank_colbert(self, model = 'BM25'):
        """
        Rank and rerank crisis-related documents using ColBERT model.

        Args:
            model (str): Initial ranking model to use (default is 'BM25').

        Returns:
            tuple: Final DataFrame containing ranked and reranked documents, runtime (in seconds), and memory used (in MB).
        """
        os.environ['IR_DATASETS_HOME'] = './'
        process = psutil.Process(os.getpid())  # Get current process
        start_memory = process.memory_info().rss  # Memory usage at start (in bytes)
        start_time = time.time()  # Start time

        final_df = pd.DataFrame()
        ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type='colbert')

        for eventId, dailyInfo in self.eventsMeta.items():
            for thisDay in dailyInfo:
                try:
                    ir_dataset_id = "crisisfacts/%s/%s" % (eventId, thisDay["dateString"])
                    print(ir_dataset_id, " processing")  
        
                    pyTerrierDataset = pt.get_dataset(f'irds:{ir_dataset_id}')
                    # queries = pyTerrierDataset.get_topics()
                    queries = pd.read_csv(f'crisisfacts/{eventId}.csv')
                    dataset = pd.DataFrame(pyTerrierDataset.get_corpus_iter(), columns=['docno', 'text', 'unix_timestamp'])
        
                    indexer = pt.IterDictIndexer("None", type=pt.index.IndexingType(3), meta=["docno", "text"], meta_lengths=[0, 200])
                    index = indexer.index(pyTerrierDataset.get_corpus_iter())
                        
                    retriever = pt.terrier.Retriever(index, wmodel=model, metadata=["docno", "text"])
                    retriever.setControl("termpipelines", "Stopwords,PorterStemmer")
        
                    for _, row in queries.iterrows():

                        retriever_df = pd.DataFrame(retriever.search(row['indicative_terms']))
                        retriever_df = retriever_df[~retriever_df['text'].isnull()]
                        retriever_df = retriever_df[retriever_df['rank']<50]
                        retriever_df['docid'] = retriever_df['docid'].astype(int)
        
        
                        retriever_df['Event'] = eventId
                        retriever_df['request_id'] = thisDay['requestID']
                        retriever_df['date'] = thisDay['dateString']
                        retriever_df['q_id'] = row['query_id']
                        retriever_df['question'] = row['text']

                        retriever_df['event_title'] = row['event_title']
                        retriever_df['trecis_category_mapping'] = row['trecis_category_mapping']
                        retriever_df['source'] = retriever_df['docno'].str.extract(r'^(?:[^-]+-){2}([^-]+)')
        
                        if not retriever_df.empty:
                            # Rerank
                            result = ranker.rank(query=row['indicative_terms'], docs=retriever_df['text'], doc_ids=retriever_df['docid'])
                            
                            rerank_score = [i.score for i in result.results]
                            rerank_rank = [i.rank for i in result.results]
                            rerank_doc = [i.doc_id for i in result.results]
        
                            # Creating a DataFrame
                            df = pd.DataFrame({
                                'rerank_score': rerank_score,
                                'rerank_rank': rerank_rank,
                                'rerank_doc': rerank_doc
                            })
        
                            retriever_df = retriever_df.merge(df, left_on='docid', right_on='rerank_doc', how='left')
        
                            retriever_df = retriever_df[retriever_df['rerank_rank']<=5]
        
                            #Clean
                            result_df = retriever_df.sort_values('rerank_rank', ascending=True).reset_index(drop=True)
                            result_df = result_df.merge(dataset[['docno', 'unix_timestamp']], on='docno', how='left')
        
                            # Append to final_df
                            final_df = pd.concat([final_df, result_df], ignore_index=True)

                except:
                    continue

        final_df['datetime'] = pd.to_datetime(final_df['unix_timestamp'], unit='s')
        final_df['date'] = final_df['datetime'].apply(lambda x: x.date())
        # final_df = final_df.merge(event_df, left_on="Event", right_on="ID", how='left')

        min_max = (
            final_df.groupby(['request_id'])
            .agg(
                min=('rerank_score','min'),
                max=('rerank_score', 'max')
            )
            .reset_index()
        )

        final_df = final_df.merge(min_max, on='request_id', how='left')
        final_df['importance'] = (final_df['rerank_score'] - final_df['min']) / (final_df['max'] - final_df['min'])
        final_df = final_df.sort_values(by=['importance'], ascending=[False])

        # Calculate runtime and memory usage
        end_time = time.time()  # End time
        end_memory = process.memory_info().rss  # Memory usage at end (in bytes)
        runtime = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert bytes to MB

        return final_df, runtime, memory_used



    def rank_rerank_T5(self, model = 'BM25'):
        """
        Rank and rerank crisis-related documents using T5 reranking pipeline.

        Args:
            model (str): Initial ranking model to use (default is 'BM25').

        Returns:
            tuple: Final DataFrame containing ranked and reranked documents, runtime (in seconds), and memory used (in MB).
        """
        process = psutil.Process(os.getpid())  # Get current process
        start_memory = process.memory_info().rss  # Memory usage at start (in bytes)
        start_time = time.time()  # Start time

        final_df = pd.DataFrame()

        for eventId,dailyInfo in self.eventsMeta.items():

            for thisDay in dailyInfo:
                ir_dataset_id = "crisisfacts/%s/%s" % (eventId, thisDay["dateString"])
                print(ir_dataset_id, " processing")  

                pyTerrierDataset = pt.get_dataset(f'irds:{ir_dataset_id}')
                queries = pd.read_csv(f'crisisfacts/{eventId}.csv')
                dataset = pd.DataFrame(pyTerrierDataset.get_corpus_iter(), columns=['docno', 'text', 'unix_timestamp'])

                indexer = pt.IterDictIndexer("None", type=pt.index.IndexingType(3), meta=["docno", "text"], meta_lengths=[0, 200])
                indexer.setProperty("termpipelines","PorterStemmer,Stopwords")
                index = indexer.index(pyTerrierDataset.get_corpus_iter())

                retriever = pt.BatchRetrieve(index, wmodel=model, metadata=["docno", "text"])
                
                monoT5 = MonoT5ReRanker(verbose=False) # loads castorini/monot5-base-msmarco by default

                mono_pipeline = retriever % 50 >> pt.text.get_text(pyTerrierDataset, "text") >> monoT5 % 5

                for index, row in queries.iterrows():
                    # matching_index = int(queries[queries['indicative_terms'] == row['indicative_terms']].index[0])
                    # print(ir_dataset_id, "query num : ",matching_index)
                    retriever_df = pd.DataFrame(retriever.search(row['indicative_terms']))
                    if not retriever_df.empty:

                        temp_rank = retriever_df['rank']

                        # result_df = mono_pipeline.transform(retriever_df).sort_values('rank',ascending=True)
                        result_df = mono_pipeline.transform(retriever_df)
                        result_df = result_df.reset_index(drop=True)

                        result_df = result_df.merge(dataset[['docno', 'unix_timestamp']], on='docno', how='left')

                        # result_df['Event'] = thisDay['eventID']
                        result_df['Event'] = eventId
                        result_df['request_id'] = thisDay['requestID']
                        result_df['date'] = thisDay['dateString']
                        result_df['q_id'] = row['query_id']
                        result_df['question'] = row['text']

                        result_df['event_title'] = row['event_title']
                        result_df['trecis_category_mapping'] = row['trecis_category_mapping']
                        result_df['source'] = result_df['docno'].str.extract(r'^(?:[^-]+-){2}([^-]+)')


                        result_df = result_df.rename(columns={
                            'rank': 'rerank_rank'     # Rename rank to rerank_rank
                        })

                        result_df['rank'] = temp_rank

                        final_df = pd.concat([final_df, result_df], ignore_index=True)



        final_df['datetime'] = pd.to_datetime(final_df['unix_timestamp'], unit='s')
        final_df['date'] = final_df['datetime'].apply(lambda x: x.date())

        min_max = (
            final_df.groupby(['request_id'])
            .agg(
                min=('score','min'),                     # Join all text values into a single string
                max=('score', 'max')
            )
            .reset_index()                                    # Reset index for a clean DataFrame
        )

        final_df = final_df.merge(min_max, on='request_id', how='left')
        final_df['importance'] = (final_df['score'] - final_df['min']) / (final_df['max'] - final_df['min'])
        final_df = final_df.sort_values(by=['importance'], ascending=[False])

        # Calculate runtime and memory usage
        end_time = time.time()  # End time
        end_memory = process.memory_info().rss  # Memory usage at end (in bytes)
        runtime = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert bytes to MB

        return final_df, runtime, memory_used


    def group_doc(self, df):
        """
        Group documents by request ID, query ID, and event attributes.

        Args:
            df (pd.DataFrame): DataFrame containing ranked and reranked document data.

        Returns:
            pd.DataFrame: Aggregated DataFrame with grouped document text, metadata, and average importance score.
        """
        result_df = (
            df.groupby(['request_id', 'q_id', 'Event', 'event_title'])
            .agg(
                texts=('text', ' '.join),                     # Join all text values into a single string
                docno_list=('docno', list),                   # Collect docno values in a list
                avg_importance=('importance', 'mean'),        # Calculate the average importance
                unix_timestamp =('unix_timestamp', 'min'),
                question = ('question', 'min'),
                query = ('query','min')
            )
            .reset_index()                                    # Reset index for a clean DataFrame
        )

        return result_df


    def gpt_summary(self, df, api):
        """
        Generate summaries for grouped documents using GPT model.

        Args:
            df (pd.DataFrame): DataFrame containing grouped document data.
            api (str): OpenAI API key for accessing GPT models.

        Returns:
            tuple: DataFrame with generated summaries, runtime (in seconds), and memory used (in MB).
        """
        # Set your OpenAI API key
        openai.api_key = api

        process = psutil.Process(os.getpid())  # Get current process
        start_memory = process.memory_info().rss  # Memory usage at start (in bytes)
        start_time = time.time()  # Start time

        answer_output = []
        for i, row in df.iterrows():
            question = str(row['question'] + "?")
            provided_text = row['texts']

            prompt = f"""
            You are a helpful assistant. Answer the question based only on the text provided below. 
            If no answers can be found at all, return "unanswerable"

            Don't make the responses conversational.
            Expressions like hundreds of thousands can be answers to questions asking how many or how much.
            Do not line break the text and just give me the output.

            Text:
            {provided_text}

            Question:
            {question}
            """

            client = openai.OpenAI()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            answer = response.choices[0].message.content
            answer_output.append(answer)
            # Print progress every 10 loops
            # if (i + 1) % 50 == 0: m
            #     print(f"Processed {i + 1} rows")
        df_mod = df.copy()
        df_mod['summary'] = answer_output

        # Extract 'request' from 'request_id'
        df_mod['request'] = df_mod['request_id'].apply(lambda x: x.split('-r')[0])
        
        # Convert 'unix_timestamp' to datetime and date formats
        df_mod['datetime'] = df_mod['unix_timestamp'].apply(lambda x: pd.to_datetime(x, unit='s'))
        df_mod['date'] = df_mod['datetime'].apply(lambda x: x.date())
        
        # Subset the DataFrame
        # df = df[['request', 'date', 'datetime', 'question', 'summary_xsum_detail', 'avg_importance']]

        # Sort by avg_importance in descending order
        df_mod = df_mod.sort_values(by=['date','avg_importance'], ascending=[True,False])

        # Ensure consistent data types
        df_mod['request'] = df_mod['request'].astype(str)
        df_mod['date'] = df_mod['date'].astype(str)
        df_mod['datetime'] = df_mod['datetime'].astype(str)

        end_time = time.time()  # End time
        end_memory = process.memory_info().rss  # Memory usage at end (in bytes)

        # Calculate metrics
        runtime = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert bytes to MB

        # Return results and performance metrics
        return df_mod, runtime, memory_used