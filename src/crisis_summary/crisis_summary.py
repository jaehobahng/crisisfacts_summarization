import pyterrier as pt
# from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import os
import pandas as pd
import psutil
import time
from rerankers import Reranker
import openai

class crisis:
    """
    A class designed to process crisis-related events by retrieving, reranking, 
    and summarizing relevant documents.

    Attributes:
        eventsMeta (dict): A dictionary containing metadata for crisis events.
    """
    
    def __init__(self, events):
        """
        Initializes the crisis class with a dictionary of events.

        Args:
            events (dict): Dictionary containing metadata for events with the structure:
                {eventId: [{"dateString": str, "requestID": str}, ...]}
        """
        self.eventsMeta = events

    def rank_rerank_colbert(self, model='BM25'):
        """
        Retrieves and reranks documents for each event using BM25 and ColBERT.

        Args:
            model (str): The retrieval model to use, default is 'BM25'.

        Returns:
            tuple: A tuple containing:
                - final_df (pd.DataFrame): The final reranked DataFrame with added importance scores.
                - runtime (float): Total runtime of the operation in seconds.
                - memory_used (float): Memory used during the process in megabytes.
        """
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
                    queries = pyTerrierDataset.get_topics()
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
                        retriever_df['q_id'] = row['qid']
                        retriever_df['question'] = row['text']
        
                        if not retriever_df.empty:
                            # Rerank
                            result = ranker.rank(query=row['indicative_terms'], docs=retriever_df['text'], doc_ids=retriever_df['docid'])
                            
                            rereank_score = [i.score for i in result.results]
                            rerank_rank = [i.rank for i in result.results]
                            rerank_doc = [i.doc_id for i in result.results]
        
                            # Creating a DataFrame
                            df = pd.DataFrame({
                                'rerank_score': rereank_score,
                                'rerank_rank': rerank_rank,
                                'rerank_doc': rerank_doc
                            })
        
                            retriever_df = retriever_df.merge(df, left_on='docid', right_on='rerank_doc', how='left')
        
                            retriever_df = retriever_df[retriever_df['rerank_rank']<=5]
        
                            # Clean
                            result_df = retriever_df.sort_values('rerank_rank', ascending=True).reset_index(drop=True)
                            result_df = result_df.merge(dataset[['docno', 'unix_timestamp']], on='docno', how='left')
        
                            # Append to final_df
                            final_df = pd.concat([final_df, result_df], ignore_index=True)
                except:
                    continue

        final_df['formatted_datetime'] = pd.to_datetime(final_df['unix_timestamp'], unit='s')

        min_max = (
            final_df.groupby(['request_id'])
            .agg(
                min=('rerank_score', 'min'),
                max=('rerank_score', 'max')
            )
            .reset_index()
        )

        final_df = final_df.merge(min_max, on='request_id', how='left')
        final_df['importance'] = (final_df['rerank_score'] - final_df['min']) / (final_df['max'] - final_df['min'])

        # Calculate runtime and memory usage
        end_time = time.time()  # End time
        end_memory = process.memory_info().rss  # Memory usage at end (in bytes)
        runtime = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert bytes to MB

        return final_df, runtime, memory_used

    def group_doc(self, df):
        """
        Groups and aggregates document data for each query.

        Args:
            df (pd.DataFrame): DataFrame containing retrieved and reranked documents.

        Returns:
            pd.DataFrame: Aggregated DataFrame with combined texts, average importance, and other metadata.
        """
        result_df = (
            df.groupby(['request_id', 'q_id'])
            .agg(
                texts=('text', ' '.join),                     # Join all text values into a single string
                docno_list=('docno', list),                   # Collect docno values in a list
                avg_importance=('importance', 'mean'),        # Calculate the average importance
                unix_timestamp=('unix_timestamp', 'min'),
                question=('question', 'min'),
                query=('query', 'min')
            )
            .reset_index()                                    # Reset index for a clean DataFrame
        )
        return result_df

    def gpt_summary(self, df, api):
        """
        Summarizes the grouped documents using OpenAI's GPT model.

        Args:
            df (pd.DataFrame): DataFrame containing grouped documents to summarize.
            api (str): OpenAI API key for authentication.

        Returns:
            tuple: A tuple containing:
                - df (pd.DataFrame): DataFrame with added summaries.
                - runtime (float): Total runtime of the operation in seconds.
                - memory_used (float): Memory used during the process in megabytes.
        """
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
            If no answers can be found at all, return "unanswerable".

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

        df['summary'] = answer_output

        end_time = time.time()  # End time
        end_memory = process.memory_info().rss  # Memory usage at end (in bytes)

        # Calculate metrics
        runtime = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # Convert bytes to MB

        # Return results and performance metrics
        return df, runtime, memory_used
