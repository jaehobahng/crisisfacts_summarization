import os
import pymongo
import pandas as pd


# MongoDB connection
mongo_host = os.getenv("MONGO_HOST", "localhost")
mongo_port = int(os.getenv("MONGO_PORT", "27017"))
client = pymongo.MongoClient(f"mongodb://{mongo_host}:{mongo_port}")
db = client['data_db']


def add_csvs_from_folder_to_mongo(folder_path, database):
    """
    Adds CSV files from a folder to MongoDB collections.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    database (pymongo.database.Database): The MongoDB database instance.
    """
    # Get the list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    existing_collections = database.list_collection_names()

    for csv_file in csv_files:
        # Determine the collection name by removing the '.csv' extension
        collection_name = os.path.splitext(csv_file)[0]
        
        # Skip if the collection already exists
        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' already exists. Skipping {csv_file}.")
            continue

        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)

        # Convert the DataFrame to a list of dictionaries
        records = df.to_dict(orient='records')

        # Insert the records into the specified collection
        collection = database[collection_name]
        collection.insert_many(records)
        print(f"Inserted {len(records)} records into the collection '{collection_name}'.")

        # Fetch and print the first record in the collection
        first_record = collection.find_one()
        print(f"First record in collection '{collection_name}': {first_record}")


if __name__ == "__main__":
    # Define the folder containing CSV files
    data_folder = os.path.join(os.getcwd(), 'output')
    
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"Folder '{data_folder}' does not exist. Please create it and add CSV files.")
    else:
        add_csvs_from_folder_to_mongo(data_folder, db)
