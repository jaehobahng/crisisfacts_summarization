from flask import Flask, jsonify, request
import pymongo
from flask_cors import CORS
import os

#Flask app
app  = Flask(__name__)
CORS(app)

# MongoDB connection details
mongo_host = os.getenv("MONGO_HOST", "localhost")
mongo_port = int(os.getenv("MONGO_PORT", "27017"))

# Connect to MongoDB
client = pymongo.MongoClient(f"mongodb://{mongo_host}:{mongo_port}")
db = client['data_db']
collection = db['summary']
original_collection = db['original']

count_summary = collection.count_documents({})
count_original = original_collection.count_documents({})

@app.route("/api/unique_requests", methods=["GET"])
def get_unique_requests():
    # Fetch all unique `request` values from the collection
    unique_requests = collection.distinct("request")
    return jsonify(sorted(unique_requests))


@app.route("/api/summary", methods=["GET"])
def get_data():
    request_filter = request.args.get("request", "All")  # Default to "All" if not provided

    query = {}
    if request_filter != "All":
        query["request"] = request_filter
    # Fetch data from MongoDB collection
    data = list(collection.find(query, {"_id": 0}))
    # data = list(collection.find({}, {"_id": 0})) 

    return jsonify(data)

@app.route("/api/original", methods=["GET"])
def get_original():
    request_filter = request.args.get("request", "All")  # Default to "All" if not provided

    query = {}
    if request_filter != "All":
        query["request"] = request_filter
    # Fetch data from the `original_data` collection
    data = list(original_collection.find(query, {"_id": 0}))  # Remove MongoDB `_id` field
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
