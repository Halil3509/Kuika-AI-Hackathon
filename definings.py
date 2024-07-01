from pymongo import MongoClient
from openai import OpenAI


class AtlasClient():
    """
    Atlas (Mongo) class to run vector search algorithm.

    """

    def __init__(self, altas_uri, dbname):
        self.mongodb_client = MongoClient(altas_uri)
        self.database = self.mongodb_client[dbname]

    ## A quick way to test if we can connect to Atlas instance
    def ping(self):
        self.mongodb_client.admin.command('ping')

    def get_collection(self, collection_name):
        collection = self.database[collection_name]
        return collection

    def find(self, collection_name, filter={}, limit=10):
        collection = self.database[collection_name]
        items = list(collection.find(filter=filter, limit=limit))
        return items

    def vector_search(self, collection_name, index_name, attr_name, embedding_vector, limit=5):
        collection = self.database[collection_name]
        results = collection.aggregate([
            {
                '$vectorSearch': {
                    "index": index_name,
                    "path": attr_name,
                    "queryVector": embedding_vector,
                    "numCandidates": 50,
                    "limit": limit,
                }
            },
            ## We are extracting 'vectorSearchScore' here
            ## columns with 1 are included, columns with 0 are excluded
            {
                "$project": {
                    '_id': 1,
                    'Element_Description': 1,
                    "search_score": {"$meta": "vectorSearchScore"}
                }
            }
        ])
        return list(results)

    def close_connection(self):
        self.mongodb_client.close()


class OpenAIClient():
    """
    Open AI class that will be used for RAG

    """

    def __init__(self, api_key) -> None:
        self.client = OpenAI(
            api_key=api_key,  # defaults to os.environ.get("OPENAI_API_KEY")
        )
        # print ("OpenAI Client initialized!")

    def get_embedding(self, text: str, model="text-embedding-3-large") -> list[float]:
        text = text.replace("\n", " ")
        resp = self.client.embeddings.create(
            input=[text],
            model=model)

        return resp.data[0].embedding