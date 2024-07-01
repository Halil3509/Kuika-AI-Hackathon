from typing import List
from langchain.schema import Document
import os
from definings import AtlasClient, OpenAIClient
from dotenv import dotenv_values
import psycopg2
import numpy as np
import ast
from prompts import main_router, mongodb_router, postgresql_router, rag_generator, casual_generator
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings

CHROMA_PATH = "C:\\Users\\halilibrahim.hatun\\Documents\\Kuika-AI-Hackathon\chroma_db"

# YOU MUST - Use same embedding function as before
embedding_function = OpenAIEmbeddings(model='text-embedding-3-large',
                                      api_key=os.environ['OPENAI_API_KEY'])

# Prepare the database
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)



env_vars = dotenv_values('<postgres.env_path>')

class MyConfig(object):
   pass


class PostgreNodeClass():
    def __init__(self, postgresql_router):

        # Define router
        self.postgresql_router = postgresql_router
        print(env_vars)
        self.POSTGRE_DATABASE_NAME = env_vars.get('DATABASE_NAME')
        self.POSTGRE_DATABASE_USER = env_vars.get('DATABASE_USER')
        self.POSTGRE_DATABASE_PASSWORD = env_vars.get('DATABASE_PASSWORD')
        self.POSTGRE_DATABASE_HOST = env_vars.get('DATABASE_HOST')
        self.OPENAI_API_KEY = env_vars.get('OPENAI_API_KEY')
        print(self.POSTGRE_DATABASE_NAME)

        self.openai_client = OpenAIClient(api_key=self.OPENAI_API_KEY)

        self.conn = None
        self.conn_db()

    def conn_db(self):
        """
        postgresql db connecting processes
        """

        self.conn = psycopg2.connect(
            dbname=self.POSTGRE_DATABASE_NAME,
            user=self.POSTGRE_DATABASE_USER,
            password=self.POSTGRE_DATABASE_PASSWORD,
            host=self.POSTGRE_DATABASE_HOST)

        if (self.conn):
            print("db connection successfull")
        else:
            print("db connection failed")

    def similarity_search(self, query_text, embeddings, texts, k=5):
        # Step 1: Retrieve top-k texts based on similarity to query_text
        print(query_text)
        query_embedding = np.array(self.openai_client.get_embedding(query_text))
        query_embedding = query_embedding.reshape(1, -1)  # Ensure the embedding is ;
        similarities = []

        print("PostgreSQL Retrieving..")

        for emb in embeddings:
            # Ensure the embedding is converted from string to numpy array
            emb = np.array(ast.literal_eval(emb)).reshape(1, -1)
            similarity = np.dot(query_embedding, emb.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(similarity[0][0])

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Retrieve top-k texts
        top_k_texts = [texts[idx] for idx in top_k_indices]

        return top_k_texts

    def do_vector_search(self, query: str, embeddings, texts, k: int = 5) -> None:
        """
        DYNAMIC VECTOR SEARCH
        """

        # Start the method
        query = query.lower().strip()
        print('query: ', query)

        retrieved_text = self.similarity_search(query, embeddings, texts)

        return retrieved_text

    def postgresql_routing(self, state):
        """
        Postgresql ROUTING
        """
        print("--- Postgresql Routing Process ---")

        question = state["question"]
        response_text = postgresql_router.invoke({"question": question}).content
        print("Postgresql Router => ", response_text)

        return {"question": question, "condition": response_text}

    def postgresql_condition(self, state):

        condition = state['condition']

        print("Postgre SQL condition: ", state)
        # Routing
        if "sipariş" in condition.lower():
            return "sipariş"

        elif "üretim" in condition.lower():
            return "üretim"

        else:
            return "Boş"

    def retrieve_siparis(self, state):
        """
        SIPARIS RETRIEVING
        """
        print("--- Retrieving Postre Sipariş ---")
        question = state['question']
        cur = self.conn.cursor()
        cur.execute(f"SELECT text, embedding FROM siparis_embeddings")
        results = cur.fetchall()

        print("Postgre Sipariş Result: ", results)

        texts = []
        embeddings = []
        for row in results:
            texts.append(row[0])
            embeddings.append(row[1])

        results = self.do_vector_search(query=question, embeddings=embeddings, texts=texts, k=200)

        return {'question': question, 'documents': results}

    def retrieve_uretim(self, state):
        """
        URETIM RETRIEVING
        """
        print("--- Retrieving Postre Uretim ---")
        question = state['question']
        cur = self.conn.cursor()
        cur.execute(f"SELECT text, embedding FROM uretim_embeddings")
        results = cur.fetchall()

        print("Postgre Üretim Result: ", results)

        texts = []
        embeddings = []
        for row in results:
            texts.append(row[0])
            embeddings.append(row[1])

        results = self.do_vector_search(query=question, embeddings=embeddings, texts=texts, k=200)

        return {'question': question, 'documents': results}


class MongoNodeClass():
    def __init__(self, db_indexes: List[str], collection_names: List[str], db_name: str, mongodb_router):
        self.db_indexes = db_indexes
        self.collection_names = collection_names

        # Define router
        self.mongodb_router = mongodb_router

        # my config
        self.myconfig = MyConfig()
        self.myconfig.ATLAS_URI = os.environ['MONGO_DB_CON_STRING']
        self.myconfig.OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        self.myconfig.DB_NAME = db_name  # 'kuikaAI'
        self.atlas_clients = []

        # Run build atlas clients
        self.build_Atlas_clients()
        self.openai_client = OpenAIClient(api_key=self.myconfig.OPENAI_API_KEY)
        print("OpenAI Client is ready.")

    def build_Atlas_clients(self):
        """
        Atlas client connecting processes
        """

        for db_index_name, collection_name in zip(self.db_indexes, self.collection_names):
            self.myconfig.INDEX_NAME = db_index_name
            self.myconfig.COLLECTION_NAME = collection_name

            atlas_client = AtlasClient(self.myconfig.ATLAS_URI, self.myconfig.DB_NAME)
            atlas_client.ping()

            print(f"{collection_name} atlas is ready :)")

            self.atlas_clients.append(atlas_client)

    def do_vector_search(self, query: str, collection_name_parameter: str, k: int = 50) -> None:
        """
        DYNAMIC VECTOR SEARCH
        """
        if collection_name_parameter == 'employee':
            index = 0
        elif collection_name_parameter == 'izin':
            index = 1
        elif collection_name_parameter == 'servis':
            index = 2
        else:
            raise ValueError(f"{collection_name_parameter} could not found.")

        # Start the method
        query = query.lower().strip()
        print('query: ', query)

        # Query Embedding
        embedding = self.openai_client.get_embedding(query)

        # Vector Search
        results = self.atlas_clients[index].vector_search(collection_name=self.collection_names[index],
                                                          index_name=self.db_indexes[index],
                                                          attr_name='Element_Description_embedding',
                                                          embedding_vector=embedding, limit=k)

        return results

    def mongo_routing(self, state):
        """
        MONGO ROUTING
        """
        print("--- Mongo Routing Process ---")

        question = state["question"]
        response_text = self.mongodb_router.invoke({"question": question}).content
        print("Mongo Router => ", response_text)

        return {"question": question, "condition": response_text}

    def mongo_condition(self, state):
        # Routing
        condition = state['condition']

        if "Çalışan" in condition:
            return "Çalışan"

        elif "İzin" in condition:
            return "İzin"

        elif "Servis" in condition:
            return "Servis"

        else:
            return "Boş"

    def retrieve_employee(self, state):
        """
        EMPLOYEE RETRIEVING
        """
        print("--- Employee Retrieving ---")
        question = state['question']
        results = self.do_vector_search(query=question, collection_name_parameter='employee')

        print("Employee Retrieving Result: ", results)
        return_text = []
        for idx, result in enumerate(results):
            return_text.append(f'{result["Element_Description"]}\n')

        return {"question": question, "documents": return_text}

    def retrieve_izin(self, state):
        """
        RETRIEVE IZIN
        """
        print("--- İzin Retrieving ---")

        question = state['question']
        results = self.do_vector_search(query=question, collection_name_parameter='izin')

        print("İzin Retrieving Result: ", results)
        return_text = []
        for idx, result in enumerate(results):
            return_text.append(f'{result["Element_Description"]}\n')

        return {"question": question, "documents": return_text}

    def retrieve_servis(self, state):
        """
        RETRIEVE SERVIS
        """
        print("--- Servis Retrieving ---")

        question = state['question']
        results = self.do_vector_search(query=question, collection_name_parameter='servis')

        print("Servis Retrieving Result: ", results)
        return_text = []
        for idx, result in enumerate(results):
            return_text.append(f'{result["Element_Description"]}\n')

        return {"question": question, "documents": return_text}


def main_routing(state):
    """
    MAIN ROUTING
    """
    print("--- Main Routing Process ---")
    question = state["question"]
    response_text = main_router.invoke({"question": question}).content
    print("Main Router => ", {'question': question, 'condition': response_text})

    return {'question': question, 'condition': response_text}


def main_condition(state):
    condition = state['condition']
    print(state['condition'])
    # Routing
    if "PostgreSQL" in condition:
        return "PostgreSQL"

    elif "MongoDB" in condition:
        return "MongoDB"

    elif "ChromaDB" in condition:
        return "ChromaDB"

    elif "Daily" in condition:
        return "Daily"

    else:
        return "Boş"


def rag_generate(state):
    """
    RAG GENERATING
    """

    print("--- RAG generating process ---")
    question = state['question']
    documents = state['documents']
    response_text = rag_generator.invoke({'question': question, 'documents': documents}).content
    print("RAG Generator result: ", response_text)

    return {"question": question, "documents": documents, "generation": response_text, 'condition': "end"}


def rag_condition(state):
    if state['condition'] == 'end':
        return "end"



def casual_generate(state):
    """
    CASUAL GENERATE
    """
    print("--- Casual generating process ---")
    question = state['question']

    response_text = casual_generator.invoke({'question': question}).content
    print("Casual Generator result: ", response_text)

    return {"question": question, "generation": response_text}

def chroma_retrieve(state):
    """
    CHROMA RAG GENERATING
    """
    print("--- CHROMA RAG GENERATING ---")
    question = state['question']

    results = chroma_db.similarity_search_with_relevance_scores(question, k=50)
    print("Chroma retrieved data:  ", [doc.page_content for doc, _score in results])

    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    return {'question': question, 'documents': [doc.page_content for doc, _score in results]}