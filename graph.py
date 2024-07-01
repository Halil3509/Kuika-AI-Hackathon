from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    question : str
    generation : str
    documents : List[str]
    condition: str


from langgraph.graph import END, StateGraph
workflow = StateGraph(GraphState)
from graph_node_defining import *

mongo_db_indexes =  ['vector_index_employee', 'vector_index_izin', 'vector_index_servis']
collections = ['employee_list', 'izin_List', 'servis_list']

mongo_node_class = MongoNodeClass(db_indexes=mongo_db_indexes, collection_names=collections, db_name='kuikaAI', mongodb_router=mongodb_router)
postgre_node_class = PostgreNodeClass(postgresql_router)
print("Mango and PostgreSQL are ready to search.")


# *** Workflow Schema ***
workflow.add_node('Main_routing', main_routing)

# MongoDB
workflow.add_node('Mongo_Employee_Retrieve', mongo_node_class.retrieve_employee)
workflow.add_node('Mongo_izin_Retreieve', mongo_node_class.retrieve_izin)
workflow.add_node('Mongo_servis_Retreieve', mongo_node_class.retrieve_servis)

# PostgreSQL
workflow.add_node('Postgre_Siparis', postgre_node_class.retrieve_siparis)
workflow.add_node('Postgre_Uretim', postgre_node_class.retrieve_uretim)

# Chroma DB
workflow.add_node('Chroma_retrieve', chroma_retrieve)

# Routing
workflow.add_node('Mongo_routing', mongo_node_class.mongo_routing)
workflow.add_node('Postgre_routing', postgre_node_class.postgresql_routing)

workflow.add_node('Casual_generator', casual_generate)
workflow.add_node('Rag_generator', rag_generate)


workflow.add_conditional_edges(
    "Main_routing",
     main_condition,
    {
        "PostgreSQL": "Postgre_routing",
        "MongoDB": "Mongo_routing",
        "ChromaDB":"Chroma_retrieve",
        "Daily": "Casual_generator",
        "Boş": END
    }
)

workflow.add_conditional_edges(
    "Mongo_routing",
    mongo_node_class.mongo_condition,
    {
        "Çalışan": "Mongo_Employee_Retrieve",
        "İzin": "Mongo_izin_Retreieve",
        "Servis": "Mongo_servis_Retreieve",
        "Boş": END
    }
)

workflow.add_conditional_edges(
    "Postgre_routing",
    postgre_node_class.postgresql_condition,
    {
        "sipariş": "Postgre_Siparis",
        "üretim": "Postgre_Uretim",
        "Boş": END
    }
)


workflow.add_conditional_edges(
    "Rag_generator",
    rag_condition,
    {
        "end": END,
    }
)


workflow.add_edge('Chroma_retrieve', 'Rag_generator')
workflow.add_edge('Mongo_Employee_Retrieve', 'Rag_generator')
workflow.add_edge('Mongo_izin_Retreieve', 'Rag_generator')
workflow.add_edge('Mongo_servis_Retreieve', 'Rag_generator')
workflow.add_edge('Postgre_Siparis', 'Rag_generator')
workflow.add_edge('Postgre_Uretim', 'Rag_generator')

workflow.set_entry_point("Main_routing")

app = workflow.compile()

