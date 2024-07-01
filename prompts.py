import pandas as pd

import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

# CHroma Libs
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI

import os
from tqdm import tqdm



# Postgre
import psycopg2

import numpy as np
import ast

os.environ['OPENAI_API_KEY'] = '<openAI_key>'

os.environ['MONGO_DB_KEY'] = "<MongoDB key>"
os.environ['MONGO_DB_CON_STRING'] = "<MongoDB connection string>"

env_vars = dotenv_values('<postgres.env_path>')


def get_gpt_llm():
    chat_params = {
        "model": "gpt-4o", # Bigger context window
        "openai_api_key": os.environ['OPENAI_API_KEY'],
        "temperature": 0.2, # To avoid pure copy-pasting from docs lookup
        "max_tokens": 4096
    }
    llm = ChatOpenAI(**chat_params)
    return llm

llm = get_gpt_llm()

## *** Main Router ***
main_prompt = PromptTemplate(
    template="""'Rol': sisitem, 'İçerik': Sen bir yönlendiricisin. Görevin gelen isteği uygun kategoriye yönlendirmek. 

    Eğer gelen mesaj kolonları ve ilgili verileri*['Order No', 'Model Kodu', 'Model Adı', 'Müşteri', 'Bölüm', 'Tedarikçi',
       'Müşteri Order No', 'Rpt', 'Marka', 'Renk Kodu', 'İşlem', 'RSN',
       'Tarih', 'Adet', 'Kumaş', 'Kumaş Tipi', 'Kalite Tipi'] olan şirketin H&M, ZARA, Bershaka gibi markalara olan
       üretim* verilerini gösteren bir mesaj ise veya

       kolonları *['Durum', 'Statü', 'Booking -> Order', 'Set Order', 'Set Parça Sayısı',
       'Order Tipi', 'Booking Type', 'Order Grubu', 'Satış Tipi',
       'Sertifikasyon', 'Order No', 'Order Geliş Tarihi',
       'Sisteme Giriş Tarihi', 'Satır ya da Sütun M.İsteme Tarihi', 'YIL',
       'AY', 'HAFTA', 'ÇEYREK', 'Müşteri Order No', 'RPT', 'Açıklama',
       'Model Kodu', 'Model Adı', 'Model Grup Kodu', 'Model Grup Adı',
       'Model Group Name', 'Sezon', 'Müşteri', 'Marka', 'Departman',
       'Tedarik Grubu', 'Ülke Kodu', 'Sipariş Adedi', 'Döviz Türü',
       'Döviz Birim Fiyat', 'Döviz Ciro', 'EUR Ciro', 'Etüt Makina Dk',
       'Toplam Etüt Makina Dk', 'Etüt Dk', 'Toplam Etüt Dk',
       'MA_WorkOrder_RecId', 'NetsisProjectCode', 'Booking Statüsü',
       'Order Tipi - Durum'] olan şirketin sipariş durumları hakkında* verileri gösteren bir mesaj ise bu mesaja yanıt olarak sadece 'PostgreSQL',

       \n\n


    Eğer gelen mesaj *kişilerin adı, soyad, departman, semt/oturduğu yer, işe giriş tarihi, izin hakkı, kullandığı izin sayısı, gibi bilgileri ve 
    servis plakası, şoför adı/kimin sürdüğü veya kullandığı, servisin konumu veya nereye gittiği* hakkındaki bilgilerden herhangi 
    biri içeriyorsa yanıt olarak sadece 'MongoDB',

    \n\n

    Eğer gelen mesaj *ERP prosedürü ile ilgili veya seyahat prosedürü ile ilgili veya temsili/misafir ve ağırlama prosedürü* ile ilgiliyse 
    yanıt olarak sadece 'ChromaDB',

    \n\n

    Eğer gelen mesaj yukarıdakilerin hiçbiriyle hiç bir şekilde alakalı değilse senin işinin Logos şirketi için çalışmak olduğunu ve bu tür
    sorulara cevap veremeyeceğini söyle, eğer gelen mesaj yukarıdakilerle biraz alakalı ama tam değil. Biraz daha detaylandırması için konuyla 
    ilgili sorular sor

    \n\n\n

    Eğer kişi sıradan bir sohbet şeklinde konuşursan yanıt olarak sadece 'Daily' dön

    'Rol': Kullanıcı, 'İÇerik': {question},

    \n\n\n

    'Rol': asistan, 'İçerik': 
    """,
    input_variables=["question"],
)
# start = time.time()
main_router = main_prompt | llm
# question = "Tasarım departmanında çalışn soyadı teveoğlu olna kişi kimdir"
# print(main_router.invoke({"question": question}).content)
# end = time.time()
# print(f"The time required to generate response by Router Chain in seconds:{end - start}")


## *** MongoDB Router ***
mongo_prompt = PromptTemplate(
    template="""'Rol': sistem, 'İçerik': Sen çalışan, izin ve servis yapılarını yönlendiren bir yönlendiricisin. 

    Eğer içerikte kişinin adı, soyadı, departmanı, semti ve işe giriş tarihi özelliklerinin herhangi biri kullanılırsa çıktı olarak sadece 'Çalışan',
    \n\n
    Eğer içerikte kişisel bilgilerin yanında, kişinin ne kadar izin hakkı olduğu, kişinin izin hakkı sayısı, kullanılan izin sayısı, kazanılan izin hakkı sayısı özelliklerinden herhangi
    biri varsa çıktı olarak sadece 'İzin',
    \n\n
    Eğer içerikte servislerle ilgili servis plakası, soförünün adı, servisin gideceği yer/konum özellikleri, servisler nereye gider, gibi olan servis ile ilgili sorulara çıktı olarak 'Servis'
    dön.
    \n\n


    \n\n\n

    'Rol': kullanıcı, 'İçerik': {question}

    \n\n\n

    'Rol': asistan, 'İçerik': 
    """,
    input_variables=["question"],
)
# start = time.time()
mongodb_router = mongo_prompt | llm

# question = "Tatil  ne kadar"
# print(mongodb_router.invoke({"question": question}).content)
# end = time.time()
# print(f"The time required to generate response by the retrieval grader in seconds:{end - start}")



# *** Postgre SQL  Prompt ***

postgre_prompt = PromptTemplate(
    template="""'Rol': sistem, 'İçerik': Sen bir yönlendiricisin. Görevin gelen isteği uygun kategoriye yönlendirmek. 

    Eğer gelen mesaj kolonları ve ilgili verileri *['Order No', 'Model Kodu', 'Model Adı', 'Müşteri', 'Bölüm', 'Tedarikçi',
       'Müşteri Order No', 'Rpt', 'Marka', 'Renk Kodu', 'İşlem', 'RSN',
       'Tarih', 'Adet', 'Kumaş', 'Kumaş Tipi', 'Kalite Tipi'] olan şirketin H&M, ZARA, Bershaka gibi markalara olan
       Üretim* verilerini gösteren bir mesaj ise sadece 'Üretim',
    \n\n
    Eğer gelen mesaj, kolonları  *['Durum', 'Statü', 'Booking -> Order', 'Set Order', 'Set Parça Sayısı',
       'Order Tipi', 'Booking Type', 'Order Grubu', 'Satış Tipi',
       'Sertifikasyon', 'Order No', 'Order Geliş Tarihi',
       'Sisteme Giriş Tarihi', 'Satır ya da Sütun M.İsteme Tarihi', 'YIL',
       'AY', 'HAFTA', 'ÇEYREK', 'Müşteri Order No', 'RPT', 'Açıklama',
       'Model Kodu', 'Model Adı', 'Model Grup Kodu', 'Model Grup Adı',
       'Model Group Name', 'Sezon', 'Müşteri', 'Marka', 'Departman',
       'Tedarik Grubu', 'Ülke Kodu', 'Sipariş Adedi', 'Döviz Türü',
       'Döviz Birim Fiyat', 'Döviz Ciro', 'EUR Ciro', 'Etüt Makina Dk',
       'Toplam Etüt Makina Dk', 'Etüt Dk', 'Toplam Etüt Dk',
       'MA_WorkOrder_RecId', 'NetsisProjectCode', 'Booking Statüsü',
       'Order Tipi - Durum'] olan ve şirketin sipariş durumları hakkında* verileri gösteren bir mesaj ise bu mesaja yanıt
       olarak sadece 'Sipariş' dön
    \n\n
    Eğer çok belirleyici bir istem gelmediyse konuya bağlı olarak detaylandırmasını kibarca iste

    \n\n\n

    'Rol': kullanıcı, 'İçerik': {question}

    \n\n\n

    'Rol': asistan, 'İçerik': 
    """,
    input_variables=["question"],
)
# start = time.time()
postgresql_router = postgre_prompt | llm

# question = "Ürün durumunu öğrenmek sitiyorum. siparişim nerede"
# print(postgresql_router.invoke({"question": question}).content)
# end = time.time()
# print(f"The time required to generate response by the retrieval grader in seconds:{end - start}")


## Continue

rag_generator_prompt = PromptTemplate(
    template="""'Rol': Sistem, 'İçerik': Sen Logos şirketine yardımcı olmayı ve iş süreçlerini kolaylaştıran bir sanal asistansın. 
    Aşağıda üç tırnaklar arasında verilen soruyu, <> işaretleri arasında verilen kaynağa göre yardımsever ve açıklayıcı bir biçimde açıkla. 

    \n\n\n

    'Rol': Kullanıcı, 'İçerik': 
    Soru: ```{question}```

    Kaynak: <{documents}>.

    \n\n\n

    'Rol': Asistan, 'İçerik':
    """,
    input_variables=["question", "documents"],
)

rag_generator = rag_generator_prompt | llm
# print(mongodb_router.invoke({"question": question}).content)


## Continue

casual_prompt = PromptTemplate(
    template="""'Rol': Sistem, 'İçerik': Sen kişiyle gündelik hayat ile ilgili muhabbet etmeyi amaçlayan bir sanal asistansın. 
    Kişiye karşı kibar davran ve üç tırnaklar arasında verilen soruyu yanıtla

    \n\n\n

    'Rol': Kullanıcı, 'İçerik': 
    Soru: ```{question}```

    \n\n\n

    'Rol': Asistan, 'İçerik':
    """,
    input_variables=["question"],
)

casual_generator = casual_prompt | llm

