# python3
# Create Date: 2024-01-10
# Author: Scc_hy
# Func: 将其其他专业数据向量化(
#        文本分块(RecursiveCharacterTextSplitter) ->
#        向量化(sentence-transformer)
#           ) -> 保存到向量数据库 -> 向量数据库持久化到磁盘上
# ==============================================================================
import os
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openxlab.dataset import download


def openxlab_download():
    load_d = '/home/xlab-app-center/data'
    if not os.path.exists(load_d):
        os.system(f'mkdir -p {load_d}')
    
    out_f =  f'{load_d}/Scchy___LLM-Data/cookingBook.json'
    if not os.path.exists(out_f):
        download(dataset_repo='Scchy/LLM-Data', source_path='cookingBook.json', target_path=load_d)
    return out_f


def file2Chroma2local():
    # 2- 加载数据 /home/xlab-app-center/
    js_f = openxlab_download()
    docs = []
    docs.extend(
        JSONLoader(
            js_f,
            jq_schema='.[].content',
            text_content=False
        ).load()
    )

    print('>>>>>>>>> [ 2- 加载数据 ]( 完成 )')
    # 3- 构建向量数据库
    ## 3.1 文本分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    print('>>>>>>>>> [ 3-构建向量数据库 | 文本分块]( 完成 )')
    ## 3.2 向量化-embedding模型
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
    print('>>>>>>>>> [ 3-构建向量数据库 | 向量化-embedding]( 完成 )')
    ## 3.3 语料加载到指定路径下的向量数据库
    # 定义持久化路径
    persist_directory = '/home/xlab-app-center/data_base/vector_db/chroma'
    if not os.path.exists(persist_directory):
        os.system(f'mkdir -p {persist_directory}')
    
    ## 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()
    print('>>>>>>>>> [ 3-构建向量数据库 | 向量数据库持久化到磁盘上]( 完成 )')
    return persist_directory


