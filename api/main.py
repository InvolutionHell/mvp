from fastapi import FastAPI, HTTPException

from typing import Any
from numpy.f2py.auxfuncs import throw_error

from pymilvus import MilvusClient
from frontmatter import Frontmatter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings

import os
from datetime import datetime

from tools.LeafSnowflakeTool import LeafSnowflakeBuilder
from env_settings.settrings import Settings
from log.logger import setup_logger

app = FastAPI()

settings = Settings()

logger = setup_logger()

embeddingModel = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)

# https://github.com/langchain-ai/langchain-milvus/blob/fa642cac6512b9b7e04a786d7e644b675ca90d92/libs/milvus/langchain_milvus/vectorstores/zilliz.py
vector_store = MilvusClient(
    uri = settings.milvus_uri,
    token = settings.milvus_token,
)

def load_markdown_file(markdown_path: str) -> list[dict[str, Any]]:
    logger.info(f"Loading markdown file {markdown_path}")
    IdGenerator = LeafSnowflakeBuilder(datacenter_id=1, worker_id=2)
    resList : list[dict[str, Any]] = []
    try:
        # {'attributes': {'title': 'Agent', 'description': '大语言模型智能体：CS294/194-196课程、ReAct、FireAct等', 'status': 'running'},
        # 'body': '本节聚合 LL'}

        for file_name in os.listdir(markdown_path):
            if file_name.endswith(".md") or file_name.endswith(".mdx"):
                mdpath = markdown_path + "/" + file_name

                fm = Frontmatter.read_file(mdpath)
                document = {
                    "doc_id":IdGenerator.get_id(),
                    "meta_data":{"source": mdpath},
                    "title": fm.get("attributes", {}).get("title", ""),
                    "doc_description": fm.get("attributes", {}).get("description", ""),
                    "tags":fm.get("attributes",{}).get("tags") or [] ,
                    "content": fm.get("body",""),
                }
                resList.append(document)
            else:
                throw_error("Markdown File does not found in root path")

    except FileNotFoundError as e:
        throw_error(f"{markdown_path} is not a markdown file error is {e}")

    finally:
        #  {'doc_id': 1176534079445803008, 'meta_data': {'source': './md_store/second.mdx'}, 'title': 'Agent',
        #   'doc_description': '这是一个测试二号文件',
        #   'tags': [],
        #   'content': '## e二号文件测试标题\n- 没想到吧 我是二号'}]
        logger.info(f"Loaded markdown file {len(resList)}")
        return resList

def markdown_chunk_spilt(list_markdown_data: list[dict[str, Any]]) -> list[dict[str, Any]]:

    resList : list[dict[str, Any]] = []
    # todo specific the spilt role
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    for doc in list_markdown_data:

        if doc.get("content") is not None and doc.get("title") is not None:
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_split = markdown_splitter.split_text(doc.get('content'))

            counter = 0
            for markdown in md_split:
                counter += 1
                if markdown.page_content is not None or doc.get('title') is not None:
                # {'doc_id': 1176519214748540928,
                # 'meta_data': {'source': './md_store/index.mdx'},
                # 'title': 'agent', 'doc_description': None,
                # 'tags': [],
                # 'content': 'page_content='本节聚合 LLM 智 \n---',
                # 'chunk_id': 1,
                # 'section_id': {'Header 2': 'OpenHands（原 OpenDevin）'}}
                    doc = {
                        "doc_id": doc.get("doc_id"),
                        "meta_data": doc.get("meta_data"),
                        "title": doc.get("title"),
                        "doc_description": doc.get("doc_description"),
                        "tags": doc.get("tags"),
                        "content": markdown.page_content,
                        "chunk_id": counter,
                        # todo some of the paragraph does not have the partition
                        "section_id": str(markdown.metadata), # 几级标题
                    }
                    resList.append(doc)
        else:
            throw_error("Markdown File title or content is None")
    return resList


def markdown_chunk_embedding(list_markdown_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ans:list[dict[str, Any]] = []
    try:
        embeddings = embeddingModel.embed_documents([doc["content"] for doc in list_markdown_data])

        ans =[{**doc, "vector": vector}
            for doc, vector in zip(list_markdown_data, embeddings)]

    except Exception as e:
        logger.error(f"Embedding error occurs: {e}, documents={list_markdown_data}")
        raise RuntimeError(f"Embedding error occurs: {e}") from e
    finally:
        logger.info(f"Embedding {len(ans)} documents")
        return ans

def markdown_store(documents: list) -> dict[str, Any]:
    ans:dict[str, Any] = {}
    try:
        for doc in documents:
            doc["created_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ans = vector_store.insert(collection_name="rag",data=documents)
    except Exception as e:
        logger.error(f"Storage error occurs: {e}, documents={documents}")
        raise RuntimeError(f"Storage error occurs: {e}") from e
    finally:
        logger.info(f"Stored document {ans.get("insert_count")} documents")
        return ans

def markdown_process(markdown_file_path: str = "./md_store"):
    try:
        markdownDocumentData = load_markdown_file(markdown_file_path)
        docList = markdown_chunk_spilt(markdownDocumentData)
        data  = markdown_chunk_embedding(docList)
        markdown_store(data)
    except Exception as e:
        logger.error(f"markdown_process failed: {e}")
        raise RuntimeError("markdown_process failed") from e

@app.get("/server/check")
async def root():
    # health check
    return {"code":200,"message": "Hello World"}

@app.get("/log/check")
async def root():
    # health check
    logger.info(f"Hello World {settings.milvus_uri}")
    return {"code":200,"message": "Hello World"}

@app.get("/markdown/process/{markdown_file_path}")
async def markdown_process(markdown_file_path: str):
    try:
        markdownDocumentData = load_markdown_file(markdown_file_path)
        docList = markdown_chunk_spilt(markdownDocumentData)
        data = markdown_chunk_embedding(docList)
        result = markdown_store(data)
        return {"code": 200, "message": "success", "data": result.get("insert_count")}
    except Exception as e:
        logger.error(f"markdown_process failed: {e}")
        raise HTTPException(status_code=500, detail=f"markdown_process failed: {e}")

@app.get("/search")
async def search():
    return {"code": 200, "message": "Hello World"}