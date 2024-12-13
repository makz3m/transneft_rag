from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import os
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from typing import Optional
from pinecone.grpc import PineconeGRPC
from RAG_prompts import *
import json, yaml
from Knowledge import Knowledge
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)

pc = PineconeGRPC(api_key=config["PineconeAPI"])

class Baseline():

  repo_id = config["reader_model_repo"]
  llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=config["LLAMA_CLOUD_API_KEY"],
)


  def get_contexts(self, chunk_size : int, knowledge : List[LangchainDocument]):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(config["tokenizer"]),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    contexts = []
    for doc in knowledge:
        contexts += splitter.split_documents([doc])

    unique_texts = {}
    contexts_unique = []
    for doc in contexts:
        if doc.page_content not in unique_texts:
          unique_texts[doc.page_content] = True
          contexts_unique.append(doc)

    return contexts_unique
  
  def load_embeddings(self, knowledge : List[LangchainDocument], chunk_size: int):
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["tokenizer"],
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
    )

    index_name = f"index_chunk_{chunk_size}_embeddings_{config['tokenizer'].replace('/', '-')}"
    
    index_folder_path = f"./data/indexes/{index_name}/"
    if os.path.isdir(index_folder_path):
        self.vectorstore = FAISS.load_local(
            index_folder_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore

    else:
        contexts = self.get_contexts(
            chunk_size,
            knowledge,
        )
        self.vectorstore = FAISS.from_documents(
            contexts, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        self.vectorstore.save_local(index_folder_path)
        return self.vectorstore
    
  def improve_query():
     pass

  def rag_pipeline(self,
    query: str,
    reranker: Optional[bool] = False,
    num_retrieved_docs: int = 50,
    num_docs_final: int = 3,
):
    self.load_embeddings(knowledge=Knowledge().get_documents(), chunk_size=config["chunk_size"])
    relevant_docs = self.vectorstore.similarity_search(query=query, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]

    if reranker:
      rerank_result = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=relevant_docs,
            top_n=25,
            return_documents=True
        )
      relevant_docs = [doc["document"]["text"] for doc in rerank_result.data]

    relevant_docs = relevant_docs[:num_docs_final]

    context = "\nИзвлеченные документы:\n"
    context += "".join([f"Документ {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_prompt.format(question=query, context=context)

    answer = self.llm.invoke(final_prompt)
    

    return answer, relevant_docs
  

      