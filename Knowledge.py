from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from pathlib import Path
import os
import yaml

with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)

os.environ["LLAMA_CLOUD_API_KEY"] = config["LLAMA_CLOUD_API_KEY"]

from llama_parse import LlamaParse

class Knowledge():
  def get_markdown(self, path : str):
    loader = UnstructuredMarkdownLoader(path)
    docs = loader.load()
    return docs
  
  def parse_docs(self, path : str, doc_name : str):
    docs = LlamaParse(result_type="markdown",
                  parsing_instruction="It's a scientific article. Don't parse References and contact information. Do NOT make summaries."
                 ).load_data(path)
    file_address = f"data\\parsed\\{doc_name}.md"
    with open(file_address, 'w', encoding="utf-8") as file:
      for doc in docs:
        file.write(doc.text)
    
  def get_file_name(self, path : str ):
    return '.'.join(str(path).split('\\')[-1].split('.')[:-1])

  def get_documents(self):
    raw_path = Path("data\\raw")
    parsed_docs = Path("data\\parsed").rglob("*.md")
    parsed_docs_names = [self.get_file_name(doc) for doc in parsed_docs]
    docs = []
    for path in raw_path.rglob("*"):
      doc_name = self.get_file_name(path)
      if doc_name not in parsed_docs_names:
        self.parse_docs(path, doc_name)
      docs += self.get_markdown(f"data\\parsed\\{doc_name}.md")
    return docs

