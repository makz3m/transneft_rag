from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from Knowledge import Knowledge
from huggingface_hub import InferenceClient
from QA_prompts import *
import pandas as pd
from tqdm import tqdm
import yaml, random, json

with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)


class QA_Generator():
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
  )
  QA = []
  llm_client = InferenceClient(model=config["QAGenerator_model_repo"], token=config["LLAMA_CLOUD_API_KEY"])

  def get_contexts(self):
    docs = Knowledge().get_documents()
    contexts = []
    for doc in docs:
      contexts += self.text_splitter.split_documents([doc])
    return contexts
  
  def llm_response(self, prompt : str):
    response = self.llm_client.post(
        json={
            "inputs": prompt,
            "task": "text-generation",

        },)
    return json.loads(response.decode())[0]["generated_text"]
  
  def generator(self, n : int):
    self.QA = []
    for context in random.sample(self.get_contexts(), n):
        try:
          output_QA_couple = self.llm_response(GEN_prompt.format(context=context.page_content))
          question = output_QA_couple.split("Вопрос: ")[-1].split("Ответ: ")[0]
          answer = output_QA_couple.split("Ответ: ")[-1]
          self.QA.append(
                {
                    "context": context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": context.metadata["source"],
                }
            )
        except:
            continue
    return self.QA
  
  def critic(self):
    for qa in tqdm(self.QA):
      evaluations = {
        "groundedness": self.llm_response(GROUNDEDNESS_prompt.format(context=qa["context"], question=qa["question"])),
        "relevance": self.llm_response(RELEVANCE_prompt.format(question=qa["question"])),
        "standalone": self.llm_response(STANDALONE_prompt.format(question=qa["question"])),
      }
      try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            self.QA.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
      except Exception as e:
        continue
    return self.QA
  def make_QA(self, n : int):
    self.generator(n)
    self.critic()
    print("Итого QA:", len(self.QA))
    try:
      with open('data\QA.json', 'r', encoding='utf-8') as f:
        already_made = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      already_made = []
    already_made += self.QA
    with open('data\QA.json', 'w', encoding='utf-8') as f:
      json.dump(already_made, f, ensure_ascii=False, indent=4)

  def get_QA(self, n : int):
     with open("data\QA.json", 'r', encoding="utf-8") as f:
        return random.sample(json.load(f), n)
     
if __name__ == "__main__":
  QA_Generator().make_QA(10)