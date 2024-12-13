from QA_Generator import QA_Generator
from Baseline import Baseline
from Knowledge import Knowledge
from typing import Optional
from tqdm import tqdm
from Evaluation_prompts import EVALUATION_prompt
import os, yaml, json
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_huggingface import HuggingFaceEndpoint



with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Ты честно оцениваешь языковые модели."),
        HumanMessagePromptTemplate.from_template(EVALUATION_prompt),
    ]
)





eval_chat_model = HuggingFaceEndpoint(
    repo_id=config["eval_chat_model"],
    huggingfacehub_api_token=config["LLAMA_CLOUD_API_KEY"],
)



class Valuator():
  settings_name = f"chunk_{config['chunk_size']}_embeddings_{config['tokenizer'].replace('/', '-')}_rerank_{config['rerank']}_reader-model_{config['reader_model_repo'].replace('/', '-')}"
  output_file = f"./output/rag_{settings_name}.json"
  def rag_test(self,
    rerank: Optional[bool] = False,
):
    eval_ds = QA_Generator().get_QA(10)
    rag = Baseline()

    try:
        with open(self.output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_ds):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        answer, relevant_docs = rag.rag_pipeline(question, reranker=rerank)
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }

        outputs.append(result)

        with open(self.output_file, "w") as f:
            json.dump(outputs, f)

  def evaluate_answers(self):
    answers = []
    if os.path.isfile(self.output_file):
        answers = json.load(open(self.output_file, "r"))

    for experiment in tqdm(answers):
      if f"eval_score_{config['eval_chat_model']}" in experiment:
            continue

      eval_prompt = evaluation_prompt_template.format_messages(
            question=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
      eval_result = eval_chat_model.invoke(eval_prompt)
      try:
        feedback, score =  [eval_result.split("<assistant>")[-1].split("[RESULT]")[i] for i in [0,1]]
        experiment[f"eval_score"] = score
        experiment[f"eval_feedback"] = feedback
      except:
        experiment[f"eval_score"] = '0'
        experiment[f"eval_feedback"] = ''
    with open(self.output_file, "w") as f:
      json.dump(answers, f)

  def evaluate(self):
       self.rag_test(config["rerank"])
       self.evaluate_answers()