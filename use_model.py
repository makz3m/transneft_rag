from Baseline import Baseline



'''простой скрипт для использования модели'''


if __name__ == "__main__":
  while True:
    query = input(">>>")
    rag = Baseline()
    ans, relevant_docs = rag.rag_pipeline(query,reranker=True)
    print(ans)

