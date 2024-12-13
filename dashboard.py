from Evaluator import Evaluator
import glob
import glob
import pandas as pd
import json
import re

def dashboard():
  for file in glob.glob("./output/*.json"):

      df = pd.DataFrame(json.load(open(file, "r")))
      marks = df["eval_score"]
      n = 0
      summa = 0
      for mark in marks:
          mark = re.search(r'\b\d+\b', mark)
          if mark != None:
            if int(mark.group()) != 0 :
              n += 1
            summa += int(mark.group())
      print(file, "\tscore =", (summa/n)/5*100 , "%" )




if __name__ == "__main__":
  Evaluator().evaluate()    #раскоментируйте, если хотите запустить валидацию
  dashboard()