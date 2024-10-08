!pip install openai
!pip install huggingface
!pip install sentence_transformers
!pip install xlsxwriter

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

sentences = """""" #zu kategorisierende wörter hier eingeben
#cleanup of data, falls nicht gebraucht rauskommentieren
splitSentences = sentences.split("\n")
i = 0
newList = []
while (i < len(splitSentences)):
  multPartsList = splitSentences[i].split(" ")
  splitSentences[i] = multPartsList[0]
  splitSentences[i] = splitSentences[i].replace("(UK", "")
  splitSentences[i] = splitSentences[i].replace("(US", "")
  if(len(splitSentences[i]) == 1):
    del splitSentences[i]
    i = i - 1
  i = i + 1
splitSentences.insert(0, 'a')
categoryList = []
currentCategory = []
i = 0
#sentence transformers von hf um die embeddings zu generieren
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(splitSentences)
i = 0
#liste von dot products
totalDotProdList = []
#liste der wörter von dot products
totalDotProdWordList = []
u = 0
while(i < len(splitSentences)):
  query_enc = model.encode(splitSentences[i])
  dotProdList = []
  dotProdWordList = []
  #dot products die für kategorie ausgewählt wurden
  closerDotProdList = []
  #passende wörter dazu
  closerDotProdWordList = []
  while(u < len(embeddings)):
    dot_prod = np.dot(query_enc, embeddings[u])
    dotProdList.append(dot_prod)
    dotProdWordList.append(splitSentences[u])
    u = u + 1
  n = 0
  while(n < len(dotProdList)):
    if(dotProdList[n] > 0.53): #0.53 als threshold für definitive einsortierung, empirisch ausgewählt
      closerDotProdList.append(dotProdList[n])
      closerDotProdWordList.append(dotProdWordList[n])
    n = n + 1
  n = 0
  if(len(closerDotProdList) < 2):
    while(n < len(dotProdList)):
      if(dotProdList[n] > 0.47): #falls nichts vorhanden, nächstkleinerer wert
        closerDotProdList.append(dotProdList[n])
        closerDotProdWordList.append(dotProdWordList[n])
      n = n + 1
  totalDotProdList.append(closerDotProdList)
  totalDotProdWordList.append(closerDotProdWordList)
  u = 0
  i = i + 1
#für nicht einsortierte wörter:
delSplitList = splitSentences
while(len(totalDotProdWordList) != 0):
  currentCategory = []
  if(len(totalDotProdWordList)!=0):
    currentCategory = totalDotProdWordList[0]
  newCategory = []
  w = 0
  while(w < len(currentCategory)):
    u = 0
    while(u < len(totalDotProdWordList)):
      currentDotProdList = totalDotProdWordList[u]
      if(currentDotProdList[0] == currentCategory[w]):
        newCategory = currentDotProdList
      u = u + 1
    o = 0
    while(o < len(newCategory)):
      currentWord = newCategory[o]
      #falls Wort in keiner Kategorie: durch embeddings errechnen, welcher kategorie am besten zuzuordnen
      if (currentWord not in currentCategory and currentWord in delSplitList):
        if(currentCategory[0] in splitSentences):
          indexOfCatWord = splitSentences.index(currentCategory[0])
          indexOfCurrWord = splitSentences.index(currentWord)
          query_enc_one = embeddings[indexOfCatWord]
          query_enc_two = embeddings[indexOfCurrWord]
        else:
          query_enc_one = model.encode(currentCategory[0])
          query_enc_two = model.encode(currentWord)
        if(np.dot(query_enc_one, query_enc_two)>= 0.4):
          currentCategory.append(currentWord)
        del delSplitList[delSplitList.index(currentWord)]
      o = o + 1
    if(currentCategory[w] in delSplitList):
      del delSplitList[delSplitList.index(currentCategory[w])]
    if not (len(totalDotProdWordList) == 0):
      del totalDotProdWordList[0]
    w = w + 1
  categoryList.append(currentCategory)
print(categoryList)
print(len(categoryList))

i = 0
#duplikate entfernen
lenList = []
wordList = []
while(i < len(categoryList)):
  categorySet = set(categoryList[i])
  categoryList[i] = list(categorySet)
  u = 0
  while(u < len(categoryList[i])):
    if not (categoryList[i][u] in wordList):
      wordList.append(categoryList[i][u])
    else:
      wordToDel = categoryList[i][u]
      del categoryList[i][u]
      if(len(categoryList[i]) == 1):
        word = categoryList[i][0]
        o = 0
        while(o < len(categoryList)):
          if(wordToDel in categoryList[o]):
            categoryList[o].append(word)
          o = o + 1
        del categoryList[i]
    u = u + 1
  if(len(categoryList[i]) == 0):
    del categoryList[i]
  i = i + 1

from openai import OpenAI
#falls API key vorhanden ist, funktioniert das Programm hier, sonst können  jegliche API calls auch manuell mit chatGPT ausgeführt werden
#hier werden die namen der kategorien definiert:
client = OpenAI(api_key = "") #api key hier einlesen
import xlsxwriter
workbook = xlsxwriter.Workbook('categoriesAndContent.xlsx')
worksheet  = workbook.add_worksheet()
worksheet.write(0, 0, "content")
worksheet.write(0, 1, "category")
i = 0
while(i < len(categoryList)):
  systemMessage = [{"role": "system", "content": "You are a clear and concise simple assistant that can only use one word to answer. Categorise the words given. Make the category as specific as you possibly can without using any words in the category. One word only. Specific. Only current category. No repetitions. Don't repeat any of your previous answers. Just the category you're currently categorising."}]
  userMessage = {"role": "user", "content": str(categoryList[i]).replace("[", " ").replace("]", " ")}
  systemMessage.append(userMessage)
  print(systemMessage)
  chat = client.chat.completions.create( model="gpt-3.5-turbo", messages=systemMessage)
  reply = chat.choices[0].message.content
  worksheet.write(i + 1, 0, str(categoryList[i]).replace("[", " ").replace("]", " "))
  worksheet.write(i + 1, 1, reply)
  print(reply)
  systemMessage = []
  userMessage = []
  i = i + 1
workbook.close()



