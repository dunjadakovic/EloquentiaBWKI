# EloquentiaBWKI
## Generelle Informationen
Diese Repository enthält jegliche Codes hinsichtlich meiner drei KI-Module: ASR (Automatic Speech Recognition), TTS (Text-To-Speech) und RAG (Retrieval Augmented Generation). 
Alle (ausführbaren) Codes sind darauf ausgelegt, auf Google Colab ausgeführt zu werden. Falls dies nicht der Fall ist, müssen alle Requirements, die aufgelistet sind, 
manuell installiert werden und ggf. die root paths von "/content/" zu dem erwünschten Root Path geändert werden. 
## ASR (Automatic Speech Recognition)
Der ASR-Teil der Repository beinhaltet die Aufbereitung der Trainingsdaten auf das erwünschte Format (manifest_asr_aufbau.py), \n
den Code für das Finetuning auf Basis von Citrinet-256 (asrtraining.py), \n
die ausführbare Demo (asrdemo.py) \n
und Dateien, die aus dem genutzten Datensatz stammen und in der Demo als Testdateien genutzt werden (testfiles). 
Um das ASR-Modell auszuführen, muss die Demo gestartet werden (vorzugsweise in Google Colab). 
## RAG (Retrieval Augmented Generation)
Der RAG-Teil der Repository beinhaltet den Code für Kategorisierung der Wörter, der als Erstellung des Datensatzes fungiert (wordcategorisation.py)
und die tatsächliche RAG-Modell Integrierung (ragmitlangchain.py). 
Um Wörter wie im Datensatz kategorisieren zu können, führen Sie bitte wordcategorisation.py aus und geben Sie ihre Wortliste als String ein. 
Um die RAG-Applikation zu testen, führen Sie bitte ragmitlangchain.py aus und geben Sie das gewünschte Level und die gewünschte Kategorie ein. 
(voller Datensatz bei: https://raw.githubusercontent.com/dunjadakovic/RAGAPI/main/ContentAndCategories.csv). 
## TTS (Text-To-Speech)
Das TTS-Modell ist eine modifizierte Version von Bark von SunoAI (https://github.com/suno-ai/bark). 
Jegliche modifizierten Files sind hier aufzufinden, wobei in jeder File gekennzeichnet ist, was verändert wurde.
Um das TTS-Modell auszuführen, führen Sie bitte ttstest.py aus und geben Sie ihre prompt und den speaker (history_prompt) falls gewünscht ein. 
Um das originelle Modell im Vergleich auszuprobieren, befolgen Sie bitte die Anweisungen in https://github.com/suno-ai/bark. 
