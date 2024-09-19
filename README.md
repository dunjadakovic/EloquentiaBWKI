# EloquentiaBWKI
## Generelle Informationen
Diese Repository enthält jegliche Codes hinsichtlich meiner drei KI-Module: 
<ol>
<li>ASR (Automatic Speech Recognition)</li>
<li>TTS (Text-To-Speech)</li>
<li> RAG (Retrieval Augmented Generation)</li>
</ol>
Alle (ausführbaren) Codes sind darauf ausgelegt, auf Google Colab ausgeführt zu werden. Falls dies nicht der Fall ist, müssen alle Requirements, die aufgelistet sind, manuell installiert werden und ggf. die root paths von "/content/" zu dem erwünschten Root Path geändert werden. 

## ASR (Automatic Speech Recognition)
Der ASR-Teil der Repository beinhaltet: 
<ol>
<li>die Aufbereitung der Trainingsdaten auf das erwünschte Format (manifest_asr_aufbau.py) </li>
<li>den Code für das Finetuning auf Basis von Citrinet-256 (asrtraining.py) </li>
<li>die ausführbare Demo (asrdemo.py) </li>
<li>Dateien, die aus dem genutzten Datensatz stammen und in der Demo als Testdateien genutzt werden (testfiles). </li>
</ol>
Um das ASR-Modell auszuführen, muss die Demo gestartet werden (vorzugsweise in Google Colab). <br/>

## RAG (Retrieval Augmented Generation)
<ol>
<li>Der RAG-Teil der Repository beinhaltet den Code für Kategorisierung der Wörter, der als Erstellung des Datensatzes fungiert (wordcategorisation.py) </li>
<li>Die tatsächliche RAG-Modell Integrierung (ragmitlangchain.py)</li>
</ol>
Um Wörter wie im Datensatz kategorisieren zu können, führen Sie bitte wordcategorisation.py aus und geben Sie ihre Wortliste als String ein. <br/>
Um die RAG-Applikation zu testen, führen Sie bitte ragmitlangchain.py aus und geben Sie das gewünschte Level und die gewünschte Kategorie ein. <br/>
(voller Datensatz bei: https://raw.githubusercontent.com/dunjadakovic/RAGAPI/main/ContentAndCategories.csv). <br/>


## TTS (Text-To-Speech)
Das TTS-Modell ist eine modifizierte Version von Bark von SunoAI (https://github.com/suno-ai/bark). <br/>
Jegliche modifizierten Files sind hier aufzufinden, wobei in jeder File gekennzeichnet ist, was verändert wurde.<br/>
Um das TTS-Modell auszuführen, führen Sie bitte ttstest.py aus und geben Sie ihre prompt und den speaker (history_prompt) falls gewünscht ein. <br/>
Um das originelle Modell im Vergleich auszuprobieren, befolgen Sie bitte die Anweisungen in https://github.com/suno-ai/bark. <br/>

## Weitere Informationen und Nutzungsmöglichkeiten 
Um eine vorläufige Demoversion der Applikation mit funktionierendem GUI auszuprobieren, besuchen Sie gerne die Website https://eloquentia.flutterflow.app. <br/>
Hier können Sie beide Tests und eine Aufgabendemo ausprobieren.<br/> 
In der Aufgabendemo dauert die Generierung der Aufgaben jedoch anfänglich sehr lange, da ein Cold Start der API durchgeführt werden muss. <br/>
Es kann des Weiteren sein, dass aufgrund dessen, wie die API konfiguriert ist, sie zeitweise aussetzt. 
