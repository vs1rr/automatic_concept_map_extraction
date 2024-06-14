

### REBEL fine-tuning

Originally pretrained on **relation extraction** task.

**Core idea** = fine-tune the model on concept map triple extraction. Can the model learn with few samples?
**Potential cons:** vocabulary is usually smaller/constrained in these tasks, more open in concept maps.

* `Corpora`: Falke's dataset

* `extract_predicates.py`
Analyse predicates from concept maps (to see if any more frequent than others, types, etc).
Can be run on the following two folders:
    * `all_gs_multi`: all concept maps from multi-document summarization
    * `all_gs_single`: all concept maps from single-document summarization
Also save the following file containing the predicate labels:
    * `predicate_label.txt`

* `map_triple_sentence.py`
For single-document summarisation, attempting to extract original sentence from which the triple was extracted. Only for Biology dataset (only one that was usable).
Rule-based system: mapping only if the subject and object are in the sentence, and if all the verbs' lemmas in the predicate are also in the sentence. Save the following document:
    * `cm_biology.csv`

* `divide_train_eval_test.py`
Divide the context sentences with their triples (`cm_biology.csv`) into train/eval/test set. Output files:
    * `cm_biology_eval.csv`
    * `cm_biology_test.csv`
    * `cm_biology_train.csv`

* `extract_vocab.py`
Extract vocab from training data (necessary for fine-tuning rebel)


* `rebel_fine_tuning.py`
Fine-tune REBEL model. Save the model:
    * `finetuned_rebel.pth`


* `run_rebel.py`
Self-explanatory.

---


### Rule-based triple extraction

**Core idea** = using dependency tree to extract triples from text (linking NPs via dependency paths that contain verbs)
**Potential cons:** cumbersome, overfitting

* `rule_based.py`
Extract triples using a rule-based system
  