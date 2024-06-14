# -*- coding: utf-8 -*-
"""
Relation extractor
"""
from typing import Union, List

import spacy
import torch
from datasets import Dataset
from openai import OpenAI
from openie import StanfordOpenIE
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.fine_tune_rebel.run_rebel import extract_triples
from src.settings import API_KEY_GPT, nlp

client = OpenAI(api_key=API_KEY_GPT)


class RelationExtractor:
    """ Extracting relations from text """

    def __init__(self, spacy_model: str, options: List[str] = ["rebel", "dependency", "chat-gpt", "corenlp"],
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None, local_rm: Union[bool, None] = None):
        """ local_m: whether the model is locally stored or not """
        self.options_to_f = {
            "rebel": self.get_rebel_rel,
            "dependency": self.get_dependencymodel,
            "chat-gpt": self.get_chat_gpt,
            "corenlp": self.get_corenlp_rel,
        }
        self.options_p = list(self.options_to_f.keys())
        self.check_params(options=options, rebel_t=rebel_tokenizer,
                          rebel_m=rebel_model, local_rm=local_rm)
        self.params = {
            "options": options,
            "rebel": {
                "tokenizer": rebel_tokenizer,
                "model": rebel_model,
                "local": local_rm
            }
        }
        self.options = options

        if "rebel" in options:
            self.rebel = {
                "tokenizer": AutoTokenizer.from_pretrained(rebel_tokenizer),
                "model": self.get_rmodel(model=rebel_model, local_rm=local_rm),
                "gen_kwargs": {"max_length": 256, "length_penalty": 0,
                               "num_beams": 3, "num_return_sequences": 3, }
            }
        else:
            self.rebel = None

        self.nlp = spacy.load(spacy_model)

    def get_corenlp_rel(self, sentences: List[str],
                        entities: Union[List[Union[str, spacy.tokens.Span, spacy.tokens.Token]], None]):
        """ Extracting relations with corenlp"""
        props = {'openie.max_char_length': '-1'}
        with StanfordOpenIE(properties=props) as client:
            triples = client.annotate("\n".join(sentences))
        triples = [(x["subject"], x["relation"], x["object"]) for x in triples]
        if entities is not None:
            # Convert all entities to their string representations
            entity_strings = [self._to_string(entity) for entity in entities]
            triples = [(a, b, c) for a, b, c in triples if
                       any((x in a) or (x in b) for x in entity_strings)]
        return triples

    def _to_string(self, entity):
        """ Helper method to convert entities to string representations """
        if isinstance(entity, spacy.tokens.Span) or isinstance(entity, spacy.tokens.Token):
            return entity.text
        return str(entity)

    @staticmethod
    def get_rmodel(model: str, local_rm: bool):
        """ Load rebel (fine-tuned or not) model """
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        if not local_rm:  # Downloading from huggingface
            model = AutoModelForSeq2SeqLM.from_pretrained(model)
        else:
            model = torch.load(model)
        model.to(device)
        return model

    @staticmethod
    def get_dependencymodel(sentences: str, entities: Union[List[str], None]):
        triplets = []
        SENTENCES = [sent.strip() for sent in TEXT.split('\n') if sent.strip()]  # Split text into sentences
        for sentence in SENTENCES:
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass", "agent", "csubjpass",
                                  "csubj", "compound"] and token.head.pos_ in ["VERB", "AUX", "ROOT", "VB", "VBD",
                                                                               "VBG", "VBN", "VBZ"]:
                    subject = token.text
                    verb = token.head.text
                    subject_pos = token.pos_
                    # print(subject_pos)
                    obj = None
                    if any(entity in subject for entity in entities):
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"]:
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)

                                if subject_pos in ["NOUN", "PROPN"] and obj_pos in ["NOUN", "PROPN"]:
                                    triplets.append((subject, verb, obj))
                    else:
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"]:
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)
                                if subject_pos in ["NOUN", "PROPN", "ADP"] and obj_pos in ["NOUN", "PROPN"] and any(
                                        entity in obj for entity in entities):
                                    triplets.append((subject, verb, obj))
        return triplets

    def check_params(self, options, rebel_t, rebel_m, local_rm):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "rebel" in options:
            if any(not isinstance(x, y) for (x, y) in \
                   [(rebel_t, str), (rebel_m, str), (local_rm, bool)]):
                raise ValueError("To extract relations with REBEL, you need to specify: " + \
                                 "`rebel_tokenizer` as string, `rebel_model` as string, `local_rm` as bool")

    # def tokenize(self, text: str):
    #     """ Text > tensor """
    #     return self.rebel['tokenizer'](
    #         text, max_length=256, padding=True,
    #         truncation=True, return_tensors='pt')

    def predict(self, input_m):
        """ Text > predict > human-readable """
        for key in ["input_ids", "attention_mask"]:
            if len(input_m[key].shape) == 1:
                #  Reshaping, has a single sample
                input_m[key] = input_m[key].reshape(1, -1)

        output = self.rebel['model'].generate(
            input_m["input_ids"].to(self.rebel['model'].device),
            attention_mask=input_m["attention_mask"].to(self.rebel['model'].device),
            **self.rebel['gen_kwargs'], )

        decoded_preds = self.rebel['tokenizer'].batch_decode(output, skip_special_tokens=False)
        return decoded_preds

    def get_dataloader(self, sent_l: List[str], batch_size: int = 16):
        if not sent_l:
            return None
        sent_l = [x for x in sent_l if x]
        sent_l = [x for x in sent_l if len(x.split()) <= 256]

        dataset = Dataset.from_dict({"text": sent_l})
        dataset = dataset.map(
            lambda examples: self.rebel['tokenizer'](examples["text"], max_length=256, padding=True, truncation=True,
                                                     return_tensors='pt'), batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])
        return DataLoader(dataset, batch_size=batch_size)

    def get_chat_gpt(self, sentences: List[str],
                     entities: Union[List[Union[str, spacy.tokens.Span, spacy.tokens.Token]], None]):
        res = []
        for sent in sentences:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user",
                     "content": f"Extract triples from this sentence:\n{sent}\n\n. One triple per line, in the format a|b|c"}
                ],
                temperature=0
            )

            try:
                output = completion.choices[0].message['content'].strip().split("\n")
                output = [tuple(x.split('|')) for x in output]
                output = [x for x in output if len(x) == 3]
                res += output
            except Exception as e:
                print(e)
                raise ValueError("Something went wrong with the summary")

        if entities:
            entity_strings = [self._to_string(entity) for entity in entities]
            res = [(a, b, c) for a, b, c in res if any(
                (x.lower() in a.lower()) or (x.lower() in b.lower()) or (x.lower() in c.lower()) for x in
                entity_strings)]

        return res

    def _to_string(self, entity):
        """ Helper method to convert entities to string representations """
        if isinstance(entity, spacy.tokens.Span) or isinstance(entity, spacy.tokens.Token):
            return entity.text
        return str(entity)

    def get_rebel_rel(self, sentences: List[str], entities: Union[List[str], None]):
        """ Extracting relations with rebel """

        # input_m = self.tokenize(text=sentences)
        dataloader = self.get_dataloader(sent_l=sentences)
        if not dataloader:  # empty sentences
            return []
        output_m = []
        for batch in dataloader:
            try:
                output_m += self.predict(input_m=batch)
            except:
                pass

        unique_triples_set = set()  # Set to store unique triples
        unique_triples_set_2 = set()  # Set to store unique triples

        res = []

        if entities:
            entity_strings = [str(entity).lower() for entity in entities if
                              isinstance(entity, str)]  # Convert entities to lowercase strings
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if any(entity.lower() in triple_part.lower() for entity in entity_strings for triple_part in
                           triple):
                        if triple not in unique_triples_set:
                            res.append(triple)
                            unique_triples_set.add(triple)
        else:
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if triple not in unique_triples_set_2:
                        res.append(triple)
                        unique_triples_set_2.add(triple)

        return res

    @staticmethod
    def post_process_rebel(x):
        """ Clean rebel output"""
        res = extract_triples(x)
        return [(elt['head'], elt['type'], elt['tail']) for elt in res]

    def __call__(self, text: Union[str, List[str]], entities: Union[List[str], None] = None):
        """ Extract relations for input text """
        if isinstance(text, list):
            sentences = text
        elif isinstance(text, str):
            sentences = [sent.strip() for sent in text.split('\n') if sent.strip()]
        else:
            raise ValueError("Input text must be either a string or a list of strings.")

        res = {}
        for option in self.options:
            curr_res = self.options_to_f[option](sentences=sentences, entities=entities)
            curr_res = [x for x in curr_res if x[0].lower() != x[2].lower()]
            res[option] = list(set(curr_res))
        return res


if __name__ == '__main__':
    REL_EXTRACTOR = RelationExtractor(
        options=["corenlp"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./fine_tune_rebel/finetuned_rebel.pth",
        local_rm=True,
        spacy_model="en_core_web_lg",
    )
    TEXT = """
    Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum. Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem. Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. Nullam quis ante. Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus. Nullam accumsan lorem in dui. Cras ultricies mi eu turpis hendrerit fringilla. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia. Nam pretium turpis et arcu. Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. Sed aliquam ultrices mauris. Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris. Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc. Nunc nonummy metus. Vestibulum volutpat pretium libero. Cras id dui. Aenean ut eros et nisl sagittis vestibulum. Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede. Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. Etiam imperdiet imperdiet orci. Nunc nec neque. Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi. Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo. Maecenas malesuada. Praesent congue erat at massa. Sed cursus turpis vitae tortor. Donec posuere vulputate arcu. Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci. Phasellus consectetuer vestibulum elit. Aenean tellus metus, bibendum sed, posuere ac, mattis non, nunc. Vestibulum fringilla pede sit amet augue. In turpis. Pellentesque posuere. Praesent turpis. Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc, eu sollicitudin urna dolor sagittis lacus. Donec elit libero, sodales nec, volutpat a, suscipit non, turpis. Nullam sagittis. Suspendisse pulvinar, augue ac venenatis condimentum, sem libero volutpat nibh, nec pellentesque velit pede quis nunc. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Fusce id purus. Ut varius tincidunt libero. Phasellus dolor. Maecenas vestibulum mollis diam. Pellentesque ut neque. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. In dui magna, posuere eget, vestibulum et, tempor auctor, justo. In ac felis quis tortor malesuada pretium. Pellentesque auctor neque nec urna. Proin sapien ipsum, porta a, auctor quis, euismod ut, mi. Aenean viverra rhoncus pede. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Ut non enim eleifend felis pretium feugiat. Vivamus quis mi. Phasellus a est. Phasellus magna. In hac habitasse platea dictumst. Curabitur at lacus ac velit ornare lobortis. Curabitur a felis in nunc fringilla tristique. Morbi mattis ullamcorper velit. Phasellus gravida semper nisi. Nullam vel sem. Pellentesque libero tortor, tincidunt et, tincidunt eget, semper nec, quam. Sed hendrerit. Morbi ac felis. Nunc egestas, augue at pellentesque laoreet, felis eros vehicula leo, at malesuada velit leo quis pede. Donec interdum, metus et hendrerit aliquet, dolor diam sagittis ligula, eget egestas libero turpis vel mi. Nunc nulla. Fusce risus nisl, viverra et, tempor et, pretium in, sapien. Donec venenatis vulputate lorem. Morbi nec metus. Phasellus blandit leo ut odio. Maecenas ullamcorper, dui et placerat feugiat, eros pede varius nisi, condimentum viverra felis nunc et lorem. Sed magna purus, fermentum eu, tincidunt eu, varius ut, felis. In auctor lobortis lacus. Quisque libero metus, condimentum nec, tempor a, commodo mollis, magna. Vestibulum ullamcorper mauris at ligula. Fusce fermentum. Nullam cursus lacinia erat. Praesent blandit laoreet nibh. Fusce convallis metus id felis luctus adipiscing. Pellentesque egestas, neque sit amet convallis pulvinar, justo nulla eleifend augue, ac auctor orci leo non est. Quisque id mi. Ut tincidunt tincidunt erat. Etiam feugiat lorem non metus. Vestibulum dapibus nunc ac augue. Curabitur vestibulum aliquam leo. Praesent egestas neque eu enim. In hac habitasse platea dictumst. Fusce a quam. Etiam ut purus mattis mauris sodales aliquam. Curabitur nisi. Quisque malesuada placerat nisl. Nam ipsum risus, rutrum vitae, vestibulum eu, molestie vel, lacus. Sed augue ipsum, egestas nec, vestibulum et, malesuada adipiscing, dui. Vestibulum facilisis, purus nec pulvinar iaculis, ligula mi congue nunc, vitae euismod ligula urna in dolor. Mauris sollicitudin fermentum libero. Praesent nonummy mi in odio. Nunc interdum lacus sit amet orci. Vestibulum rutrum, mi nec elementum vehicula, eros quam gravida nisl, id fringilla neque ante vel mi. Morbi mollis tellus ac sapien. Phasellus volutpat, metus eget egestas mollis, lacus lacus blandit dui, id egestas quam mauris ut lacus. Fusce vel dui. Sed in libero ut nibh placerat accumsan. Proin faucibus arcu quis ante. In consectetuer turpis ut velit. Nulla sit amet est. Praesent metus tellus, elementum eu, semper a, adipiscing nec, purus. Cras risus ipsum, faucibus ut, ullamcorper id, varius ac, leo. Suspendisse feugiat. Suspendisse enim turpis, dictum sed, iaculis a, condimentum nec, nisi. Praesent nec nisl a purus blandit viverra. Praesent ac massa at ligula laoreet iaculis. Nulla neque dolor, sagittis eget, iaculis quis, molestie non, velit. Mauris turpis nunc, blandit et, volutpat molestie, porta ut, ligula. Fusce pharetra convallis urna. Quisque ut nisi. Donec mi odio, faucibus at, scelerisque quis, convallis in, nisi. Suspendisse non nisl sit amet velit hendrerit rutrum. Ut leo. Ut a nisl id ante tempus hendrerit. Proin pretium, leo ac pellentesque mollis, felis nunc ultrices eros, sed gravida augue augue mollis justo. Suspendisse eu ligula. Nulla facilisi. Donec id justo. Praesent porttitor, nulla vitae posuere iaculis, arcu nisl dignissim dolor, a pretium mi sem ut ipsum. Curabitur suscipit suscipit tellus. Praesent vestibulum dapibus nibh. Etiam iaculis nunc ac metus. Ut id nisl quis enim dignissim sagittis. Etiam sollicitudin, ipsum eu pulvinar rutrum, tellus ipsum laoreet sapien, quis venenatis ante odio sit amet eros. Proin magna. Duis vel nibh at velit scelerisque suscipit. Curabitur turpis. Vestibulum suscipit nulla quis orci. Fusce ac felis sit amet ligula pharetra condimentum. Maecenas egestas arcu quis ligula mattis placerat. Duis lobortis massa imperdiet quam. Suspendisse potenti. Pellentesque commodo eros a enim. Vestibulum turpis sem, aliquet eget, lobortis pellentesque, rutrum eu, nisl. Sed libero. Aliquam erat volutpat. Etiam vitae tortor. Morbi vestibulum volutpat enim. Aliquam eu nunc. Nunc sed turpis. Sed mollis, eros et ultrices tempus, mauris ipsum aliquam libero, non adipiscing dolor urna a orci. Nulla porta dolor. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos. Pellentesque dapibus hendrerit tortor. Praesent egestas tristique nibh. Sed a libero. Cras varius. Donec vitae orci sed dolor rutrum auctor. Fusce egestas elit eget lorem. Suspendisse nisl elit, rhoncus eget, elementum ac, condimentum eget, diam. Nam at tortor in tellus interdum sagittis. Aliquam lobortis. Donec orci lectus, aliquam ut, faucibus non, euismod id, nulla. Curabitur blandit mollis lacus. Nam adipiscing. Vestibulum eu odio. Vivamus laoreet. Nullam tincidunt adipiscing enim. Phasellus tempus. Proin viverra, ligula sit amet ultrices semper, ligula arcu tristique sapien, a accumsan nisi mauris ac eros. Fusce neque. Suspendisse faucibus, nunc et pellentesque egestas, lacus ante convallis tellus, vitae iaculis lacus elit id tortor. Vivamus aliquet elit ac nisl. Fusce fermentum odio nec arcu. Vivamus euismod mauris. In ut quam vitae odio lacinia tincidunt. Praesent ut ligula non mi varius sagittis. Cras sagittis. Praesent ac sem eget est egestas volutpat. Vivamus consectetuer hendrerit lacus. Cras non dolor. Vivamus in erat ut urna cursus vestibulum. Fusce commodo aliquam arcu. Nam commodo suscipit quam. Quisque id odio. Praesent venenatis metus at tortor pulvinar varius. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum. Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem. Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. Nullam quis ante. Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc, quis gravida magna mi a libero. Fusce vulputate eleifend sapien. Vestibulum purus quam, scelerisque ut, mollis sed, nonummy id, metus. Nullam accumsan lorem in dui. Cras ultricies mi eu turpis hendrerit fringilla. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In ac dui quis mi consectetuer lacinia. Nam pretium turpis et arcu. Duis arcu tortor, suscipit eget, imperdiet nec, imperdiet iaculis, ipsum. Sed aliquam ultrices mauris. Integer ante arcu, accumsan a, consectetuer eget, posuere ut, mauris. Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc. Nunc nonummy metus. Vestibulum volutpat pretium libero. Cras id dui. Aenean ut eros et nisl sagittis vestibulum. Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede. Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. Etiam imperdiet imperdiet orci. Nunc nec neque. Phasellus leo dolor, tempus non, auctor et, hendrerit quis, nisi. Curabitur ligula sapien, tincidunt non, euismod vitae, posuere imperdiet, leo. Maecenas malesuada. Praesent congue erat at massa. Sed cursus turpis vitae tortor. Donec posuere vulputate arcu. Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci. Phasellus consectetuer vestibulum elit. Aenean tellus metus, bibendum sed, posuere ac, mattis non, nunc. Vestibulum fringilla pede sit amet augue. In turpis. Pellentesque posuere. Praesent turpis. Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc, eu sollicitudin urna dolor sagittis lacus. Donec elit libero, sodales nec, volutpat a, suscipit non, turpis. Nullam sagittis. Suspendisse pulvinar, augue ac venenatis condimentum, sem libero volutpat nibh, nec pellentesque velit pede quis nunc. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Fusce id purus. Ut varius tincidunt libero. Phasellus dolor. Maecenas vestibulum mollis diam. Pellentesque ut neque. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. In dui magna, posuere eget, vestibulum et, tempor auctor, justo. In ac felis quis tortor malesuada pretium. Pellentesque auctor neque nec urna. Proin sapien ipsum, porta a, auctor quis, euismod ut, mi. Aenean viverra rhoncus pede. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Ut non enim eleifend felis pretium feugiat. Vivamus quis mi. Phasellus a est. Phasellus magna. In hac habitasse platea dictumst. Curabitur at lacus ac velit ornare lobortis. Curabitur a felis in nunc fringilla tristique. Morbi mattis ullamcorper velit. Phasellus gravida semper nisi. Nullam vel sem. Pellentesque libero tortor, tincidunt et, tincidunt eget, semper nec, quam. Sed hendrerit. Morbi ac felis. Nunc egestas, augue at pellentesque laoreet, felis eros vehicula leo, at malesuada velit leo quis pede. Donec interdum, metus et hendrerit aliquet, dolor diam sagittis ligula, eget egestas libero turpis vel mi. Nunc nulla. Fusce risus nisl, viverra et, tempor et, pretium in, sapien. Donec venenatis vulputate lorem. Morbi nec metus. Phasellus blandit leo ut odio. Maecenas ullamcorper, dui et placerat feugiat, eros pede varius nisi, condimentum viverra felis nunc et lorem. Sed magna purus, fermentum eu, tincidunt eu, varius ut, felis. In auctor lobortis lacus. Quisque libero metus, condimentum nec, tempor a, commodo mollis, magna. Vestibulum ullamcorper mauris at ligula. Fusce fermentum. Nullam cursus lacinia erat. Praesent blandit laoreet nibh. Fusce convallis metus id felis luctus adipiscing. Pellentesque egestas, neque sit amet convallis pulvinar, justo nulla eleifend augue, ac auctor orci leo non est. Quisque id mi. Ut tincidunt tincidunt erat. Etiam feugiat lorem non metus. Vestibulum dapibus nunc ac augue. Curabitur vestibulum aliquam leo. Praesent egestas neque eu enim. In hac habitasse platea dictumst. Fusce a quam. Etiam ut purus mattis mauris sodales aliquam. Curabitur nisi. Quisque malesuada placerat nisl. Nam ipsum risus, rutrum vitae, vestibulum eu, molestie.
            """
    # SENTENCES = [sent.strip() for sent in TEXT.split('\n') if sent.strip()]  # Split text into sentences
    # ENTITIES = {'dbpedia_spotlight': [
    #     ('http://dbpedia.org/resource/7_World_Trade_Center', '7 World Trade Center'),
    #     ('http://dbpedia.org/resource/Benchmarking', 'benchmark'),
    #     ('http://dbpedia.org/resource/Safety', 'safety'),
    #     ('http://dbpedia.org/resource/7_World_Trade_Center', '7 WTC'),
    #     # ("http://dbpedia.org/resource/Moody's_Investors_Service", 'Moody'),
    #     # ('http://dbpedia.org/resource/New_York_City', 'New York'),
    #     ('http://dbpedia.org/resource/Joe_Mansueto', 'Mansueto Ventures'),
    #     ('http://dbpedia.org/resource/MSCI', 'MSCI'),
    #     ('http://dbpedia.org/resource/Elisha_Cook_Jr.', 'Wilmer'),
    #     ('http://dbpedia.org/resource/Hale,_Greater_Manchester', 'Hale')]}
    # ENTITIES = [x[1] for x in ENTITIES["dbpedia_spotlight"]]

    # ENTITY_EXTRACTOR = EntityExtractor(options=["nps"])
    # ENTITIES = ENTITY_EXTRACTOR(text=TEXT)
    # ENTITIES = [x[1] for x in ENTITIES["dbpedia_spotlight"]]
    # print(ENTITIES)

    print("## WITHOUT ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT)
    print(RES)
    print("==========")
    print("## WITH ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT)
    print(RES)
