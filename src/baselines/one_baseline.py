import os

from openai import OpenAI

api_key = ""

class OneBaseline:
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_concept_maps(self, folder_path):
        texts = self._load_texts(folder_path)
        concept_maps = self._generate_concept_maps(texts)
        return concept_maps

    def _load_texts(self, folder_path):
        texts = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r") as file:
                    texts.append(file.read())
        return texts

    def _generate_concept_maps(self, texts):
     # Aggregate all texts into a single string
     aggregated_text = " ".join(texts)

     # Split the aggregated text into smaller chunks
     chunk_size = 15000
     text_chunks = [aggregated_text[i:i + chunk_size] for i in range(0, len(aggregated_text), chunk_size)]

     concept_maps = []

     for i, chunk in enumerate(text_chunks):
      prompt_template = """
                 Task Description: Concept Map Generation

                 Your task is to process a collection of texts and extract triples from them.

                 Subsequently, you'll aggregate this information to construct a comprehensive and unique Concept Map that represents all the texts in the given folder.

                 To illustrate, you can reference the Concept Map contained within the {folder_number} file, and consider the sample files from which it was extracted provided {folder_path}.

                 The resulting Concept Map should adhere to the following structure:
                 Each line in the map should represent a relationship between entities.
                 The format for each line is as follows: <Subject> - <Predicate> - <Object>.
                 Multiple lines will comprise the Concept Map, each representing a distinct relationship derived from the aggregated text data.

                 """
      prompt = prompt_template.format(folder_number=f"concept_map_{i + 1}", folder_path=chunk)

      completion = self.client.chat.completions.create(
       model=self.model,
       messages=[{"role": "user", "content": prompt}],
       temperature=0
      )

      concept_maps.append(completion.choices[0].message.content)

     return concept_maps

    def save_concept_maps(self, concept_maps, output_folder, folder_number):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, f"concept_map_{folder_number}.csv")

        with open(file_path, "w") as file:
            file.write(concept_maps[0])

        print(f"Concept map saved to: {file_path}")

if __name__ == '__main__':
    api_key = api_key
    model = "gpt-3.5-turbo-0125"
    test_folder = "./data/Corpora_Falke/Wiki/train"
    output_folder = "./experiments_emnlp/baselines/one_baseline/results-one-train"
    concept_map_extractor = OneBaseline(api_key=api_key, model=model)

    for folder_name in os.listdir(test_folder):
        folder_path = os.path.join(test_folder, folder_name)
        if os.path.isdir(folder_path):
            concept_maps = concept_map_extractor.extract_concept_maps(folder_path)
            folder_number = folder_name.split("/")[-1]
            concept_map_extractor.save_concept_maps(concept_maps, output_folder, folder_number)
