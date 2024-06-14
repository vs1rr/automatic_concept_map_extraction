import os

from openai import OpenAI

api_key = ""

class ZeroBaseline:
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
        aggregated_text = " ".join(texts)

        chunk_size = 15000
        text_chunks = [aggregated_text[i:i + chunk_size] for i in range(0, len(aggregated_text), chunk_size)]

        concept_maps = []

        for chunk in text_chunks:
            prompt_template = """
                    Task Description: Concept Map Generation

                    Your task is to process a collection of {} and extract triples from them.

                    Subsequently, you'll aggregate this information to construct a unique and comprehensive Concept Map representing the information 
                    in all the texts in the given folder.

                    The resulting Concept Map should adhere to the following structure:
                    <Subject> - <Predicate> - <Object>,
                    <Subject> - <Predicate> - <Object>,
                    <Subject> - <Predicate> - <Object>,

                    The Concept Map should contain only the most important triple that best summarizes the content of all texts and avoid redundancy across triples.
                    In your answer, you must give the output in a .csv file with the columns `subject`, `predicate`, and `object`.

                    The output is a single:
                    ```csv 
                    """

            prompt = prompt_template.format(chunk)

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
    test_folder = "./data/Corpora_Falke/Wiki/test"
    output_folder = "./experiments_emnlp/baselines/zero_baseline/results-zero-train"
    concept_map_extractor = ZeroBaseline(api_key=api_key, model=model)

    for folder_name in os.listdir(test_folder):
        folder_path = os.path.join(test_folder, folder_name)
        if os.path.isdir(folder_path):
            concept_maps = concept_map_extractor.extract_concept_maps(folder_path)
            folder_number = folder_name.split("/")[-1]
            concept_map_extractor.save_concept_maps(concept_maps, output_folder, folder_number)
