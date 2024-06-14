from typing import Dict, List

class PostProcessor:
    """ Main class for post-processing """

    def __init__(self):
        """ Initialize the post-processor """
        self.processed_triples = {}

    def remove_redundant_triples(self, triples: Dict[str, List[tuple]]) -> Dict[str, List[tuple]]:
        """ Remove redundant triples """
        self._find_unique_triples(triples)
        return self.processed_triples

    def _find_unique_triples(self, triples: Dict[str, List[tuple]]) -> None:
        """ Find unique triples """
        self.processed_triples = {}
        for category, values in triples.items():
            unique_triples = []
            for triple in values:
                elements = triple
                unique_elements = set(elements)
                if len(unique_elements) == 3:
                    # Check for overlapping of elements in the triple
                    overlap_threshold = 0.6
                    for existing_triple in unique_triples:
                        existing_elements = existing_triple
                        overlap_count = sum(1 for element in elements if element in existing_elements)
                        if overlap_count / len(unique_elements) >= overlap_threshold:
                            # If overlap exceeds 60%, skip the triple
                            break
                    else:
                        # If no overlap exceeds 60%, add the triple to unique triples
                        unique_triples.append(elements)
            self.processed_triples[category] = unique_triples

if __name__ == "__main__":
    post_processor = PostProcessor()
    triples = {'rebel': [('7 World Trade Center', 'area', '52'), ('7 World Trade Center', 'elevation above sea', '52'), ('7 World Trade Center', 'instance of', 'building')], 'dependency': [('A', 'B', 'C'),('A', 'B', 'C')]}
    unique_triples = post_processor.remove_redundant_triples(triples)
    for category, triples_list in unique_triples.items():
        print(f"{category}:")
        for triple in triples_list:
            print(triple)