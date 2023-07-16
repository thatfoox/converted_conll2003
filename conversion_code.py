from datasets import load_dataset
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    span_start: int
    span_end: int


@dataclass(frozen=True)
class ConvertedExample:
    sentence: str
    entities: List[Entity]


def convert_entities(tokens: List[str], ner_tags: List[int], sentence: str, entity_type_mapping: dict) -> List[Entity]:
    """
        converting entities to our scheme format.
    Arguments:
        :param tokens: List of tokens in the sentence.
        :param ner_tags: List of entity tags for each token.
        :param sentence: The sentence text.
        :param entity_type_mapping: Mapping of entity tag IDs to entity types.
    Returns:
        :return: List[Entity]: List of converted entities.
    """
    entities = []
    prev_entity = None
    token_start = 0

    for i in range(len(tokens)):
        token = tokens[i]
        ner_tag = ner_tags[i]
        if ner_tag != 'O' and ner_tag != 0:
            entity_type = entity_type_mapping[ner_tag]

            if prev_entity is not None and entity_type == prev_entity.type:
                prev_entity = Entity(
                    text=prev_entity.text + " " + token,
                    type=entity_type,
                    span_start=prev_entity.span_start,
                    span_end=sentence.index(token, prev_entity.span_start) + len(token)
                )
                entities[-1] = prev_entity
            else:
                entity = Entity(
                    text=token,
                    type=entity_type,
                    span_start=sentence.index(token, token_start),
                    span_end=sentence.index(token, token_start) + len(token)
                )
                entities.append(entity)
                prev_entity = entity
                token_start = entity.span_end + 1
        else:
            prev_entity = None
    return entities


def convert_conll_dataset(dataset) -> List[ConvertedExample]:
    """
     Converting conll dataset to our scheme format.
     Arguments:
        :param dataset: conll dataset.
     Returns:
        :return: List[ConvertedExample]: list of converted examples.
    """
    entity_type_mapping = {
        0: 'O',
        1: 'PER',
        2: 'PER',
        3: 'ORG',
        4: 'ORG',
        5: 'LOC',
        6: 'LOC',
        7: 'MISC',
        8: 'MISC'
    }

    converted_dataset = []

    for example in dataset:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        sentence = ""
        for token in tokens:
            if sentence and not token.startswith(("'", ",", "!", ".", "?", "%", ":", ";")):
                sentence += " "
            sentence += token
        entities = convert_entities(tokens, ner_tags, sentence, entity_type_mapping)
        converted_example = ConvertedExample(sentence=sentence, entities=entities)
        converted_dataset.append(converted_example)

    return converted_dataset


def main():
    # Load the conll2003 dataset (in this case, the "test" split)
    conll_dataset = load_dataset("conll2003", split="test")

    converted_conll_dataset = convert_conll_dataset(conll_dataset)
    output_path = Path("./converted_conll_test_dataset.parquet")
    assert not output_path.exists(), f"The output path contains a file already: {output_path}"
    df = pd.DataFrame([asdict(example) for example in converted_conll_dataset])
    df.to_parquet(output_path)

    print("Conversion completed and saved to:", output_path)


if __name__ == '__main__':
    main()
