import pytest
from conversion_code import convert_conll_dataset, ConvertedExample


def test_convert_conll_dataset():
    dataset = [
        {'tokens': ["Syria", ":", "24", "-", "Salem", "Bitar", ",", "3", "-", "Bachar", "Srour", ";", "4", "-", "Hassan",
                    "Abbas", ",", "5", "-", "Tarek", "Jabban", ",", "6", "-", "Ammar", "Awad", "(", "9", "-",
                    "Louay", "Taleb", "69", ")", ",", "8", "-", "Nihad", "al-Boushi", ",", "10", "-",
                    "Mohammed", "Afash", ",", "12", "-", "Ali", "Dib", ",", "13", "-", "Abdul", "Latif",
                    "Helou", "(", "17", "-", "Ammar", "Rihawiy", "46", ")", ",", "14", "-", "Khaled", "Zaher",
                    ";", "16", "-", "Nader", "Jokhadar", "." ],
         'ner_tags': [5, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1,
                      2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 2, 0, 0, 0, 1, 2, 0,
                      0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0]}
    ]

    converted_dataset = convert_conll_dataset(dataset)

    assert len(converted_dataset) == 1

    example = converted_dataset[0]
    assert example.sentence == 'Syria: 24 - Salem Bitar, 3 - Bachar Srour; 4 - Hassan Abbas, 5 - Tarek Jabban,' \
                               ' 6 - Ammar Awad ( 9 - Louay Taleb 69 ), 8 - Nihad al-Boushi, 10 - Mohammed Afash,' \
                               ' 12 - Ali Dib, 13 - Abdul Latif Helou ( 17 - Ammar Rihawiy 46 ), 14 - Khaled Zaher;' \
                               ' 16 - Nader Jokhadar.'

    entities = example.entities
    assert len(entities) == 14

    entity1 = entities[0]
    assert entity1.text == example.sentence[0:5]
    assert entity1.type == 'LOC'
    assert entity1.span_start == 0
    assert entity1.span_end == 5

    entity2 = entities[1]
    assert entity2.text == example.sentence[12:23]
    assert entity2.type == 'PER'
    assert entity2.span_start == 12
    assert entity2.span_end == 23

    entity3 = entities[2]
    assert entity3.text == example.sentence[29:41]
    assert entity3.type == 'PER'
    assert entity3.span_start == 29
    assert entity3.span_end == 41

    entity4 = entities[3]
    assert entity4.text == example.sentence[47:59]
    assert entity4.type == 'PER'
    assert entity4.span_start == 47
    assert entity4.span_end == 59

    entity5 = entities[4]
    assert entity5.text == example.sentence[65:77]
    assert entity5.type == 'PER'
    assert entity5.span_start == 65
    assert entity5.span_end == 77

    entity6 = entities[5]
    assert entity6.text == example.sentence[83:93]
    assert entity6.type == 'PER'
    assert entity6.span_start == 83
    assert entity6.span_end == 93

    entity7 = entities[6]
    assert entity7.text == example.sentence[100:111]
    assert entity7.type == 'PER'
    assert entity7.span_start == 100
    assert entity7.span_end == 111

    entity8 = entities[7]
    assert entity8.text == example.sentence[122:137]
    assert entity8.type == 'PER'
    assert entity8.span_start == 122
    assert entity8.span_end == 137

    entity9 = entities[8]
    assert entity9.text == example.sentence[144:158]
    assert entity9.type == 'PER'
    assert entity9.span_start == 144
    assert entity9.span_end == 158

    entity10 = entities[9]
    assert entity10.text == example.sentence[165:172]
    assert entity10.type == 'PER'
    assert entity10.span_start == 165
    assert entity10.span_end == 172

    entity11 = entities[10]
    assert entity11.text == example.sentence[179:196]
    assert entity11.type == 'PER'
    assert entity11.span_start == 179
    assert entity11.span_end == 196


if __name__ == '__main__':
    pytest.main()
