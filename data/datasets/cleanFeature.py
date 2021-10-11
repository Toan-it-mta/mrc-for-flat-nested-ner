import json

# Opening JSON file
f = open('E:\\Code\\Python\\mrc-for-flat-nested-ner\\data\\datasets\\conll03\\mrc-ner.dev', encoding='utf-8')

# returns JSON object as
# a dictionary
list_data_clean = []
list_data = json.load(f)
print(len(list_data))
# Iterating through the json
# list
for data in list_data:
    data_clean = dict()
    data_clean['context'] = data['context']
    data_clean['end_position'] = data['end_position']
    data_clean['entity_label'] = data['entity_label']
    data_clean['query'] = data['query']
    data_clean['start_position'] = data['start_position']

    list_data_clean.append(data_clean)

print(list_data_clean)
# Closing file
f.close()

with open('E:\\Code\\Python\\mrc-for-flat-nested-ner\\data\\datasets\\conll03\\mrc-ner.dev', 'w',
          encoding='utf-8') as fout:
    json.dump(list_data_clean, fout)

fout.close()

# f = open('E:\\Code\\Python\\mrc-for-flat-nested-ner\\data\\datasets\\conll03\\mrc-ner.train.json', encoding='utf-8')
# list_data = json.load(f)
# print(len(list_data))
#
# f.close()
