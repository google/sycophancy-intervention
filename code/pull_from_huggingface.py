"""Retrieves datasets from HuggingFace and builds them into dictionaries.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import random
from typing import List, Tuple, Dict
import datasets
import utils

DATA_FOLDER = 'data_source_nlp'
DATASET_NAMES = [
    'sst2',
    'rotten_tomatoes',
    'tweet_eval',
    'tweet_eval',
    'tweet_eval',
    'glue',
    'glue',
    'paws',
    'super_glue',
    'glue',
    'glue',
    'glue',
    'snli',
    'super_glue',
    'trec',
    'ag_news',
    'glue',
]
DATASET_SUBSETS = [
    '',
    '',
    'sentiment',
    'offensive',
    'irony',
    'qqp',
    'mrpc',
    'labeled_final',
    'rte',
    'wnli',
    'qnli',
    'mnli',
    '',
    'cb',
    '',
    '',
    'cola',
]
DATASET_LABELS = [
    ['Negative Sentiment', 'Positive Sentiment'],
    ['Negative Sentiment', 'Positive Sentiment'],
    ['Negative Sentiment', 'Neutral Sentiment', 'Positive Sentiment'],
    ['Not Offensive', 'Offensive'],
    ['Not Irony', 'Irony'],
    ['Not Duplicate', 'Duplicate'],
    ['Not Equivalent', 'Equivalent'],
    ['Different Meaning', 'Paraphrase'],
    ['Entailment', 'Not Entailment'],
    ['Not Entailment', 'Entailment'],
    ['Entailment', 'Not Entailment'],
    ['Entailment', 'Neither Entailment Nor Contradiction', 'Contradiction'],
    ['Entailment', 'Neither Entailment Nor Contradiction', 'Contradiction'],
    ['Entailment', 'Contradiction', 'Neither Entailment Nor Contradiction'],
    ['Abbreviation', 'Entity', 'Description or Abstract Concept', 'Human Being', 'Location', 'Numeric Value'],  # pylint:disable=line-too-long
    ['World', 'Sports', 'Business', 'Science and Technology'],
    ['Unacceptable Sentence', 'Acceptable Sentence'],
]
DATASET_FIELDS = [
    ['sentence'],
    ['text'],
    ['text'],
    ['text'],
    ['text'],
    ['question1', 'question2'],
    ['sentence1', 'sentence2'],
    ['sentence1', 'sentence2'],
    ['premise', 'hypothesis'],
    ['sentence1', 'sentence2'],
    ['sentence', 'question'],
    ['premise', 'hypothesis'],
    ['premise', 'hypothesis'],
    ['premise', 'hypothesis'],
    ['text'],
    ['text'],
    ['sentence'],
]
DATASET_LABEL_NAMES = [
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'label',
    'coarse_label',
    'label',
    'label',
]


def clean_input(example: str) -> str:
  example = example.replace('"', '')
  example = example.replace('\n', '')
  example = example.strip()
  example = f'"{example}"'
  return example


def build_dataset(
    dataset_name: str,
    dataset_subset: str,
    label_name: str,
    text_fields: List[str],
    natural_language_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
  """Uses inputted dataset details to build dictionary of train/test values."""
  if not dataset_subset:
    dataset_dict = datasets.load_dataset(dataset_name)
  else:
    dataset_dict = datasets.load_dataset(dataset_name, name=dataset_subset)

  train_dict, test_dict = {}, {}
  has_validation = False

  if 'validation' in dataset_dict:
    train_data, test_data = dataset_dict['train'], dataset_dict['validation']
    has_validation = True
  elif 'test' in dataset_dict:
    train_data, test_data = dataset_dict['train'], dataset_dict['test']
  else:
    print('WARNING: NO EXISTING SPLITS FOUND')
    temp = list(dataset_dict['train'])
    random.shuffle(temp)
    num_test_examples = int(.2 * len(temp))
    train_data, test_data = temp[:-num_test_examples], temp[-num_test_examples:]

  for data, dict_ in zip([train_data, test_data], [train_dict, test_dict]):
    for i in range(len(data)):
      text = ' and '.join([clean_input(data[i][x]) for x in text_fields])
      label = data[i][label_name]
      dict_[text] = natural_language_labels[label]

  print(f'Found {len(train_dict)} training examples')
  print(f'Found {len(test_dict)} testing examples')

  return train_dict, test_dict, has_validation


def build_all_datasets() -> None:
  """Builds all input:label dictionaries and saves them."""
  for name, subset, labels, fields, label_name in zip(
      DATASET_NAMES,
      DATASET_SUBSETS,
      DATASET_LABELS,
      DATASET_FIELDS,
      DATASET_LABEL_NAMES,
  ):
    full_name = name if not subset else f'{name}_{subset}'
    out_path = os.path.join(DATA_FOLDER, full_name + '.pickle')

    if not os.path.exists(DATA_FOLDER):
      os.mkdir(DATA_FOLDER)

    # Do not reprocess datasets
    if os.path.exists(out_path):
      continue

    train_dict, _, _ = build_dataset(name, subset, label_name, fields, labels)
    utils.print_an_example(train_dict)

    utils.save_pickle(out_path, train_dict)


def collect_all_datasets() -> Dict[str, str]:
  build_all_datasets()

  result = {}

  for name, subset in zip(DATASET_NAMES, DATASET_SUBSETS):
    full_name = name if not subset else f'{name}_{subset}'
    out_path = os.path.join(DATA_FOLDER, full_name + '.pickle')
    result.update(utils.load_pickle(out_path))

  return result
