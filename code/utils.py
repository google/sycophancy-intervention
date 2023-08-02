"""Contains common utility functions to be used across code files.

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
import pickle
import random
from typing import Any, Dict, List


def print_progress(sentence: str, progress: int, out_of: int) -> None:
  """Prints the given sentences with a progress bar."""
  if progress == 0:
    print()

  sentence = sentence + ' ' if sentence[-1] != ' ' else sentence
  num_remaining = os.get_terminal_size().columns - len(sentence) - 2

  if num_remaining >= 5:
    num_fill = int(float(progress / out_of) * num_remaining) or 1
    fill = '=' * num_fill
    fill = fill[1:] + '>' if fill else '>'
    empty = ' ' * (num_remaining - num_fill)
    sentence = f'{sentence}[{fill}{empty}]'

  print(sentence, end='\r')

  if progress == out_of:
    print()


def ensure_dir(folder_path: str) -> bool:
  if '/' in folder_path:
    folder_path = '/'.join(folder_path.split('/')[:-1])

  if not os.path.exists(folder_path):
    os.mkdir(folder_path)
    return True

  return False


def print_an_example(input_dict: Dict[Any, Any]) -> None:
  """Prints one key, value pair from the dictionary."""
  print(f'-------{len(input_dict)} total examples-------')
  if input_dict:
    keys = list(input_dict.keys())
    key = keys[random.randint(0, len(keys) - 1)]
    print(f'------Key------\n{key}')
    print(f'------Value------\n{input_dict[key]}')


def load_txt(filename: str) -> List[str]:
  with open(filename) as f:
    return f.readlines()


def save_pickle(filename: str, data: Dict[Any, Any]) -> None:
  if os.path.exists(filename):
    print(f'WARNING: PICKLE ALREADY EXISTS AT {filename}, CONTINUING')
    return

  with open(filename, 'wb') as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    print(f'Saved pickle at {filename}')


def load_pickle(filename: str) -> Dict[Any, Any]:
  with open(filename, 'rb') as file:
    return pickle.load(file)
