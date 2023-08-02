"""Pipeline for generating synthetic tuning/evaluation data.

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
import generate_data as gd
import utils

DATA_DIR = 'data'
utils.ensure_dir(DATA_DIR)

generate_train = True
max_train_ex = 1e5

generate_synthetic_eval = True
add_user_opinion = True
################################################################################
#################                    MAIN                   ####################
################################################################################
if generate_train:
  generated_examples = gd.generate_nlp_data(max_train_ex)
  utils.print_an_example(generated_examples)
  out_name = f'synthetic_train_{len(generated_examples)}.tsv'
  out_path = os.path.join(DATA_DIR, out_name)
  utils.save_pickle(out_path, generated_examples)

if generate_synthetic_eval:
  examples = gd.generate_math_eval_data(add_user_opinion)
  utils.print_an_example(examples)
  out_name = f'synthetic_eval_opinion{add_user_opinion}_{len(examples)}.tsv'
  out_path = os.path.join(DATA_DIR, out_name)
  utils.save_pickle(out_path, examples)
