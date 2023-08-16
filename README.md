# Simple synthetic data reduces sycophancy in large language models
This repository includes code for generating synthetic data for intervention, as proposed in ["Simple synthetic data reduces sycophancy in large language models"](https://arxiv.org/abs/2308.03958).

# Code files
`code/dataset_pipeline.py`: runnable script to generate synthetic data that will be saved as a pickle file.
`code/generate_data.py`: functions for generating synthetic finetuning data or evaluation data.
`code/pull_from_huggingface.py`: compiles NLP datasets from Hugging Face as specified in the paper.
`code/utils.py` : utility functions used by this repository.
`code/names.txt`: list of 10,000 names used for generating synthetic data.

# Requirements
Code generation requires installing the `datasets` package, which is used for downloading datasets from Hugging Face:
```
pip install datasets
```

# Generating synthetic finetuning data
To generate finetuning data, run `python code/dataset_pipeline.py`.
- This step can be skipped by changing`generate_train =  True` to `generate_train = False`.
- By default, the script will generate 100,000 prompt--answer pairs, but this can be changed by editing the `max_train_ex` field.

The outputted file will be saved as a `.pickle` file in the `data/` directory.

# Generating simple addition statements
To generate sycophancy evaluation data on simple addition statements, run `python code/dataset_pipeline.py`.
- This step can be skipped by changing `generate_synthetic_eval = True` to `generate_synthetic_eval = False`.
- By default, generated examples includes the simulated user opinions ("Human: Hello, my name is John Doe. I am currently a professor of Mathematics. I agree with the claim that..."), but this can be removed by setting `add_user_opinion` to `False`.

The outputted filed will be saved as a `.pickle` file in the `data/` directory.

# Citing
If you found our work helpful, you can cite us as the following:
```
@misc{wei2023simple,
    title={Simple synthetic data reduces sycophancy in large language models},
    authors={Jerry Wei and Da Huang and Yifeng Lu and Denny Zhou and Quoc V. Le},
    year={2023},
    url={arxiv.org/abs/2308.03958},
}
