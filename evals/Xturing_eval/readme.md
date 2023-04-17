# Xturing_evals

## Description
This is a tool for evaluating large language models supported by Xturing

## Installation

- Make sure the python version being run is 3.9 or higher.
- Load the evals submodule when cloning Xturing by running the following:
    - `git clone --recursive git@github.com:stochasticai/xturing.git`
- cd into the evals folder
- Install the dependencies by running
  - `pip install -e .`
- Add the relevant evaluation datasets under `/evals/registry/data` from the [drive folder](https://drive.google.com/drive/folders/14pldYTNpB19WAmF_i1XEGl2id4INrSIE?usp=sharing)

## Running an evaluation

There are two options, either use a cli by running:
- `!EVALS_THREADS=1 oaieval_custom gpt2_lora test-match`

Or by directly calling the `evaluate` function under `evals/Xturing_eval/evaluate.py` (work in progress)



