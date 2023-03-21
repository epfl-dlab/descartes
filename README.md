# Descartes
This repository contains the PyTorch implementation for the experiments in [Descartes: Generating Short Descriptions of Wikipedia Articles](https://arxiv.org/pdf/2205.10012.pdf). The codebase was built upon [huggingface transformers library](https://huggingface.co/docs/transformers/index).

```
@article{sakota2022descartes,
  title={Descartes: generating short descriptions of wikipedia articles},
  author={Sakota, Marija and Peyrard, Maxime and West, Robert},
  journal={arXiv preprint arXiv:2205.10012},
  year={2022}
}
```
Please consider citing our work, if you found the provided resources useful.

## Setup instructions

Start by cloning the repository:
```
git clone https://github.com/epfl-dlab/descartes.git
```
### Environment setup
We recomment creating a new conda virtual environment:
```
conda env create -n descartes --file=requirements.yaml
conda activate descartes
```
Install transformers from source:
```
cd transformers
pip install -e .
```
Then, install the remaining packages using:
```
pip install -r versions.txt
```

## Usage

### 1. Training
To train the model, run the `train_descartes.sh` script. To specify the directory where data is located use `--data_dir`. For the directory where model checkpoints will be saved, specify `--output_dir`. If you want to use knowledge graph embeddings, use `--use_graph_embds` flag. If you want to use existing descriptions, specify the path of the model used to embed them with `--bert_path` (for example, `bert-base-multilingual-uncased`). For monolingual baselines, use `--baseline` tag.

### 2. Testing
To test the model, run the `test_model.sh` script. To specify the directory where data is located use `--data_dir`. For the directory where model was stored use `--output_dir`. For the directory where textual outputs will be saved, use `--output_folder`.

## License
Distributed under the MIT License. See [LICENSE](https://github.com/epfl-dlab/descartes/blob/main/LICENSE) for more information.
