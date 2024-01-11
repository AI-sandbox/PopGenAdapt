# PopGenAdapt

PopGenAdapt is a deep learning model that applies semi-supervised domain adaptation (SSDA) to improve genotype-to-phenotype prediction in underrepresented populations. The approach leverages the large amount of labeled data from well-represented populations, as well as the limited labeled and the larger amount of unlabeled data from underrepresented populations. This helps to address the imbalance present in genetic datasets, which predominantly represent individuals of European ancestry. 

## Installation

Install the dependencies with the following command:

```
pip install -r requirements.txt
```

Note that the project was developed using Python 3.11 and PyTorch 2.0.1.

## Usage

To train a model for a given source population (e.g. White British), a target population (e.g. Native Hawaiian),
and a phenotype (e.g. diabetes), you will need five [.npz files](https://numpy.org/doc/stable/reference/generated/numpy.savez.html):

- Labeled data from the source domain for training (e.g. `whitebritish_diabetes_train.npz`)
- Labeled data from the target domain for training (e.g. `hawaiian_diabetes_train.npz`)
- Unlabeled data from the target domain for training (e.g. `hawaiian_unlabeled.npz`)
- Labeled data from the target domain for validation (e.g. `hawaiian_diabetes_validation.npz`)
- Labeled data from the target domain for testing (e.g. `hawaiian_diabetes_test.npz`)

The .npz files for labeled data should contain two arrays:

- `x`: of shape `(n_samples, n_snps)`, containing the SNPs, encoded as `0`, `1`, `2`
- `y`: of shape `(n_samples,)`, containing the phenotypes, encoded as `0`, ..., `n_classes - 1`

The .npz file for unlabeled data should contain only the `x` array. Note that regression tasks are not supported.

Then, create a dataset configuration file in JSON format with the same structure as the following example, but with your own paths:

```json
{
    "root": "/home/salcc/PopGenAdapt/data",
    "source": {
        "train": "whitebritish_diabetes_train.npz"
    },
    "target": {
        "train": "hawaiian_diabetes_train.npz",
        "unlabeled": "hawaiian_unlabeled.npz",
        "validation": "hawaiian_diabetes_validation.npz",
        "test": "hawaiian_diabetes_test.npz"
    }
}
```

Finally, execute the following command to train the model:

```
python main.py --data dataset.json --mme --sla
```

Run `python main.py --help` to see all the available options.


### Hyperparameter Tuning

The choice of hyperparameters can have a significant impact on the performance of the model.
We use [Weights & Biases](https://wandb.ai/) to perform hyperparameter search. Given a dataset and a method, to find good hyperparameters,
run `python sweep.py --data dataset.json --mme --sla` to initialize a hyperparameter sweep and start an agent.

## Citation

If you use PopGenAdapt in your research, please cite our [paper](https://psb.stanford.edu/psb-online/proceedings/psb24/comajoan.pdf):

```
@inproceedings{comajoan2023popgenadapt,
author = {Comajoan Cara, Marçal and Mas Montserrat, Daniel and Ioannidis, Alexander},
title = {PopGenAdapt: Semi-Supervised Domain Adaptation for Genotype-to-Phenotype Prediction in Underrepresented Populations},
booktitle = {Biocomputing 2024},
pages = {327-340},
doi = {10.1142/9789811286421_0026},
URL = {https://psb.stanford.edu/psb-online/proceedings/psb24/comajoan.pdf},
eprint = {https://www.worldscientific.com/doi/pdf/10.1142/9789811286421_0026},
}
```

## Acknowledgement

The code is partially based on [MME](https://github.com/VisionLearningGroup/SSDA_MME) and [SLA](https://github.com/chu0802/SLA).
