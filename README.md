# GNN-based Recommender Systems Reproducibility

This is the official repository for the paper 
"_GNN-based Recommender Systems Reproducibility_".

This repository is heavily dependent on the framework **Elliot**, 
so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) 
and [documentation](https://elliot.readthedocs.io/en/latest/).

## Prerequisites

We implemented and tested our models using `Python 3.10.12` and `PyTorch==2.1.0`, 
with CUDA `12.1`. 
Additionally, some of graph-based models require `PyTorch Geometric`, 
which is compatible with the versions of CUDA and `PyTorch` we indicated above.
Specifically, you may create a virtual environment using with the 
`requirements.txt` file we included in this repository.


## Datasets

### Reproducibility datasets

For our reproducibility study we used **Gowalla**, **Yelp 2018**, and **Amazon Book** datasets.
In `./data/` we provide the tsv files for all the datasets in this repo, 
already in a format compatible for Elliot (i.e., tsv file with user/item).

The original links may be found here, where the train/test splitting has already been provided:
[Gowalla](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla), 
[Yelp 2018](https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018), 
[Amazon Book](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/amazon-book).

After downloading, create three folders ```./data/{dataset_name}```, 
one for each dataset. Then, run the script ```./map_dataset.py```, 
by changing the name of the dataset within the script itself. 
It will generate the train/test files for each dataset in a format compatible 
for Elliot.

### Additional datasets
In `./data/gowalla/` we provide the train/validation/test splittings for the four different versions
of the Gowalla dataset, i.e., [Wang et al.](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla), 
[Peng et al.](https://github.com/tanatosuu/svd_gcn/tree/main/datasets/gowalla), 
[Cai et al.](https://github.com/HKUDS/LightGCL/tree/main/data/gowalla), and 
[Yang et al.](https://github.com/YangLiangwei/GraphAU/tree/main/datasets/gowalla).

In `./data/` we also provide the train/validation/test splittings for the additional datasets
used in the banchmarking analysis: 
[Allrecipes](https://github.com/elisagao122/HAFR), 
[BookCrossing](https://dl.acm.org/doi/pdf/10.1145/1060745.1060754), 
[CiteULike(a)](https://www.datarechub.com/assets/pages/datasets/citeulike_a/), 
[Amazon Beauty](https://github.com/YangLiangwei/GraphAU/tree/main/datasets/amazon-beauty),
and [Amazon CDs](https://huggingface.co/datasets/reczoo/AmazonCDs_m1/tree/main).

## Usage

To reproduce our experiments, you have to train, validate and test a model on
a specific dataset.
To do so, you should run the following script (with arguments):

```
$ python start_experiments.py \
$ --dataset {dataset_name} \
$ --model {model_name} 
```

### Notes

Before running the experiments, consider the following notes:

* In `start_experiments.py`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (which may change depending on your configuration) 
is set to enable deterministic behavior with CUDA, 
in order to ensure the complete reproducibility of the experiments.
* The hyper-parameter settings for the models are available in the 
`./config_files/` folder, where all configuration files are stored
and named in this way: `<model>_<dataset_name>.yml`.
* Depending on the model, the dataset and your workstation, 
the training and evaluation could take long time. 
* After the training and evaluation are done, you will find all performance files 
in the folder `./results/<dataset_name>/performance/`.

## Research Questions

To reproduce the experiments of our Research Questions (RQs), 
please follow the experimental settings described in the paper
and the following notes:

### RQ1: Reproducibility analysis

Dataset used: Gowalla (by Wang et al.), Yelp-2018, Amazon-book.

For RQ1 you just have to train and test the models, 
since  optimal configurations for each dataset are provided
within the respective publications and/or associated code repositories.

### RQ2: Benchmarking with different splits

Datasets used: Gowalla by Wang et al., Peng et al., Cai et al., and Yang et al.

The dataset name in the config files for
the validation phase is `<dataset_name>=gowalla_<version>`,
where `<version>` is wang, peng, cai or yang. For the test phase, the dataset name is
`<dataset_name>=gowalla_<version>_test`.

The `./config_files/gowalla_<version>.yml` files are used to split the 
original datasets in order to obtain the validation set when it is
not provided in the original repository. The splits are also provided in 
`./data/gowalla/<version>/`.


### RQ3: Benchmarking with new baseline models

Dataset used: Gowalla (by Wang et al).

In order to train, validate and test the baseline models 
(Reference and Classic CF models),
you can follow the general instructions.


### RQ4: Benchmarking with new datasets

Datasets used: Allrecipes, BookCrossing, CiteULike(a), Amazon Beauty, Amazon CDs.

The `./config_files/<dataset_name>.yml` files are used to split the 
datasets in a train, validation and test sets. 
The splits are also provided in `./data/<dataset_name>/`.

