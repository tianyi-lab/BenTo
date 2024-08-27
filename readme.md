# Benchmark Reduction with In-Context Transferability
This is the Github repo for Our paper 'Benchmark Reduction with In-Context Transferability'.

![image](images/combined-graph.png)

## Usage
You can install the environments with 
```
conda env create -f environment.yml
```

## Datasets
You can download the datasets at [huggingface](https://huggingface.co/datasets/cindermond/bento). 
We also support directly load datasets from huggingface by adding '--use_remote_data' flag in the inference script. In that case, '--data_folder' should be a cache directory to save the downloaded data.

## Inference
We provide an [example inference script](inference.sh).

## License
This repo and the dataset is under Apache 2.0 license. The dataset is based on [MMLU](https://arxiv.org/abs/2009.03300), [FLAN](https://arxiv.org/abs/2109.01652), [Big Bench Hard](https://arxiv.org/abs/2210.09261) and [AgiEval English](https://arxiv.org/abs/2304.06364). Please consider cite our paper and these original datasets if you find this work useful.

The non-"reduced" benchmark on huggingface is the original benchmark, except for FLAN, which is a sampled version. 
The "reduced" benchmark only contains a few representative tasks in the original ones, such that the performance on the "reduced" benchmark can serve as an approximation to the performance on the original ones.
