# benchmark-reduction
This repo contains the code for reproducing the main results for our paper ``Benchmark Reduction with In-Context Transferability''. 

## Prepare datasets
You need to create a data folder and download MMLU datasets from [OpenNMT](https://github.com/OpenNMT/OpenNMT-py/tree/master/eval_llm/MMLU/data). You also need to create an empty folder called images.

## Run the code
The most simple way to replicate our main results is simply run [analysis.py](analysis.py) and [analysis_benchmark.py](analysis_benchmark.py) since we've uploaded the required log files. You can also get those files by: 
1. Run [run_all.py](run_all.py) with start_task_id ranging from 0 to 56 to get the ICT matrix.
2. Read the log with [save_csv.py](save_csv.py). 
3. You can compute the set that maximizes the facility location function with [facility_location.py](facility_locaiton.py). 

