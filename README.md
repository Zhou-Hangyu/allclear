# allclear

## Setup
`AllClear` comes with minimal package requirements. To further facilitate environment setup, you can use the provided
`benchmark_dependencies.yml` to update your existing conda environment for benchmarking purposes. Please run the following 
command, replacing `your_env_name` with the name of your conda environment:
```bash
conda env update -n your_env_name -f benchmark_dependencies.yml
```
This will install any packages that are missing from your environment but required by AllClear, while preserving the rest 
of your environment's setup.



This repository contains multiple baseline repo as submodules. To include them in the project, run the following command:

```bash
git submodule update --init --recursive
```




## Internal Notes (for developers)
* The main package folder is `allclear`. Should only contain reusable code directly related to the use of the dataset and benchmark.
* Every baseline we proposed or reproduced should have one folder in the `/baselines` folder.
  * They will have a wrapper in `allclear/baselines.py` with uniform input/output format for easy comparison.
* The `demo` folder contains minimal code to demonstrate the use of the dataset and benchmark.
* For all other code, please put them in the `/experimental_scripts` folder for now.