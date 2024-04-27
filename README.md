# allclear

## Setup


This repository contains multiple baseline repo as submodules. To include them in the project, run the following command:

```bash
git submodule update --init --recursive
```

## Internal Notes (for developers)
* The main package folder is `allclear`. Should only contain reusable code directly related to the use of the dataset and benchmark.
* Every baseline we proposed or reproduced should have one folder in the `/baselines` folder.
  * They will have a wrapper in `allclear/baselines.py` with uniform input/output format for easy comparison.
* 