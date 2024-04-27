# allclear

`AllClear` is a comprehensive dataset/benchmark for cloud detection and removal. 


## Setup
`AllClear` comes with minimal package requirements. It can be easily installed using pip. 
Please navigate to the root directory of this project and run the following commands:

```bash
pip install -e .
```

This repository contains multiple baseline repo as submodules. To include them in the project, run the following command:

```bash
git submodule update --init --recursive
```

## License

This project is licensed under the [MIT License](LICENSE).


## Internal Notes (for developers)
* The main package folder is `allclear`. Should only contain reusable code directly related to the use of the dataset and benchmark.
* Every baseline we proposed or reproduced should have one folder in the `/baselines` folder.
  * They will have a wrapper in `allclear/baselines.py` with uniform input/output format for easy comparison.
* The `demo` folder contains minimal code to demonstrate the use of the dataset and benchmark.
* For all other code, please put them in the `/experimental_scripts` folder for now.