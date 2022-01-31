# Differentiable Causal Discovery from Interventional Data

![DCDI paper graphics](https://github.com/slachapelle/dcdi/raw/master/paper_graphic.png)

Repository for "Differentiable Causal Discovery from Interventional Data".

* The code for DCDI is available [here](./dcdi).

* The datasets used in the paper are available [here](./data)

The code for the following baselines is also provided:
* [CAM](./cam)
* [GIES](./gies)
* [IGSP and UT-IGSP](./igsp)

To run DCDI, you can use a command similar to this one:
`python ./dcdi/main.py --train --data-path ./data/perfect/data_p10_e10_n10000_linear_struct --num-vars 10 --i-dataset 1 --exp-path exp --model DCDI-DSF --intervention --intervention-type perfect --intervention-knowledge known`

Here, we assume the zip files in the directory data have been unzipped and that the results will be saved in the directory `exp`. With this command, you will train the model DCDI-DSF in the perfect known setting (with reasonable default hyperparameters) using the first instance (`i_dataset = 1`) of the 10-node graph dataset with linear mechanisms and perfect intervention. For further details on other hyperparameters (e.g. the architecture of the networks), see the `main.py` file where hyperparameters have a description.
