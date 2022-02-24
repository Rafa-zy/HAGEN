# HAGEN: Homophily-Aware Graph Convolutional Recurrent Network for Crime Forecasting

This is a PyTorch implementation of  **HAGEN: Homophily-Aware Graph Convolutional Recurrent Network for Crime Forecasting**.



## Requirements

- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- pyyaml
- statsmodels
- torch
- tables
- future
- sklearn

Dependency can be installed using the following command:

```bash
pip install -r requirements.txt
```



## Model Training

Here are commands for training the model on `LA`.

```bash
python hagen_train.py --config_filename ./crime-data/CRIME-LA/la_crime_9.yaml --month 9
```

Experimental settings and some supplemental results can be referred to `HAGEN_suppl.pdf`.
