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

# Citation
Please cite us if it is useful in your work:
```
@inproceedings{wang2022hagen,
  title={Hagen: Homophily-aware graph convolutional recurrent network for crime forecasting},
  author={Wang, Chenyu and Lin, Zongyu and Yang, Xiaochen and Sun, Jiao and Yue, Mingxuan and Shahabi, Cyrus},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={4},
  pages={4193--4200},
  year={2022}
}
```

