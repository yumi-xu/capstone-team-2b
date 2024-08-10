#!/usr/bin/env bash

# conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia \
# dask-cudf cupy scikit-learn matplotlib plotly \
# cudf=24.06 cuml=24.06 python=3.11 cuda-version=12.2
# pip install kaleido
pip install pandas scikit-learn imbalanced-learn transformers torch seaborn matplotlib
