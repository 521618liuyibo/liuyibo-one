## How to install
### Requirements: 
An [Anaconda python environment](https://www.anaconda.com/download) is recommend.
Check the environment.yml file, but primarily:
- Python >= 3.5
- numpy
- pandas
- sklearn
- RDKit
- xgboost


## Run
grid search:
```
python run.py --model Random Forest --grid_search True
```
use params:

`python run.py --model Random Forest`

## Output File

dataset split file

`/model_results/split_data`

model predict result file

`/model_results/result`
