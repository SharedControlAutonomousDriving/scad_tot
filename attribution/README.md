## Understanding Model Behaviour using Attribution Analysis

### Contributor
Aman Mohanty

Contact at: amanmoha@andrew.cmu.edu 

### Description
*coming soon*

### Dependencies
Check `./requirements.txt`

### Model
For old model, please use `scad_tot/network/models/latest/model.h5`. Download the old model to `./tot_models`

For new data, *coming soon*

### Dataset
Old data: Under the **Other dataset files** in [README.md](https://github.com/SharedControlAutonomousDriving/scad_tot/blob/master/README.md) of the repository. Please download old data to `./old-data`

New data: *coming soon*

Models trained on data with dropped features can be found in `./attribution-models`.

Feature importances using the SHAP, LIME and Integrated Gradients can be found in `./importance-data`.

### How to run
To run importance analysis, please run the `./codes/importance-code.ipynb`. 

To run model re-training, please run the `./codes/network-training.ipynb`

Please make sure to change the FOLDERPATH in the notebook files. If the repository is cloned to downloads then `FOLDERPATH = '/home/downloads/scad-tot/attribution/codes/`. 

### Results
All the results of the project can be found in `./results`
