# MediLeaf-AI

Medileaf is an application whose motive is to help the individual to identify medicinal plant with their properties by just scanning the leaf of any plant which might result creating curiosity about plant that lead to the preservation of the valuable plants as well as source of income.

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


## How to Run?
### STEPS:

Clone the repository

```bash
https://github.com/surajkarki66/MediLeaf_AI
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medileaf python=3.8 -y
```

```bash
conda activate medileaf
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python main.py
```

Now,
```bash
Model training will start soon
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag



## Model deployment locally in BentoML
### STEPS:
Must activate the conda environment `medileaf`
### STEP 01- Run

```bash
python deploy.py
```
It will convert tensorflow model into a BentoML model which will be stored locally in your computer.
To check whether the model is created or not enter below command.
```bash
bentoml models list
```
The output will look something like this:

![bentoml](https://github.com/surajkarki66/MediLeaf_AI/assets/50628520/1d079582-31d4-4cc4-8f70-ce1047e9c068)

### STEP 02
Copy the Tag shown in the above output and add it inside the `service.py` file in the following way:

`BENTO_MODEL_TAG = <copied tag>`


### STEP 03- Run the following command to serve the model locally in BentoML
```bash
BENTOML_CONFIG=/home/surajkarki/Documents/My-Workspace/My-Work/Final-Year-Project/MediLeaf_AI/bentoml_configuration.yaml bentoml serve service:medileaf_service --reload --development
```

Now, open http://localhost:3001/

Classification API: http://localhost:3001/classify