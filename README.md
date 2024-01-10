# MediLeaf-AI

MediLeaf is an application whose motive is to help the individual to identify medicinal plant with their properties by just scanning the leaf of any plant which might result creating curiosity about plant that lead to the preservation of the valuable plants as well as source of income. This is also an end to end deep learning project focusing from development to deployment.

MediLeaf model can classify leaf of following types of plant.
|           Plants              |        Plants                 |           Plants              |
|-------------------------|-------------------------|-------------------------|
| Alpinia Galanga         | Amaranthus Viridis      | Artocarpus Heterophyllus|
| Azadirachta Indica      | Basella Alba            | Brassica Juncea         |
| Carissa Carandas        | Citrus Limon            | Ficus Auriculata        |
| Ficus Religiosa         | Hibiscus Rosa-sinensis  | Jasminum                |
| Mangifera Indica        | Mentha                  | Moringa Oleifera        |
| Muntingia Calabura      | Murraya Koenigii        | Nerium Oleander         |
| Nyctanthes Arbor-tristis| Ocimum Tenuiflorum       | Piper Betle             |
| Plectranthus Amboinicus | Pongamia Pinnata        | Psidium Guajava         |
| Punica Granatum         | Santalum Album          | Syzygium Cumini         |
| Syzygium Jambos         | Tabernaemontana Divaricata| Trigonella Foenum-graecum|


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


## How to Train?

### A. Normal way of training

1. Clone the repository
```bash
https://github.com/surajkarki66/MediLeaf_AI
```
2. Create a Python virtual environment and activate the environment based on your machine(Linux, MacOS, and Windows)

3. Install the requirements
```bash
pip install -r requirements.txt
```
3. Run the following command

```bash
# Finally run the following command
python main.py
```

Now,
```bash
Model training will start soon
```

`Note`: Check `params.yaml` to tweak the configuration of the model.

### B. Using MLops tool DVC for training

1. First run
```bash
dvc repro
```

2. Then,
```bash
dvc dag
```


## Live Preview
To check the live demo of the prediction, [click here]()

## Running Prediction API locally

