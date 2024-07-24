# nma_changing_things

[Reading material for project](https://github.com/pravsels/nma_changing_things/blob/main/reading.md)


## Setup for conda

Build conda environment: 
```
conda env create -f conda_env.yml
```

Activate the conda environment: 
```
conda activate nct
```

## Run

Run sample script: 
```
python random_actions.py 
```

## Train

Run train script: 
```
python training/mlp_train.py 
```

## Running saved model 

Simulate worm using trained model: 
```
python play_saved_model.py --model_folder=[folder_name]
```
Where [folder_name] is the name of thefolder containg the experiment you want to play (ex. mlp_128)

## Plotting  

Plot the performances of the MLP and NCAP models: 
```
python plotting_logs.py 
```

## Network viz  

Plot visualizations of the two networks: 
```
python draw_network.py 
```

## Bonus: Setup for docker 

Build docker container:
```
chmod u+x *.sh

./docker_build.sh
```

Run the container:
```
./run_docker_container.sh
```
