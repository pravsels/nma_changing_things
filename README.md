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
python training/main.py 
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
