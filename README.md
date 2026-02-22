# FFF memory acceleration PyTorch branch

### Pick your config file:
```bash
    cp conf/main_local.yaml conf/main.yaml
```

### Run an experiment:
```bash
    python src/main.py --multirun exp=myexp seed=1,2,3,4
```

### Export dataset sample binaries
Set batch size as the number of samples in the testset so that all samples are exported in one go.
```bash
    python src/main.py exp=export --multirun loader=mnist loader.batch_size=10000
    python src/main.py exp=export --multirun loader=sc loader.batch_size=4890
    python src/main.py exp=export --multirun loader=ms loader.batch_size=12000
```
