## FedAdvan (randomly named)
This is a framework for federated learning.

### structure
- clients
  - client.py (client base)
  - [fl algorithm].py (client in different algorithm)
- dataset
  - collector
    - tools.py(seperating dataset)
  - agnews.py (dataset provided)
  - download.py (download dataset)
- fedlog
- models (initialize models in clients respectively)
- servers
  - serverbase.py (server base)
  - serverapi.py (execute entrance)
- utils (utils package)
- config.yaml(config file, help to construct parameters)

### Algorithm support
- FedAvg
- Ditto

### Run it
```bash
python main.py
```
or
```bash
python main.py -f config.yaml
```

### future work
1. Supporting command parameters input.
2. Emulating parallel computing structure
3. Implementing more general federated learning algorithms