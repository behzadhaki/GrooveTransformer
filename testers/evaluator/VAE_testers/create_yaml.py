#create a dictionary

sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'loss',
    'goal': 'minimize'
    }
sweep_config['metric'] = metric

#save it like a yaml
import yaml
f = open('sweep.yaml', 'w+')
yaml.dump(sweep_config, f, allow_unicode=True)