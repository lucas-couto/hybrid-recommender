import yaml

def get_config():
  conf = yaml.safe_load(open("config.yml"))
  return conf