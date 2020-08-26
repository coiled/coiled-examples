import coiled


software_environments = {
    "xgboost": {"conda": "scaling-xgboost/environment.yml"},
    "pangeo": {"conda": "pangeo/environment.yml"},
    "pytorch": {"conda": "hyper-parameter-optimmization/environment.yml"},
}

cluster_configurations = {
    "xgboost": {"worker_cpu": 4, "worker_memory": "8 GiB"},
    "pangeo": {"worker_cpu": 4, "worker_memory": "8 GiB"},
    "pytorch": {"worker_cpu": 4, "worker_memory": "16 GiB"},
}

assert software_environments.keys() == cluster_configurations.keys()

for name, spec in software_environments.items():
    full_name = f"coiled-examples/{name}"
    coiled.create_software_environment(name=full_name, **spec)
    coiled.create_cluster_configuration(name=full_name, **cluster_configurations[name])
