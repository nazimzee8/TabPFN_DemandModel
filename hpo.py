"""
Hyperparameter optimization using Snowflake ML Tuner + Bayesian Optimization.

Run before train.py to find the best config. Writes best_config.json to
@MODEL_STAGE/hpo/ on completion.

Usage (via run_training_job.py or directly):
    python hpo.py
"""
import json, os, tempfile

from snowflake.ml.modeling.tune import Tuner, TunerConfig
from snowflake.ml.modeling.tune.search import BayesOpt
from snowflake.snowpark import Session

from train import train_fn   # reuse the same training function


def train_for_hpo(dataset_map, hyper_params):
    hyper_params = dict(hyper_params)
    hyper_params["max_epochs"] = "30"   # short runs for HPO signal
    return train_fn(dataset_map, hyper_params)


tuner = Tuner(
    train_fn=train_for_hpo,
    param_space={
        "lr":           (1e-4, 1e-2),          # log-uniform
        "weight_decay": (1e-5, 1e-3),          # log-uniform
        "d_phi":        [64, 128, 256],
        "d_rho":        [128, 256, 512],
        "dropout":      (0.0, 0.3),
        "pool":         ["pna", "attn", "multipool"],
    },
    tuner_config=TunerConfig(
        num_trials=20,
        metric="val_mse",
        mode="min",
        search_alg=BayesOpt(),
    ),
)

results      = tuner.fit()
best_config  = results.get_best_config()
print("Best hyperparameters:", best_config)

# Persist best config to stage for full training run
session = Session.builder.configs({
    "account":  os.environ["SNOWFLAKE_ACCOUNT"],
    "user":     os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "database": "TABPFN_DB", "schema": "TABPFN_SCHEMA",
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
}).create()

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(best_config, f)
    tmp = f.name
session.file.put(tmp, "@MODEL_STAGE/hpo/best_config.json", overwrite=True)
print("Uploaded best_config.json to @MODEL_STAGE/hpo/")
