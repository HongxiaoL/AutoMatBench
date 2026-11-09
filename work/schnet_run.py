import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
try:
    _intra = int(os.environ.get('TF_NUM_INTRAOP_THREADS', '0') or '0')
    _inter = int(os.environ.get('TF_NUM_INTEROP_THREADS', '0') or '0')
    if _intra > 0:
        tf.config.threading.set_intra_op_parallelism_threads(_intra)
    if _inter > 0:
        tf.config.threading.set_inter_op_parallelism_threads(_inter)
except Exception:
    pass
import logging
from pathlib import Path
from pymatgen.core.structure import Structure
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.Schnet import make_crystal_model
from sklearn.preprocessing import StandardScaler
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
from kgcnn.utils.devices import set_devices_gpu
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
from sklearn.model_selection import train_test_split

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

try:
    from common import params
except ImportError:
    try:
        from work.common import params
    except ImportError:
        import common.params as params

model_name = 'schnet'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')

def setup_logging():
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(logging.FileHandler(log_file))
    logging.info("Initializing SchNet script...")

setup_logging()

callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
}

hyper_1 = {
    "model": {
        "module_name": "kgcnn.literature.Schnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Schnet",
            "inputs": [
                {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64}
            },
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "mean"},
            "depth": 4,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
            "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": None,
        }
    },
    "training": {
        "cross_validation": None,
        "execute_folds": None,
        "fit": {
            "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2, "validation_split": 0.1,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                    "verbose": 0}
                 },
                 {"class_name": "EarlyStopping", "config": {
                    "monitor": "val_loss", "patience": 50, "restore_best_weights": True, "verbose": 1}
                 }
            ]
        },
        "compile": {
            "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
            "loss": "mean_absolute_error"
        },
        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
        "multi_target_indices": None
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
            ]
        },
        "data_unit": ""
    },
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "2.1.0"
    }
}

hyper_2 = {
    "model": {
        "module_name": "kgcnn.literature.Schnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Schnet",
            "inputs": [
                {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64}
            },
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "mean"},
            "depth": 4,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
            "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": None,
        }
    },
    "training": {
        "cross_validation": None,
        "execute_folds": None,
        "fit": {
            "batch_size": 64, "epochs": 600, "validation_freq": 10, "verbose": 2, "validation_split": 0.1,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    "learning_rate_start": 1e-4, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 600,
                    "verbose": 0}
                 },
                 {"class_name": "EarlyStopping", "config": {
                    "monitor": "val_loss", "patience": 50, "restore_best_weights": True, "verbose": 1}
                 }
            ]
        },
        "compile": {
            "optimizer": {"class_name": "Adam", "config": {"lr": 1e-4}},
            "loss": "mean_absolute_error"
        },
        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
        "multi_target_indices": None
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
            ]
        },
        "data_unit": ""
    },
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "2.1.0"
    }
}

hyper_map = {
    "formation_energy": hyper_1,
    "band_gap": hyper_2,
    "metallicity": hyper_2
}

def get_data(ids, all_targets, jarvis_dir, mp_dir, target_col):
    structures = []
    targets = []
    valid_ids = []
    
    for mat_id in ids:
        if mat_id not in all_targets:
            continue
            
        target_val = all_targets[mat_id]
        
        if mat_id.startswith("JVASP-"):
            file_path = os.path.join(jarvis_dir, "jarvis_data", f"{mat_id}.cif")
        elif mat_id.startswith("mp-") or mat_id.startswith("mvc-"):
            file_path = os.path.join(mp_dir, "MP_data", f"{mat_id}.cif")
        else:
            logging.warning(f"Unknown ID format {mat_id}, skipping.")
            continue
            
        if not os.path.exists(file_path):
             continue
             
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
            targets.append(target_val)
            valid_ids.append(mat_id)
        except Exception as e:
            logging.error(f"Error loading {mat_id}: {e}")
            continue
            
    df = pd.DataFrame({
        "structure": structures,
        target_col: targets
    }, index=valid_ids)
    
    return df

def run(task_name, R, N, S):
    if N == 60 and S in [9, 10]:
        return

    param_str = f"({R},{N},{S})"
    folder_name = f"{task_name}_{param_str}_folds"
    data_dir = os.path.join(parent_dir, "work/data", folder_name)
    result_dir = os.path.join(parent_dir, "work/result", model_name + '_' + folder_name.rsplit('_', 1)[0])
    result_file = os.path.join(result_dir, "results.json")

    if not os.path.exists(data_dir):
        logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
        return

    if os.path.exists(result_file):
        logging.info(f"Task {folder_name} already completed. Skipping.")
        return

    logging.info(f"Starting task: {folder_name}")

    try:
        TARGET = task_name
        JARVIS_DIR = os.path.join(parent_dir, "jarvis_data", f"jarvis_{task_name}")
        MP_DIR = os.path.join(parent_dir, "MP_data", f"MP_{task_name}")
        
        os.makedirs(result_dir, exist_ok=True)
        
        logging.info("Loading labels...")
        jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
        if os.path.exists(jarvis_labels_path):
            df_jarvis = pd.read_csv(jarvis_labels_path)
            jarvis_targets = dict(zip(df_jarvis["material_id"], df_jarvis[TARGET]))
        else:
            jarvis_targets = {}

        mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
        if os.path.exists(mp_labels_path):
            df_mp = pd.read_csv(mp_labels_path)
            mp_targets = dict(zip(df_mp["material_id"], df_mp[TARGET]))
        else:
            mp_targets = {}

        all_targets = {**jarvis_targets, **mp_targets}
        logging.info(f"Loaded {len(all_targets)} targets.")

        logging.info("Loading splits...")
        with open(os.path.join(data_dir, "train.json"), "r") as f:
            train_splits = json.load(f)
        with open(os.path.join(data_dir, "test.json"), "r") as f:
            test_splits = json.load(f)
        
        val_path = os.path.join(data_dir, "val.json")
        if os.path.exists(val_path):
            with open(val_path, "r") as f:
                val_splits = json.load(f)
        else:
            logging.warning(f"Validation file {val_path} not found.")
            val_splits = {}

        min_val = -float('inf')
        max_val = float('inf')
        
        if "shear_modulus" in task_name or "bulk_modulus" in task_name:
            min_val = -500.0
            max_val = 1000.0
            logging.info(f"Filtering {task_name} values outside [{min_val}, {max_val}]")

        hyper = hyper_map.get(task_name, hyper_1).copy()
        
        is_classification = task_name == "metallicity"
        if is_classification:
            hyper["training"]["compile"]["loss"] = {"class_name": "BinaryCrossentropy", "config": {"from_logits": True}}
            hyper["training"]["scaler"] = None
        else:
            hyper["training"]["compile"]["loss"] = "mean_absolute_error"
            hyper["training"]["scaler"] = {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.scaler.scaler",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            }

        results = {}
        actual_folds = len(train_splits.keys())
        logging.info(f"Starting {actual_folds}-fold Cross Validation...")

        for fold_idx in range(actual_folds):
            fold_key = str(fold_idx)
            if fold_key not in train_splits or fold_key not in test_splits:
                continue
                
            logging.info(f"Processing Fold {fold_idx}...")
            
            train_ids = train_splits[fold_key]
            test_ids = test_splits[fold_key]
            val_ids = val_splits.get(fold_key, [])
            
            df_train = get_data(train_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET)
            df_test = get_data(test_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET)
            df_val = get_data(val_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET)
            
            df_train = df_train.dropna(subset=[TARGET])
            df_test = df_test.dropna(subset=[TARGET])
            df_val = df_val.dropna(subset=[TARGET])

            # Outlier filtering applies to train and val only.
            if min_val > -float('inf') or max_val < float('inf'):
                len_train_before = len(df_train)
                df_train = df_train[(df_train[TARGET] >= min_val) & (df_train[TARGET] <= max_val)]
                if len(df_train) < len_train_before:
                     logging.info(f"Filtered {len_train_before - len(df_train)} training samples (outliers).")

                if len(df_val) > 0:
                     len_val_before = len(df_val)
                     df_val = df_val[(df_val[TARGET] >= min_val) & (df_val[TARGET] <= max_val)]
                     if len(df_val) < len_val_before:
                         logging.info(f"Filtered {len_val_before - len(df_val)} validation samples (outliers).")
            
            if len(df_train) == 0 or len(df_test) == 0:
                logging.warning("Empty train or test set, skipping fold.")
                continue

            data_train = CrystalDataset()
            data_train._map_callbacks(df_train["structure"].reset_index(drop=True), df_train[TARGET].reset_index(drop=True), callbacks)
            data_train.set_methods(hyper["data"]["dataset"]["methods"])
            data_train.clean(hyper["model"]["config"]["inputs"])
            
            y_train = np.array(data_train.get("graph_labels"))
            x_train = data_train.tensor(hyper["model"]["config"]["inputs"])

            if is_classification:
                scaler = None
            else:
                scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
                y_train = scaler.fit_transform(y_train)

            validation_data = None
            if len(df_val) > 0:
                data_val = CrystalDataset()
                data_val._map_callbacks(df_val["structure"].reset_index(drop=True), df_val[TARGET].reset_index(drop=True), callbacks)
                data_val.set_methods(hyper["data"]["dataset"]["methods"])
                data_val.clean(hyper["model"]["config"]["inputs"])
                
                y_val = np.array(data_val.get("graph_labels"))
                x_val = data_val.tensor(hyper["model"]["config"]["inputs"])
                
                if not is_classification and scaler is not None:
                    y_val = scaler.transform(y_val)
                    
                validation_data = (x_val, y_val)
            else:
                logging.warning("No validation data found for this fold.")

            fit_args = hyper["training"]["fit"]

            model = make_crystal_model(**hyper["model"]["config"])
            
            opt_config = hyper["training"]["compile"]["optimizer"]
            optimizer = tf.keras.optimizers.get(opt_config)
            
            loss_config = hyper["training"]["compile"]["loss"]
            if isinstance(loss_config, dict):
                loss = tf.keras.losses.get(loss_config)
            else:
                loss = loss_config # string
                
            model.compile(loss=loss, optimizer=optimizer)
            
            fit_args = hyper["training"]["fit"]
            callbacks_list = []
            for x in fit_args["callbacks"]:
                if x["class_name"] == "EarlyStopping":
                    callbacks_list.append(tf.keras.callbacks.EarlyStopping(**x["config"]))
                else:
                    callbacks_list.append(tf.keras.utils.deserialize_keras_object(x))
            
            logging.info("Starting training...")
            hist = model.fit(
                x_train, y_train,
                batch_size=fit_args["batch_size"],
                epochs=fit_args["epochs"],
                verbose=fit_args["verbose"],
                callbacks=callbacks_list,
                validation_data=validation_data
            )
            logging.info("Training completed.")

            data_test = CrystalDataset()
            # Keep real targets so they survive clean().
            data_test._map_callbacks(df_test["structure"].reset_index(drop=True), df_test[TARGET].reset_index(drop=True), callbacks)
            data_test.set_methods(hyper["data"]["dataset"]["methods"])
            
            data_test.clean(hyper["model"]["config"]["inputs"]) 
            
            y_true_clean = np.array(data_test.get("graph_labels"))
            x_test = data_test.tensor(hyper["model"]["config"]["inputs"])
            
            predictions_model = model.predict(x_test)
            
            if is_classification:
                def np_sigmoid(x):
                    return np.exp(-np.logaddexp(0, -x))
                predictions = np_sigmoid(predictions_model)
            else:
                predictions = scaler.inverse_transform(predictions_model)
                
            if predictions.shape[-1] == 1:
                predictions = np.squeeze(predictions, axis=-1)
            
            if y_true_clean.shape[-1] == 1:
                 y_true_clean = np.squeeze(y_true_clean, axis=-1)

            if len(predictions) != len(y_true_clean):
                logging.warning(f"Shape mismatch: pred {len(predictions)} vs true {len(y_true_clean)}")
                continue
                
            mae = float(mean_absolute_error(y_true_clean, predictions))
            rmse = float(np.sqrt(mean_squared_error(y_true_clean, predictions)))
            
            try:
                mask = y_true_clean != 0
                if np.sum(mask) > 0:
                    mape = float(mean_absolute_percentage_error(y_true_clean[mask], predictions[mask]))
                else:
                    mape = float(np.nan)
            except:
                mape = float(np.nan)
                
            max_err_val = float(max_error(y_true_clean, predictions))
            
            results[f"fold_{fold_idx}"] = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "max_error": max_err_val,
                "predictions": predictions.tolist(),
                "targets": y_true_clean.tolist()
            }
            logging.info(f"Fold {fold_idx} completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            tf.keras.backend.clear_session()
            del model, data_train, data_test, x_train, x_test
            import gc
            gc.collect()

        if results:
            avg_metrics = {
                "mae": float(np.mean([res["mae"] for res in results.values()])),
                "rmse": float(np.mean([res["rmse"] for res in results.values()])),
                "mape": float(np.nanmean([res["mape"] for res in results.values()])),
                "max_error": float(np.mean([res["max_error"] for res in results.values()]))
            }

            final_output = {
                "task": task_name,
                "method": "SchNet",
                "params": {"R": R, "N": N, "S": S},
                "fold_results": results,
                "average_metrics": avg_metrics
            }

            with open(result_file, "w") as f:
                json.dump(final_output, f, indent=4)

            logging.info(f"Done processing {folder_name}.")
            logging.info(f"Average MAE: {avg_metrics['mae']:.4f}")
        else:
            logging.warning(f"No successful folds for {folder_name}.")

    except Exception as e:
        logging.error(f"Critical error processing task {folder_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser(description='Train SchNet.')
    parser.add_argument("--gpu", required=False, help="GPU index used for training.",
                        default=None, nargs="+", type=int)
    args, unknown = parser.parse_known_args()
    
    # Without --gpu, force CPU to avoid CUDA errors.
    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        set_devices_gpu(args.gpu)

    R_values = params.R_values
    N_values = params.N_values
    S_values = params.S_values
    tasks = params.tasks
    
    for task_name in tasks:
        for R in R_values:
            for N in N_values:
                for S in S_values:
                    run(task_name, R, N, S)
    
    logging.info("All tasks completed.")
