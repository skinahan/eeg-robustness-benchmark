import os

import pandas as pd
from moabb.evaluations import WithinSessionEvaluation
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from config import MODEL_REGISTRY
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, format_params, unified_cv_training_loop_method
from utils import create_output_path

import time

# NoiseSessionEvaluator extends WithinSessionEvaluation for noise augmentation and perturbation experiments.
#  - WithinSessionEvaluation does not allow dataset-level control.
class NoiseWithinSessionEvaluation(WithinSessionEvaluation):
    def __init__(self, paradigm, datasets, overwrite, hdf5_path, random_state, mode, noise_dict, resample,
                 model_name):
        super().__init__(self, paradigm=paradigm, datasets=datasets, overwrite=overwrite, hdf5_path=hdf5_path,
                       random_state=random_state)
        self.paradigm = paradigm
        self.datasets = datasets
        self.overwrite = overwrite
        self.hdf5_path = hdf5_path
        self.random_state = random_state
        self.mode = mode
        self.prefix = ""
        # Update prefix logic to include new modes
        if self.mode in ['perturb', 'augment', 'perturb_notune', 'augment_notune']:
            self.prefix = 'base_pipeline__'
        self.noise_dict = noise_dict
        self.noise_type = self.noise_dict["noise_type"]
        self.intensity = self.noise_dict["intensity"]
        self.seed = random_state
        self.resample = resample if resample else 250.0
        self.model_name = model_name
        self.model_fn = MODEL_REGISTRY[model_name]
        self.model = None

    def get_wrapped_model_function(self):
        wrapped_model_fn = None

        if self.mode in ['perturb', 'perturb_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                return TrainOnlyNoiseClassifier(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        elif self.mode in ['augment', 'augment_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        return wrapped_model_fn

    def evaluate_without_tuning(self, X, y_encoded, metadata, wrapped_model_fn):
        """Evaluate model performance without hyperparameter tuning using default parameters."""
        results = []
        row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                       'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                       'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                       'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length',
                       'module__lstm_hidden_size'}
        
        self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
        
        for session in metadata["session"].unique():
            session_mask = metadata["session"] == session
            X_mask = X[session_mask]
            y_mask = y_encoded[session_mask]
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
            groups = None
            X_obj = X_mask
            y_obj = y_mask
            
            # Augmentation mode: Supplement clean input data with contaminated samples
            if self.mode in ['augment', 'augment_notune']:
                cv = GroupKFold(n_splits=3, shuffle=True, random_state=self.seed)
                # If doing concatenated data augmentation, we need to track what set (training/validation) each sample belonged to originally
                X_obj, y_obj, groups = self.model.concat_and_augment(X_mask, y_mask)
            
            start_time = time.time()
            roc_auc_score = unified_cv_training_loop_method(self.model, cv, X_obj, y_obj, trial=None, groups=groups)
            end_time = time.time()
            
            result_row = {
                'score': roc_auc_score,
                'time': end_time - start_time,
                'samples': len(X_obj),
                'subject': str(metadata["subject"].iloc[0]) if len(metadata["subject"].unique()) == 1 else "multiple",
                'session': session,
                'channels': X_obj.shape[1],
                'n_sessions': len(metadata["session"].unique()),
                'dataset': metadata["dataset"].iloc[0] if "dataset" in metadata.columns else "unknown",
                'pipeline': f"{self.model_name}+MotorImagery",
                'seed': self.seed,
                'mode': self.mode,
                'model': self.model_name,
                'paradigm': 'MotorImagery',
                'resample': self.resample,
            }
            
            config = self.model.get_params()
            for k, v in config.items():
                if k.startswith(self.prefix):
                    no_prefix = k[len(self.prefix):]
                    if no_prefix in row_headers:
                        result_row[no_prefix] = v
                elif k in row_headers:
                    result_row[k] = v
            
            results.append(result_row)
        
        return pd.DataFrame.from_records(results)

    def process_subj(self, process_dict, dataset, subj):
        X, y, metadata = self.paradigm.get_data(dataset, subjects=[subj])
        y_encoded = LabelEncoder().fit_transform(y)
        results = []
        wrapped_model_fn = self.get_wrapped_model_function()

        for k, v in process_dict.items():
            process_name = k
            paradigm_name = process_name.split("+")[1]
            
            # Check if we should use tuning or not
            if self.mode in ['augment_notune', 'perturb_notune']:
                # Evaluate without tuning using default parameters
                result_df = self.evaluate_without_tuning(X, y_encoded, metadata, wrapped_model_fn)
                results.append(result_df)
            else:
                # Original tuning logic for 'augment' and 'perturb' modes
                out_dir = create_output_path(self.model_name, self.seed, subj, '0train', self.mode)
                output_root = os.path.join(out_dir, f"optuna_results_{self.noise_type}_{self.intensity}")
                best_params, best_score = alternate_two_stage_optuna(model_fn=wrapped_model_fn, model_name=self.model_name, X=X, y=y_encoded,
                                                    metadata=metadata, resample=self.resample, seed=self.seed,
                                                    mode=self.mode, noise_dict=self.noise_dict,
                                                    output_root=output_root, arch_trials=10, train_trials=10)
                final_params = format_params(best_params, self.prefix)
                # Evaluate on 0train and 1test with tuned params.
                row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                               'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                               'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                               'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length',
                               'module__lstm_hidden_size'}
                self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
                for session in metadata["session"].unique():
                    self.model.set_params(**final_params) ## Re-initializes model
                    session_mask = metadata["session"] == session
                    X_mask = X[session_mask]
                    y_mask = y_encoded[session_mask]
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
                    groups = None
                    X_obj = X_mask
                    y_obj = y_mask
                    # Augmentation mode: Supplement clean input data with contaminated samples
                    if self.mode == 'augment':
                        cv = GroupKFold(n_splits=3, shuffle=True, random_state=self.seed)
                        # If doing concatenated data augmentation, we need to track what set (training/validation) each sample belonged to originally
                        X_obj, y_obj, groups = self.model.concat_and_augment(X_mask, y_mask)
                    start_time = time.time()
                    roc_auc_score = unified_cv_training_loop_method(self.model, cv, X_obj, y_obj, trial=None, groups=groups)
                    end_time = time.time()
                    result_row = {
                        'score' : roc_auc_score,
                        'time' : end_time - start_time,
                        'samples' : len(X_obj),
                        'subject': str(subj),
                        'session': session,
                        'channels': X_obj.shape[1],
                        'n_sessions': len(metadata["session"].unique()),
                        'dataset': dataset.code,
                        'pipeline': process_name,
                        'seed': self.seed,
                        'mode': self.mode,
                        'model': self.model_name,
                        'paradigm': paradigm_name,
                        'resample': self.resample,
                    }
                    config = self.model.get_params()
                    for k, v in config.items():
                        if k.startswith(self.prefix):
                            no_prefix = k[len(self.prefix):]
                            if no_prefix in row_headers:
                                result_row[no_prefix] = v
                        elif k in row_headers:
                            result_row[k] = v
                    results.append(result_row)
        
        return pd.concat(results) if results else pd.DataFrame()


    def process(self, process_dict):
        all_results = []
        for dataset in self.datasets:
            subject_list = dataset.subject_list
            for subj in subject_list:
                result_df = self.process_subj(process_dict, dataset, subj)
                all_results.append(result_df)
        return pd.concat(all_results)