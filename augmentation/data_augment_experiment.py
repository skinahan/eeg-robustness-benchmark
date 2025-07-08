import numpy as np
from moabb.datasets import BNCI2014_001
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from augmentation.noise import EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from config import MODEL_REGISTRY
from evaluation.run_experiments import get_paradigm
from evaluation.two_stage_hp_opt import run_two_stage_optuna


def run_grouped_augmented_experiment(model_name, subject_list, seed, resample, noise_type, intensity):
    model_fn = MODEL_REGISTRY[model_name]
    base_model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
    augmenter = EEGNoiseAugmentor(noise_type=noise_type, intensity=intensity, seed=seed)
    concat_aug = ConcatenatedNoiseAugmenter(augmenter)

    for subj in subject_list:
        dataset = BNCI2014_001()
        dataset.subject_list = [subj]
        paradigm = get_paradigm(resample=resample)

        X, y, metadata = paradigm.get_data(dataset, subjects=[subj])
        y_encoded = LabelEncoder().fit_transform(y)

        X_aug, y_aug, groups = concat_aug.transform_with_groups(X, y_encoded)

        cv = GroupKFold(n_splits=3)
        fold_scores = []
        for train_idx, val_idx in cv.split(X_aug, y_aug, groups):
            model = clone(base_model)
            model.max_epochs = 100
            model.train_split = None
            model.callbacks = []
            model.fit(X_aug[train_idx], y_aug[train_idx])
            fold_scores.append(model.score(X_aug[val_idx], y_aug[val_idx]))

        print(f"Subject {subj}, grouped CV accuracy: {np.mean(fold_scores):.4f}")
