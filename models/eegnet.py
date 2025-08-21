import torch
from braindecode.models import EEGNetv4
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from globals import get_seed


def create_eegnet_classifier(n_chans, n_times, n_outputs, seed=get_seed()):
    """
    Return a configured EEGClassifier using EEGNetv4.

    Parameters:
    - n_chans: int
    - n_times: int
    - n_outputs: int
    - device: 'cuda' or 'cpu'
    - seed: int, random seed for reproducibility

    Returns:
    - EEGClassifier instance
    """
    return EEGClassifier(
        EEGNetv4,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        batch_size=64,
        max_epochs=100,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__final_conv_length='auto',
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EarlyStopping(patience=10, load_best=True)],
        # verbose=0
    )
