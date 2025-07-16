from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from models.cnnncp import create_cnnncp_classifier, create_cnnncfc_classifier
from models.sppncp import create_sppncp_classifier

# Centralized experiment configuration

# Dataset and paradigm defaults
default_subjects = None  # or list(range(1, 10)) for debugging
DEFAULT_DATASET = BNCI2014_001
DEFAULT_PARADIGM = MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8,
        fmax=35,
        tmin=0.0,
        tmax=None,
        baseline=None,
        resample=None,
        n_classes=2
    )
DEFAULT_SEED = 42
# Registered models
def get_model_registry():
    return {
        "eegnet": create_eegnet_classifier,
        "reegnet": create_reegnet_classifier,
        # "cnn_ncp": create_cnnncp_classifier,
        "cnn_ncp": create_cnnncfc_classifier,
        "spp_ncp": create_sppncp_classifier
    }


def get_paradigm(resample=None):
    return MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8, fmax=35,
        tmin=0.0, tmax=None,
        baseline=None,
        resample=resample,
        n_classes=2
    )


MODEL_REGISTRY = get_model_registry()