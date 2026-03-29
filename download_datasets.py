# Utility Script to download MNE datasets locally.

import moabb_braindecode_compat  # noqa: F401 — fixes MOABB Windows path sanitization
from moabb_braindecode_compat import fix_moabb_lee2019_session_filter

import mne
# Motor Imagery Datasets
from moabb.datasets import BI2015a, BNCI2014_001, Lee2019_MI, Lee2019_SSVEP
from moabb.datasets import Shin2017A, Chang2025, Yang2025

def download_dataset(dataset):
    print(f"Starting dataset {dataset.code} download...")
    # Force download and cache for all subjects
    for subject in dataset.subject_list:
        try:
            dataset._get_single_subject_data(subject)
            print(f"Subject {subject} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading subject {subject}: {e}")

def download_bnci2014_001():
    dataset = BNCI2014_001()
    return download_dataset(dataset)



def download_lee2019_MI():
    dataset = Lee2019_MI()
    fix_moabb_lee2019_session_filter(dataset)
    return download_dataset(dataset)

def download_lee2019_SSVEP():
    dataset = Lee2019_SSVEP()
    fix_moabb_lee2019_session_filter(dataset)
    return download_dataset(dataset)

def download_BI2015a():
    dataset = BI2015a()
    return download_dataset(dataset)


def download_shin2017a():
    dataset = Shin2017A(accept=True)
    return download_dataset(dataset)

def download_chang2025():
    dataset = Chang2025(paradigm_type="MI")
    return download_dataset(dataset)

def download_yang2025():
    dataset = Yang2025(paradigm_type="2C")
    return download_dataset(dataset)

if __name__ == "__main__":

    # download_bnci2014_001()
    # download_bnci2014_004()
    # download_bnci2014_009()
    # download_lee2019_MI()
    # download_lee2019_SSVEP()
    # download_BI2015a()
    # download_shin2017a()
    download_chang2025()