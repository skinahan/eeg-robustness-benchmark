# Utility Script to download MNE datasets locally.

import mne
# Motor Imagery Datasets
from moabb.datasets import BI2015a, BNCI2014_001, Lee2019_SSVEP

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
def download_lee2019_SSVEP():
    dataset = Lee2019_SSVEP()
    return download_dataset(dataset)

def download_BI2015a():
    dataset = BI2015a()
    return download_dataset(dataset)

if __name__ == "__main__":

    # download_bnci2014_001()
    # download_bnci2014_004()
    # download_bnci2014_009()
    download_lee2019_SSVEP()