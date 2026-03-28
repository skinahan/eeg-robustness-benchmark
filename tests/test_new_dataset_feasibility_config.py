"""Config helpers for Shin2017A / Chang2025 / Yang2025 feasibility wiring (no MOABB data download)."""

from config import classification_num_classes, get_dataset_sampling_rate, get_paradigm


def test_classification_num_classes():
    assert classification_num_classes("Chang2025") == 3
    assert classification_num_classes("Yang2025") == 2
    assert classification_num_classes("Shin2017A") == 2
    assert classification_num_classes("Lee2019_SSVEP") == 4
    assert classification_num_classes("BNCI2014_001") == 2


def test_sampling_rates():
    assert get_dataset_sampling_rate("Shin2017A") == 200.0
    assert get_dataset_sampling_rate("Chang2025") == 1000.0
    assert get_dataset_sampling_rate("Yang2025") == 1000.0


def test_paradigm_epoch_windows():
    p_shin = get_paradigm(dataset="Shin2017A")
    assert p_shin.tmin == 0.0 and p_shin.tmax == 10.0 and p_shin.n_classes == 2

    p_yang = get_paradigm(dataset="Yang2025")
    assert p_yang.tmin == 0.0 and p_yang.tmax == 4.0 and p_yang.n_classes == 2

    p_chang = get_paradigm(dataset="Chang2025")
    assert p_chang.tmin == 0.0 and p_chang.tmax == 4.0 and p_chang.n_classes == 3
    assert set(p_chang.events) == {"left_hand", "right_hand", "both_hands"}
