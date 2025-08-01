import os


def create_hdf5_model_path(model, seed, session, mode, session_type='WithinSessionEvaluation', paradigm='MotorImagery', others=[]):
    full_list = [
        "results",
        paradigm,
        "BNCI2014_001",
        model,
        session_type,
        str(seed),
        f"checkpoints",
        session,
        mode
    ]
    if len(others) > 0:
        full_list.extend(others)

    return os.path.join(
        "//".join(full_list)
    )

def  create_output_path(model, seed, subject, session, mode, session_type='WithinSessionEvaluation', paradigm='MotorImagery', others=[]):
    full_list = [
        "results",
        paradigm,
        "BNCI2014_001",
        model,
        session_type,
        str(seed),
        f"sub-{int(subject):03d}",
        session,
        mode
    ]
    if len(others) > 0:
        full_list.extend(others)

    return os.path.join(
        "//".join(full_list)
    )