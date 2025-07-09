import os

from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation

from globals import set_seeds
from models.cnnncp import create_cnnncp_classifier, create_cnnncpv4_classifier
from config import MODEL_REGISTRY
from evaluation.run_experiments import create_output_path, get_paradigm, extract_model_params, two_stage_opt
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, format_params

#v3: CfC
# Inference time: ~0.25 - 0.59 s / epoch

# v4: FastCfC
# Inference time: ~0.25 - 0.87 s / epoch
# Verdict: Terrible performance with standard params

seed=42
set_seeds(seed)
model_name = 'cnn_ncp'
mode='baseline'
is_perturbed = False
noise_type = None
dataset = BNCI2014_001()
subj = 1
dataset.subject_list = [subj]
subject_list = dataset.subject_list

model = create_cnnncp_classifier(n_chans=22, n_times=1000, n_outputs=2)
model.train_split = None
model.max_epochs = 100
model.callbacks = []
paradigm = get_paradigm()
final_params = two_stage_opt(dataset, subj, paradigm, model_name, create_cnnncp_classifier, seed, mode, 250.0)
model.set_params(**final_params)
dataset.subject_list = [subj]
out_dir = "results"
hdf5_path = os.path.join(out_dir,
                         f"{model_name}_{mode}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}.h5")
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
    hdf5_path=hdf5_path,
    random_state=seed
)
results = evaluation.process({f"{model_name}v3+Optuna": model})
df = results.copy()
config = extract_model_params(model)
df['seed'] = seed
df['mode'] = mode
df['model'] = model_name
df['paradigm'] = 'MotorImagery'
df['resample'] = 250.0
df['optimizer__lr'] = config['optimizer__lr']
df['batch_size'] = config['batch_size']
df['max_epochs'] = config['max_epochs']

if model_name == 'cnn_ncp':
    df['module__ncp_hidden_dim'] = config['module__ncp_hidden_dim']
    df['module__sparsity'] = config['module__sparsity']
    df['optimizer__weight_decay'] = config['optimizer__weight_decay']

for subj in df['subject'].unique():
    subject_df = df[df['subject'] == subj]
    for session in df['session'].unique():
        session_df = subject_df[subject_df['session'] == session]
        out_dir = create_output_path(model_name, seed, int(subj), session, mode)
        os.makedirs(out_dir, exist_ok=True)
        filename_suffix = f"_{noise_type}" if is_perturbed and noise_type else ""
        out_file = os.path.join(out_dir,
                                f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
        session_df.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")