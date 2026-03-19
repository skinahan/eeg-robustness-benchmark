\twocolumn[

\title{Model Card: HYDRA}

\author{Hierarchical Yield-adaptive Dynamic Recurrent Architecture (HYDRA)}

\textbf{Version}: 1.0  
\textbf{Date}: 2025-01-16

\vskip 0.2in
]

\subsection{Model Overview}

HYDRA is a task-agnostic hybrid convolutional-recurrent EEG decoding architecture combining CNN feature extraction with structured closed-form continuous-time (CfC) recurrence. The architecture processes input $\mathbf{X} \in \mathbb{R}^{C \times T}$ through:

\begin{itemize}
\item \textbf{CNN Feature Extraction}: Maps $\mathbf{X} \in \mathbb{R}^{C \times T}$ to $\mathbf{F} \in \mathbb{R}^{D \times T'}$
\item \textbf{Multi-Scale Temporal Block}: Parallel dilated convolutions with residual connection
\item \textbf{SNR Gating}: Per-channel gain factors $\gamma \in [0,1]$ via $\mathbf{F}_{\text{gated}} = \mathbf{F} \odot \gamma$
\item \textbf{Temporal Binning}: Partitions $\mathbf{F}$ into $B$ overlapping bins $\{\mathbf{F}_1, \dots, \mathbf{F}_B\}$
\item \textbf{Branched CfC Processing}: 
  $$\mathbf{h}_{t+1} = \mathrm{CfC}\!\left(\mathbf{h}_t, \mathbf{x}_t; \theta\right)$$
\item \textbf{Adaptive Carry Gate}:
  $$\mathbf{Z}_b = (1 - \alpha)\,\mathbf{U}_b + \alpha\,\mathbf{V}_b$$
  where $\alpha \in [0,1]$ is learnable, initialized to $0$ (inverse ReZero)
\item \textbf{Hierarchical Pooling}: Intra-bin $\mathbf{r}_b = \mathrm{Pool}(\mathbf{Z}_b)$, inter-bin $\mathbf{z} = \mathrm{Pool}(\{\mathbf{r}_b\}_{b=1}^B)$
\item \textbf{Classification}: $\hat{\mathbf{y}} = \mathbf{W}\,\mathrm{Dropout}(\mathrm{LN}(\mathbf{z})) + \mathbf{b}$
\end{itemize}

\subsection{Validation Datasets}

\begin{table}[h]
\centering
\small
\begin{tabular}{lcccc}
\toprule
Dataset & Paradigm & Subjects & Sessions & Channels \\
\midrule
BNCI2014\_001 & Motor Imagery & 9 & 2 & 22 \\
Lee2019\_SSVEP & SSVEP & 54 & 2 & 62 \\
BI2015a & ERP/P300 & 43 & 3 & 32 \\
\bottomrule
\end{tabular}
\caption{Dataset characteristics. Trials per subject: BNCI2014\_001 (576), Lee2019\_SSVEP (200), BI2015a (8,280).}
\end{table}

\paragraph{Evaluation Regimes} Within-session (5-fold CV), cross-session, cross-subject (3-fold CV)

\subsection{Architecture Hyperparameters}

\paragraph{CNN Front-end} $F_1=8$, $F_2=16$ ($D=2$), kernel length $=125$, pool $=4$, dropout $p=0.25$

\paragraph{Multi-Scale Block} Kernels $(9, 15, 31)$, dilations $(1, 4, 16)$

\paragraph{Binning} $L=48$, stride $=44$, overlap $o \approx 0.08$

\paragraph{CfC} Mixed memory enabled, LeCun tanh activation, hidden dimension $H=43$ (Architecture \#4)

\paragraph{Adaptive Carry Gate} $\alpha$ initialized to $0$, learnable scalar

\subsection{Training Configuration}

\paragraph{Optimizer} AdamW, learning rate $=10^{-2}$ (optimized: $[10^{-6}, 10^{-2}]$), weight decay $=0$ (optimized: $[10^{-6}, 10^{-2}]$)

\paragraph{Training} Batch size $=64$ (optimized: $\{4, 8, 16, 32, 64\}$), max epochs $=300$, early stopping patience $=5$, gradient clipping $=1.0$ (L2 norm), ExponentialLR scheduler ($\gamma=0.97$)

\paragraph{Hyperparameter Optimization} Two-stage Optuna (20-40 trials per stage), objective: maximize validation ROC-AUC

\subsection{Model Complexity}

\paragraph{Parameters} $\sim$4,000-15,000 (depends on wiring configuration)

\paragraph{Component Breakdown}
\begin{itemize}
\item CNN front-end: $\sim$1,400
\item Multi-scale block: $\sim$1,680
\item SNR gate: $\sim$192
\item Temporal downsampler: $\sim$768
\item Recurrent (Architecture \#4): Variable
\item Attention mechanisms: $\sim$500-1,000
\item Classification head: $\sim$64
\end{itemize}

\paragraph{Computational Complexity} Scales linearly with bins $B$, hidden dimension $H$, and binned sequence length $L$. CfC updates are constant-time per step.

\subsection{Performance Summary}

\paragraph{Motor Imagery (BNCI2014\_001, Cross-Session)}
\begin{itemize}
\item Mean ROC-AUC: $0.95 \pm 0.01$ (range: $0.88$-$0.98$)
\item Mean accuracy: $85$-$90\%$
\end{itemize}

\paragraph{Robustness (Motor Imagery)}
\begin{itemize}
\item Clean ROC-AUC: $0.9512 \pm 0.0146$
\item Corrupted ROC-AUC: $0.8006 \pm 0.1392$
\item Retention: $84.18\% \pm 14.66\%$
\item By perturbation type: EOG $93.01\%$, Gaussian $89.66\%$, channel dropout $69.87\%$
\end{itemize}

\paragraph{Paradigm-Specific}
\begin{itemize}
\item \textbf{MI}: Robustness exceeds prior recurrent decoders, comparable to CNN baselines
\item \textbf{SSVEP}: CNN models (EEGNet) outperform; HYDRA competitive under cross-subject evaluation
\item \textbf{ERP/P300}: Validated across sessions with consistent robustness
\end{itemize}

\subsection{Limitations}

\paragraph{Scope} Not intended for clinical diagnosis, medical decision-making, or real-time applications without validation

\paragraph{Generalization} Performance varies across subjects; may require subject-specific tuning. Cross-dataset deployment requires domain adaptation.

\paragraph{Architecture} Fixed input dimensions; hyperparameter sensitivity requires careful tuning

\paragraph{Robustness} Performance degrades under high-intensity noise ($>100\%$ Gaussian) or severe channel dropout ($>50\%$ channels)

\subsection{Reproducibility}

\paragraph{Random Seeds} Fixed seeds for all operations, CUDA deterministic mode enabled

\paragraph{Dependencies} PyTorch ($\geq$1.9.0), braindecode, MOABB, ncps, scikit-learn, numpy

\paragraph{Model Artifacts} Architecture configuration (JSON), hyperparameters, training code (version-controlled). Trained weights and raw data not shared due to licensing/privacy.

\subsection{Citation}

\begin{verbatim}
@article{hydra2025,
  title={HYDRA: Robust EEG Decoding with Closed-Form 
         Continuous-Time Recurrent Networks},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
\end{verbatim}

\vskip 0.1in

\textbf{Document Version}: 1.0  
\textbf{Last Updated}: 2025-01-16
