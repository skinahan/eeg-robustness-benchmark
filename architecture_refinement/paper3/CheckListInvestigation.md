## Compact list: remaining codebase facts we should extract to close checklist gaps

Here are the specific items we should pull (and where they likely live), so we can finish checklist item 4 cleanly and also tee up 6/7/8 later without surprises:

1. **Exact hyperparameter values used in Paper 3 runs (the “locked config”)**

   * `F1`, `D`, `drop_prob`, `kernel_length`
   * `temporal_kernel_size`, `temporal_stride`, `max_seq_length`, `mixed_memory`
   * `lr`, `batch_size`, `weight_decay`, `gradient_clip_value`
   * `DEFAULT_MAX_EPOCHS` actual value (and reconcile with “100 vs 200 epochs” in the paper)

2. **Exact data split protocol used in Paper 3 training**

   * You use `ValidSplit(0.2, stratified=True, random_state=seed)` inside the classifier factory.
   * We need to confirm how that interacts with MOABB cross-session:

     * What is considered the “training dataset” passed to skorch (session train only)?
     * Is the 20% validation split drawn only from the training session?
   * This is a core reproducibility detail.

3. **Loss and label format**

   * You use `CrossEntropyLoss` → implies logits over `n_outputs` with integer class labels.
   * Paper text currently mentions ROC-AUC; we should confirm how ROC-AUC is computed for binary classification (probability for positive class from logits, etc.).

4. **Preprocessing pipeline details actually used (not just “as in Paper 1”)**

   * In Paper 1 supplement, the evaluation modes are described (cross-session leave-one-session-out), and training uses early stopping patience 20, threshold 1e-5, Adam lr 1e-3, max epochs 100. 
   * But we still need: filter band, epoch window, any resampling, channel set, baseline correction, etc. (These weren’t in the snippet we looked at; they may be elsewhere in Paper 1 or the code.)

5. **BNCI2014_001 dataset identity statement**

   * Paper 1 explicitly states BNCI2014_001 is BCI Competition IV-2a MI with 9 subjects, two sessions, and that performance is ROC-AUC. 
   * Good to cite in the dataset/preprocess appendix and the main setup section (you already cite Tangermann).

6. **Early stopping callback exact definition**

   * `get_early_stopping_callback()` likely encodes:

     * monitor key (`valid_loss`?), patience, threshold, whether lower is better, etc.
   * This should be stated explicitly in App.~training (or App.~preprocess/training).

7. **Compute resources logging hooks (for later checklist item 8)**

   * If your pipeline logs wall-clock per run, GPU model, or memory, we should extract where that’s recorded.

---