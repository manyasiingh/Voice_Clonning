Dev-selected checkpoints (selected_epoch 0 = initial model kept).

| model | n_seeds | EER_mean_% | EER_std_% | minDCF_mean | minDCF_std | EER_speaker_CI95_%_per_run | selected_epoch_per_run | dev_EER_%_start->end_per_run | n_test_speakers | n_trials | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ablation_K2 | 1 | 1.71 | - | 0.122 | - | 0.81-2.83 | 0 | 1.74->1.90 | 40 | 179700 | ablation_K2_s0 |
| ablation_T0.05 | 1 | 1.71 | - | 0.122 | - | 0.81-2.83 | 0 | 1.74->3.64 | 40 | 179700 | ablation_T0.05_s0 |
| ablation_T0.2 | 1 | 1.71 | - | 0.122 | - | 0.81-2.83 | 0 | 1.74->2.79 | 40 | 179700 | ablation_T0.2_s0 |
| finetune | 3 | 1.71 | 0.00 | 0.122 | 0.000 | 0.81-2.83; 0.81-2.83; 0.81-2.83 | 0,0,0 | 1.74->1.88; 1.74->1.91; 1.74->1.88 | 40 | 179700 | finetune_s0,finetune_s1,finetune_s2 |
| pretrained | 1 | 1.71 | - | 0.122 | - | 0.81-2.83 | - | - | 40 | 179700 | pretrained |
| scratch | 3 | 5.06 | 0.20 | 0.443 | 0.042 | 3.17-7.10; 3.40-7.57; 3.14-7.31 | 10,10,10 | 29.31->8.07; 29.71->8.17; 28.90->7.90 | 40 | 179700 | scratch_s0,scratch_s1,scratch_s2 |
