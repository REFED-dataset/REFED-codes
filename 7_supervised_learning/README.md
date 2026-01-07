## Supervised learning example on the REFED dataset
This example demonstrates how to perform supervised learning using the REFED dataset. The REFED dataset contains labeled data suitable for training and evaluating machine learning models.
- Dataset: REFED (available at: https://refed-dataset.github.io/)
- Task: Supervised Learning (classification, regression)
- Data split: Leave-One-Trial-Out (LOTO), 10% validation set from training data
- Model: TSMMF (Si et al., 2025) [[link]](https://github.com/ThreePoundUniverse/TSMMF-ESWA)
- Framework: PyTorch
### Instructions to run the example
- Download the REFED dataset. The default path is `./REFED-dataset` (see `./utils/utils_args.py`).
- Train and evaluate the model:
```bash
# Regression on valence
python 7.1_run_TSMMF.py --label_dim valence --label_mode 0-1 --final_activation sigmoid
# Regression on arousal
python 7.1_run_TSMMF.py --label_dim arousal --label_mode 0-1 --final_activation sigmoid
# 3-class classification on valence
python 7.1_run_TSMMF.py --label_dim valence --label_mode 3c --final_activation None
# 3-class classification on arousal
python 7.1_run_TSMMF.py --label_dim arousal --label_mode 3c --final_activation None
```
- To draw regression results, run the notebook `7.2_draw_regression_results.ipynb` after training.
