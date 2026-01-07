# Demo Codes for the REFED Dataset

Demo codes and notebooks for the REFED dataset. You can learn how to use the REFED dataset through this example. To access the dataset, please visit https://refed-dataset.github.io/.

>  *REFED: A Subject Real-time Dynamic Labeled EEG-fNIRS Synchronized Recorded Emotion Dataset, NeurIPS 2025*

## About REFED

**Affective brain-computer interfaces (aBCIs)** play a crucial role in personalized human–computer interaction and neurofeedback modulation. We present the **Real-time labeled EEG-fNIRS Emotion Dataset (REFED)**, which simultaneously records brain signals from both EEG and fNIRS modalities while providing continuous, real-time annotations of valence and arousal. The results of the data analysis demonstrate the effectiveness of emotion inducement and the reliability of real-time annotation. This dataset offers the first possibility for studying the neural-vascular coupling mechanism under emotional evolution and for developing dynamic, robust affective BCIs.

## Code content

1. `process_sample`: Process the EEG/fNIRS data as samples based on time windows.
2. `feature_EEG/fNIRS`: Extract traditional features from the raw EEG/fNIRS data. 
3. `load_data`: How to load the EEG/fNIRS data and emotion labels from REFED dataset.
4. `label_figure`: Draw the violin plot of the emotional scores.
5. `label_distribution`: Draw the trajectory and distribution of dynamic emotion labels.
6. `EEG/fNIRS_topo`: Draw brain topographic maps among different emotional states.
7. `supervised_learning`: An example of LOTO (leave-one-trial-out) supervised learning using the TSMMF model on the REFED dataset.

- `load_REFED.py`: The Python tool for loading the REFED datasets.
- `ch_info_EEG/fNIRS.pkl`: The channel name list and position list (for mne) for EEG and fNIRS data.

## Quickly load the REFED dataset

Using Python script `load_REFED.py` to load data and label as a 'dict' type:

```python
from load_REFED import load_data, load_label
data  = load_data ("REFED-dataset/data", sub_list=['1','2'], modality=['EEG','fNIRS'])
label = load_label("REFED-dataset/annotations", dimension=['Valence','Arousal'])
```

Verified environment
-----------------

- Python 3.10
- mne 1.7.1
- pandas 1.5.3
- numpy 1.26.4
- scipy 1.14.1
- matplotlib 3.8.3
- seaborn 0.13.2
- PyTorch 2.1.2 + CUDA 12.4

## Citation

```
@inproceedings{NEURIPS2025_REFED,
  title = {REFED: A Subject Real-time Dynamic Labeled EEG-fNIRS Synchronized Recorded Emotion Dataset},
  author = {Ning, Xiaojun and Wang, Jing and Feng, Zhiyang and Xin, Tianzuo and Zhang, Shuo and Zhang, Shaoqi and Lian, Zheng and Ding, Yi and Lin, Youfang and Jia, Ziyu},
  booktitle = {The Thirty-Ninth Annual Conference on Neural Information Processing Systms},
  year = {2025}
}
```

## License

The REFED dataset and code are made available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0) International License. Detailed license terms can be found at [https://creativecommons.org/licenses/by-nc-sa/4.0/ ](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![CC-BY-NC-SA 4.0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)





