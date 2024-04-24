# Modal Decomposition of LASER beams using Convolutional Neural Networks

This repository contains the code for modal decomposition of LASER beams using Convolutional Neural Networks (CNNs). The research focuses on using deep learning techniques to classify higher-order laser modes, which is crucial for the calibration of gravitational wave detectors like LIGO.

## Introduction

In the field of gravitational wave astrophysics, instruments like the Laser Interferometer for Gravitational-Wave Observations (LIGO) are susceptible to being disrupted by small changes, like wind or seismic vibrations. This research aims to address the challenges related to the classification of laser modes, specifically higher-order modes, using deep learning techniques and a decentralized beam imaging methodology.

## Research Highlights

- Utilization of Convolutional Neural Networks (CNNs) for modal decomposition of LASER beams.
- Focus on classifying higher-order laser modes for gravitational wave detectors like LIGO.
- Development of a comprehensive dataset through data augmentation techniques.
- Investigation of model performance under various conditions including noise levels, mode order, and image offsets.

## Key Components

### 1. Data Preparation

- **Data Generation:** Synthetic data is generated using libraries to produce mode images.
- **Data Labeling:** Filenames contain critical information regarding the mode, signal-to-noise ratio, and image offset from the center.

### 2. Modal Architecture

- **Classification Model:** Inspired by AlexNet, the model utilizes CNNs for mode classification.
- **Model Stacking:** Two models are used, one for predicting the 'm' mode and another for predicting the 'n' mode.

### 3. Evaluation Metrics

- **F1 Score:** The F1 score, derived from the harmonic mean of precision and recall, is used as the evaluation metric for model performance.

## Experiments

### Experiment 1

- **Test 1:** Classification of modes 0, 1, and 2 under low noise conditions.
- **Test 2:** Classification of modes 0, 1, 2, 3, and 4 under low noise conditions.
- **Test 3:** Classification of modes 0, 1, 2, 3, and 4 under high noise conditions.

### Experiment 2

- Comparison of results obtained by the classification model with a state-of-the-art regression model.

## Results

- The classification model shows better precision for lower-order modes.
- Precision of the regression model was below the desired level, especially under high noise conditions.
- The classification model remains effective even when mode images are offset from the center.

## Code

The code for the research is available in this repository. 

## How to Use

1. Clone the repository.
2. Install the required dependencies.
3. Run the scripts for data generation, model training, and evaluation.

## Contributors

- Siddhika Sriram
- Truong X. Tran

## Acknowledgement

The authors express their gratitude to Dr. Anupreeta More, Dr. Shivaraj Kandasamy, and Dr. Suresh Doravari from The Inter-University Center for Astronomy and Astrophysics, Pune, India, for their guidance and support.

## References

[1] H. Wittel et al., “Thermal correction of astigmatism in the gravitational wave observatory GEO 600,” Classical and Quantum Gravity, vol. 31, no. 6, pp. 065008–065008, Feb. 2014, doi: https://doi.org/10.1088/0264- 9381/31/6/065008.

[2] Schiworski, M. G., Brown, D. D., Ottaway, D. J. (2021). Modal decomposition of complex optical fields using convolutional neural networks. JOSA A, 38(11), 1603-1611.

[3] Hofer, L. R., Jones, L. W., Goedert, J. L., Dragone, R. V. (2019). Hermite–Gaussian mode detection via convolution neural networks. JOSA A, 36(6), 936-943.

[4] An, Y., Huang, L., Li, J., Leng, J., Yang, L., Zhou, P. (2019). Learning to decompose the modes in few-mode fibers with deep convolutional neural network. Optics express, 27(7), 10127-10137.

[5] M. Jiang et al., “Fiber laser development enabled by machine 
