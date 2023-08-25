# Tutorial on Noise Reduction and Signal Reconstruction

## Dataset 

| CONTENT       | Noise Level | N_Samples   | Train Set | Test Set |
| ------------- | ----------- | ----------- | --------- | -------- |
| Simulated SCG | 0           | 5000 / 3000 | &#10004;  | &#10004; |
| Simulated SCG | 0.1         | 5000 / 3000 | &#10004;  | &#10004; |
| Simulated SCG | 0.8         | 5000 / 3000 | &#10004;  | &#10004; |



## Academic Signals Generation

### Base Waves Generation

| Content                 | Impement | Example  |
| :---------------------- | -------- | -------- |
| Sine Waves              | &#10004; | &#10004; |
| Triangle Waves          | &#10004; | &#10004; |
| Square Waves            | &#10004; | &#10004; |
| Chirp Waves             | &#10004; | &#10004; |
| Linear Chirp Waves      | &#10004; | &#10004; |
| Exponential Chirp Waves | &#10004; | &#10004; |
| Hyperbolic Waves        | &#10004; | &#10004; |
| Pulse Waves             | &#10004; | &#10004; |

### Noises

| CONTENT                            | IMPLEMENT | EXAMPLE  |
| ---------------------------------- | --------- | -------- |
| White Noise                        | &#10004;  |          |
| Band-Limited White Noise           | &#10004;  |          |
| Coloured Noise                     | &#10004;  |          |
| Impulsive and Click Noise          | &#10004;  |          |
| ~~Transient Noise Pulses~~         |           |          |
| Thermal Noise                      | &#10004;  |          |
| ~~Shot Noise~~                     |           |          |
| Filcker(1/f) Noise                 | &#10004;  |          |
| Burst Noise                        | &#10004;  |          |
| ~~Natural Sources of Radio Noise~~ |           |          |
| Man-made Sources of Radio Noise    | &#10004;  |          |
| Echo and Multi-path Reflections    | &#10004;  | &#10004; |

## Noise Reduction

### Noise Filters

| TYPE                      | CONTENT                             | IMPLEMENT | EXAMPLE  |
| ------------------------- | ----------------------------------- | --------- | -------- |
| Linear Filters            | Bandpass                            | &#10004;  | &#10004; |
|                           | Bandstop                            | &#10004;  | &#10004; |
|                           | Lowpass                             | &#10004;  | &#10004; |
|                           | Highpass                            | &#10004;  | &#10004; |
|                           | Simple Moving Average               | &#10004;  | &#10004; |
|                           | Exponential Moving Average          | &#10004;  | &#10004; |
| Adaptive Filters          | State-space Kalman Filters          |           |          |
|                           | Extended Kalman Filter (EFK)        |           |          |
|                           | Unscented Kalman Filter (UFK)       |           |          |
|                           | Sample Adaptive Filters (LMS & RLS) | &#10004;  |          |
| Savgol Filters            | Savgol Filters                      |           |          |
| Wiener Filters            | Wiener Filters                      |           |          |
| Matched Filters           | Matched Filters                     |           |          |
| Wavelet denoising         | Wavelet denoising                   | &#10004;  | &#10004; |
| FFT denoising             | FFT denoising                       | &#10004;  | &#10004; |
| PCA-based noise reduction | PCA-based noise reduction           | &#10004;  | &#10004; |

### Decomposition

| CONTENT                           | IMPLEMENT | EXAMPLE  |
| --------------------------------- | --------- | -------- |
| EMD                               | &#10004;  | &#10004; |
| EEMD                              | &#10004;  | &#10004; |
| VMD                               | &#10004;  | &#10004; |
| Seasonal Decompose                | &#10004;  | &#10004; |
| Empirical wavelet transform (EWT) | &#10004;  | &#10004; |
| Singular Spectrum Analysis (SSA)  | &#10004;  | &#10004; |
| PCA based Blind source separation | &#10004;  | &#10008; |
| ICA based Blind source separation | &#10004;  | &#10008; |

## Machine learning for signals

### Clustering

| CONTENT                      | IMPLEMENT | EXAMPLE |
| ---------------------------- | --------- | ------- |
| K-means                      |           |         |
| DBSCAN                       |           |         |
| K-shape                      |           |         |
| Gaussian Mixture Model (GMM) |           |         |
| Spectral Clustering          |           |         |



### Classification

| CONTENT                                        | IMPLEMENT | EXAMPLE |
| ---------------------------------------------- | --------- | ------- |
| KNN                                            |           |         |
| Support Vector Machine (SVM)                   |           |         |
| Decision trees                                 |           |         |
| Random forest                                  |           |         |
| Time-series specific Support Vector Classifier |           |         |

### Regression

| CONTNET                                       | IMPLEMENT | EXAMPLE |
| --------------------------------------------- | --------- | ------- |
| Time-series specific Support Vector Regressor |           |         |



### Semi-supervised approaches

| CONTNET | IMPLEMENT | EXAMPLE |
| ------- | --------- | ------- |
|         |           |         |



### Ensemble methods

| CONTNET                        | IMPLEMENT | EXAMPLE |
| ------------------------------ | --------- | ------- |
| Random Forest (bagging)        |           |         |
| AdaBoost (boosting)            |           |         |
| XGBoost (boosting and bagging) |           |         |





