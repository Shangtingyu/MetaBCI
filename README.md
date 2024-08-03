# MetaBCI

## Welcome! 
MetaBCI is an open source, non-invasive brain computer interface platform.The MetaBCI project is led by Prof. Minpeng Xu of Tianjin University, China. This fork of MetaBCI was developed by Hangzhou Mind Matrixes Technology Co. 
The primary contribution of this branch is enabling MetaBCI to support deep learning-based sleep staging scenarios. This allows new users to quickly and easily develop AI-driven sleep staging models using single-channel or multi-channel data on the MetaBCI platform. Specifically, the sleep staging task is divided into three sections: data, algorithm, and result.

* Data Section: Provides data acquisition, preprocessing, and storage functionalities.
* Algorithm Section: Offers single-channel and multi-channel algorithm models.
* Result Section: Supports model training and testing, and displays prediction results.

![img_3.png](.\img.jpg)

All of the above is based on MetaBCI. And we also rewrote and added some basic classes.

## Content!
- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* MetaBCI's datasets are lack of the type of sleep staging
* Lack of Raw datasets decoding and preprocessing
* Lack of deep learning algorithm for sleep stage 
* Lack of demo of combining models and data


If a fresh man wishes to obtain EEG data and labels for sleep staging, he needs to spend a lot of time finding the right dataset, downloading, processing and decoding the data. Then he also needs to find papers, replicate the model and train the model.

### The solution

Now metabci provides a great framework to help us do that!
This fork of MetaBCI will:

* Allow users to download sleep staging data via URL or get it directly locally.
* Add many methods in brainda.datasets for reading and preprocessing data.
* Add many sleep staging algorithms in brainda.algorithms.deep_learning for single-channel and multi-channel data.
* Provide sleep staging demo code.

The goal of this fork of MetaBCI is to enables users to use the MetaBCI platform to conveniently develop AI sleep staging tools based on EEG data, improving the accuracy of sleep staging.

## Features

* Improvements to base datasets
   - add method for preprocessing and save raw dataset
   - add method for read data and label
* Improvements to base deep_learning 
   - add AvgPool1dWithConv method for average pooling of 1d data
   - improve MaxNormConstraintConv2d for data in any dimension

* New Supported Datasets
   - SS Datasets
     - sleep_telemetry: cited by 
        > M. S. Mourtazaev, B. Kemp, A. H. Zwinderman, and H. A. C. Kamphuisen, “Age and Gender Affect Different
    Characteristics of Slow Waves in the Sleep EEG,” Sleep, vol. 18, no. 7, pp. 557–564, Sep. 1995, doi: 10.1093/sleep/18.7.557
     - sleep_cassette: cited by 
        > B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, “Analysis of a sleep-dependent
        neuronal feedback loop: the slow-wave microcontinuity of the EEG,” IEEE Trans. Biomed. Eng.,
        vol. 47, no. 9, pp. 1185–1194, Sep. 2000, doi: 10.1109/10.867928.
     - sleep_shhs: cited by 
        > S. Quan et al., “The Sleep Heart Health Study: design, rationale, and methods,” Sleep,
        vol. 20, no. 12, pp. 1077–1085, Dec. 1997, doi: 10.1093/sleep/20.12.1077.
     - sleep_apples: cited by
        > S. F. Quan et al., “The Association between Obstructive Sleep Apnea and Neurocognitive Performance—The Apnea
        Positive Pressure Long-term Efficacy Study (APPLES),” Sleep, vol. 34, no. 3, pp. 303–314, Mar. 2011,
        doi: 10.1093/sleep/34.3.303.
     - sleep_msp: cited by
        > DiPietro JA, Raghunathan RS, Wu HT, Bai J, Watson H, Sgambati FP, Henderson JL, Pien GW. Fetal heart rate
        during maternal sleep. Dev Psychobiol. 2021 Jul;63(5):945-959. doi: 10.1002/dev.22118. Epub 2021 Mar 25.
        PMID: 33764539.
     - sleep_msro: cited by 
        > T. Blackwell et al., “Associations Between Sleep Architecture and Sleep‐Disordered Breathing and Cognition in
        Older Community‐Dwelling Men: The Osteoporotic Fractures in Men Sleep Study,” J American Geriatrics Society,
        vol. 59, no. 12, pp. 2217–2225, Dec. 2011, doi: 10.1111/j.1532-5415.2011.03731.x.

* New sleep staging algorithms
   - Deep Learning
     - AttnSleepNet: cited by
        > E. Eldele et al., “An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.
     - TinySleepNet: cited by 
        > A. Supratak et al., "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 641-644, doi: 10.1109/EMBC44109.2020.9176741.
     - DeepSleepNet: cited by 
        > A. Supratak et al.,  "DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 11, pp. 1998-2008, Nov. 2017, doi: 10.1109/TNSRE.2017.2721116.

* New demo and methods:
   - run : Demonstration of the overall process of sleep staging
   - predict : Save prediction labels and prediction scores
   - show : show sleep trend graphs and staging percentage pie charts
   - smooth : Smoothing of predicted labels


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Shangtingyu/MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Hangzhou Mind Matrixes Technology Co, China
- University of Bath, England
- Peking University Sixth Hospital, China


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: xingjian.zhang@mindmatrixes.com


## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
