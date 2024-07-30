# MetaBCI

## Welcome! 
MetaBCI is an open source, non-invasive brain computer interface platform.The MetaBCI project is led by Prof. Minpeng Xu of Tianjin University, China. This fork of MetaBCI is made of Hangzhou Mind Matrixes Technology Co. The branch has 3 main additional contributions:
* datasets: add sleep_edf dataset for import, pre-processing, reading EEG data and decoding algorithms
* deepl_learning: add Attnsleep model for sleep stage with a multiple attention model
* sleep_run: main file for instantiating datasets, deep_learning model, fit and train model, evaluate result

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

* Allow users to download sleep staging data via URL or get it directly locally
* add an abstraction method in brainda.datasets.base for preprocessing and reading
* provide the latest sleep staging algorithm - Attensleep(suitable for one, two or three channels of data)
* Provide sleep staging demo code based on eegnet and Attensleep

The goal of this fork of MetaBCI is to enables users to use the MetaBCI platform to conveniently develop AI sleep staging tools based on EEG data, improving the accuracy of sleep staging.

## Features

* Improvements to base datasets
   - add abstraction method for preprocessing and save raw dataset
   - add abstraction method for read data and label
* Improvements to base deep_learning 
   - add AvgPool1dWithConv method for average pooling of 1d data
   - improve MaxNormConstraintConv2d for data in any dimension

* New Supported Datasets
   - SI Datasets
     - sleep_edf: cited by https://physionet.org/content/sleep-edfx/1.0.0/

* New BCI algorithms
   - Deep Learning
     - AttnSleep: cited by https://github.com/emadeldeen24/AttnSleep

* New demo
   - Attnsleep_run
   - eegnet-run


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

## Paper



## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
