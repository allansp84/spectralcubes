## Face Spoofing Detection Through Visual Codebooks of Spectral Temporal Cubes

**SpectralTemporalCubes** is an open source python API for face spoofing detection. This is an implementation of our work published in *IEEE Transactions on Image Processing*, whose reference to the original manuscript and its BibTeX are available in this document.

### Requirements to Run this Software

The following software is required to run SpectralTemporalCubes:

1. [OpenCV-2.4.13](http://opencv.org/);
2. [joblib-0.8.4](https://pythonhosted.org/joblib/)
3. [Pillow-2.5.3](https://python-pillow.org/)
4. [Numpy-1.8.1](http://www.numpy.org/)
5. [Matplotlib-1.5.1](http://matplotlib.org/)
6. [Scikit-image-0.10.1](http://scikit-image.org/)
7. [Scikit-learn-0.15.0](http://scikit-learn.org/)
8. [Scipy-0.13.3](https://www.scipy.org/)
9. [Bob Package](https://www.idiap.ch/software/bob/)

Except for the OpenCV, these software products are python packages that can be easily installed using pip. We provide a script *(install_requirements.sh)* to install these python requirements via **pip command**. SpectralTemporalCubes runs on Linux Operating systems, and we tested it on Ubuntu 14.04 LTS.

#### Installing our software and its dependencies

We recommend using the virtualenv tool to create an appropriate environment to install and run our software. Follows the command lines to do this:

1. Installing the virtualenv and virtualenvwrapper tools:
>     $ sudo apt-get install python-pip
>     $ pip install virtualenv
>     $ pip install virtualenvwrapper
*Obs.: It is necessary to add the command to source /usr/local/bin/virtualenvwrapper.sh to your shell startup file. Please, change the path to virtualenvwrapper.sh depending on where it was installed by pip. You can find more information about virtualenv and virtualenvwrapper in https://virtualenv.pypa.io/en/stable/ and https://virtualenv.pypa.io/en/stable/, respectively.*

2. Installing all depedences of our sofwware:
>     $ mkvirtualenv spectralcubes-env                      # -- create a new environment
>     $ workon spectralcubes-env                            # -- active the virtualenv
>     (spectralcubes-env)$ ./install_requirements.sh        # -- run the bash script provided in this repository (https://github.com/allansp84/spectralcubes/blob/master/install_requirements.sh)

3. Installing our software:
>     (spectralcubes-env)$ python setup.py install          # -- install our software in your machine


### How to Use this Software?

After installing our software, we can use it via command line interfaces (CLIs).  To see how to use this software, execute the following command in any directory, since it is already installed in your system:

>     (spectralcubes-env)$ spectralcubesantispoofing.py --help

### Examples

1. Run this software to reproduce the results from Replay-Attack dataset:
>     
>     (spectralcubes-env)$ spectralcubesantispoofing.py --dataset 0 --dataset_path datasets/replayattack --output_path ./working --protocol intra_dataset
>     

1. Run this software to reproduce the results from CASIA FASD dataset:
>     
>     (spectralcubes-env)$ spectralcubesantispoofing.py --dataset 1 --dataset_path datasets/casia --output_path ./working --protocol intra_dataset
>     

1. Run this software to reproduce the results from 3DMAD dataset:
>     
>     (spectralcubes-env)$ spectralcubesantispoofing.py --dataset 2 --dataset_path datasets/3dmad --output_path ./working --protocol intra_dataset
>     

1. Run this software to reproduce the results from UVAD dataset:
>     
>     (spectralcubes-env)$ spectralcubesantispoofing.py --dataset 3 --dataset_path datasets/uvad/release_1 --output ./working --protocol intra_dataset
>     

### Reference

If you use this software, please cite our paper published in *IEEE Transactions on Image Processing*:

> **Plain Text**
>
>     A. Pinto, H. Pedrini, W. R. Schwartz and A. Rocha, "Face Spoofing Detection Through Visual Codebooks of Spectral Temporal Cubes," in *IEEE Transactions on Image Processing*, vol. 24, no. 12, pp. 4726-4740, Dec. 2015.
>     doi: 10.1109/TIP.2015.2466088
>     Abstract: Despite important recent advances, the vulnerability of biometric systems to spoofing attacks is still an open problem. Spoof attacks occur when impostor users present synthetic biometric samples of a valid user to the biometric system seeking to deceive it. Considering the case of face biometrics, a spoofing attack consists in presenting a fake sample (e.g., photograph, digital video, or even a 3D mask) to the acquisition sensor with the facial information of a valid user. In this paper, we introduce a low cost and software-based method for detecting spoofing attempts in face recognition systems. Our hypothesis is that during acquisition, there will be inevitable artifacts left behind in the recaptured biometric samples allowing us to create a discriminative signature of the video generated by the biometric sensor. To characterize these artifacts, we extract time-spectral feature descriptors from the video, which can be understood as a low-level feature descriptor that gathers temporal and spectral information across the biometric sample and use the visual codebook concept to find mid-level feature descriptors computed from the low-level ones. Such descriptors are more robust for detecting several kinds of attacks than the low-level ones. The experimental results show the effectiveness of the proposed method for detecting different types of attacks in a variety of scenarios and data sets, including photos, videos, and 3D masks.
>     keywords: {face recognition;feature extraction;image coding;image sensors;spatiotemporal phenomena;acquisition sensor;biometric system vulnerability;discriminative video signature;face biometrics;face recognition systems;face spoofing detection;low-level feature descriptor;recaptured biometric samples;software-based method;spectral information;spectral temporal cubes;spoofing attack;synthetic biometric samples;temporal information;time-spectral feature descriptor extraction;visual codebook concept;visual codebooks;Face;Face recognition;Feature extraction;Noise;Three-dimensional displays;Visualization;Face spoofing attack detection;face biometric system;mobile device;mobile device, face biometric system;spectral analysis;time-spectral visual features;timespectral visual features;visual codebook},
>     URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7185398&isnumber=7271151


> **BibTeX**
>
>     @ARTICLE{7185398,
>     author={Pinto, A. and Pedrini, H. and Robson Schwartz, W. and Rocha, A.},
>     journal={Image Processing, IEEE Transactions on},
>     title={Face Spoofing Detection Through Visual Codebooks of Spectral Temporal Cubes},
>     year={2015},
>     volume={24},
>     number={12},
>     pages={4726-4740},
>     abstract={Despite important recent advances, the vulnerability of biometric systems to spoofing attacks is still an open problem. Spoof attacks occur when impostor users present synthetic biometric samples of a valid user to the biometric system seeking to deceive it. Considering the case of face biometrics, a spoofing attack consists in presenting a fake sample (e.g., photograph, digital video, or even a 3D mask) to the acquisition sensor with the facial information of a valid user. In this paper, we introduce a low cost and software-based method for detecting spoofing attempts in face recognition systems. Our hypothesis is that during acquisition, there will be inevitable artifacts left behind in the recaptured biometric samples allowing us to create a discriminative signature of the video generated by the biometric sensor. To characterize these artifacts, we extract time-spectral feature descriptors from the video, which can be understood as a low-level feature descriptor that gathers temporal and spectral information across the biometric sample and use the visual codebook concept to find mid-level feature descriptors computed from the low-level ones. Such descriptors are more robust for detecting several kinds of attacks than the low-level ones. The experimental results show the effectiveness of the proposed method for detecting different types of attacks in a variety of scenarios and data sets, including photos, videos, and 3D masks.},
>     keywords={face recognition;feature extraction;image coding;image sensors;spatiotemporal phenomena;acquisition sensor;biometric system vulnerability;discriminative video signature;face biometrics;face recognition systems;face spoofing detection;low-level feature descriptor;recaptured biometric samples;software-based method;spectral information;spectral temporal cubes;spoofing attack;synthetic biometric samples;temporal information;time-spectral feature descriptor extraction;visual codebook concept;visual codebooks;Face;Face recognition;Feature extraction;Noise;Three-dimensional displays;Visualization;Face spoofing attack detection;face biometric system;mobile device;mobile device, face biometric system;spectral analysis;time-spectral visual features;timespectral visual features;visual codebook},
>     doi={10.1109/TIP.2015.2466088},
>     ISSN={1057-7149},
>     month={Dec},}


### License

This software is available under condition of the [AGPL-3.0 Licence](https://github.com/allansp84/spectralcubes/blob/master/LICENSE).

Copyright (c) 2015, Allan Pinto, William Robson Schwartz, Helio Pedrini, and Anderson Rocha
