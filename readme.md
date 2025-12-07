# Convolutional Neural Networks LoRaWAN Device Identification 

This repo contains my implementation and exploration of the models described by Professors Mex-Perera and Bazdresch in their 2023 paper, 'ML-Based RF Fingerprinting for LoRaWAN Device Identification' [1]

The project (and repo) is split into two distinct sections:

    'data collection' which provides the code and gnu radio projects used to collect data to train and test the models.

    'model implementation' which provides the code neccessary to train and test the models one data has been collected. 

This document contains a lot of info and is split into four sections detailing instructions for reproduction, background information, and the results:
    1. Data Collection (how to)
    2. Model Training & Testing (how to)
    3. Background Information
    4. Experiments & Results

If you aren't interested in collecting your own data, you can use the data I collected, which is preorganized in the 'dataset' folder. In this case you can skip the section on data collection.

If you aren't interested in running the code yourself, and just want to see the results, you can skip to the "Experiments & Results" section, and I would also encourage you to read the original paper for a published academic description.

## Data Collection

Data for this project was collected using GNU-Radio (and GNU-Radio Companion) [2].

Hardware used was:
    1. HackRF Portapack Mayhem Edition SDR [3]
    2. 8 Heltec Automation LoRa32 v3 development boards [4]
    3. 2 30db Attenuators, and appropriate SMA Cabling & antennas.

In order to get started, you will first need to install gnu radio:

https://wiki.gnuradio.org/index.php/InstallingGR

Then, install the required dependancy for capturing Lora in GnuRadio [5]:

https://github.com/tapparelj/gr-lora_sdr

You may also need to install drivers for your sdr.

Once you've done that, you can open and use the lora_preamble_collector.grc project.

This project contains a simple pipeline which captures preample samples at the Nyquist sampling rate. 

In this context, the sampling rate is the sampling rate which can capture all frequencies in the given bandwidth, which is also equivilent to the bandwidth. 

In this case, the frame sync block handles this downsampling for you. 

The basic architecture of the project is:
- Incoming samples come from the 'Soapy HackRF Source' block.
- Those samples go into a waterfall gui display (not required but nice to have to verify that things are working)
- They are then filtered and sent into the 'Frame Sync' and demodulation pipeline provided by the gr-lora_sdr library. This will demodulate and collect frame info, then synchronize that and output tagged data. 
- The tagged data will go into the 'Triggered IQ Record Block'. This is a custom python block which interprets the tags and buffers incoming data to record exactly 1024 samples for one preamble (or whatever this comes out to for your Spread Factor, see the background info).

It's important to note that you may need to tweak the input gain on the sdr to appropriate levels depending on context. I used 24 for the wired connection and 16 for the wireless connection.

At this stage, you should be able to see the waterfall display, but no (intentional) signals showing up.

If you would like, you can change the sync word to [0, 0] to capture meshtastic packets and use a meshtastic device to send packets and confirm everything is working. Make sure you also adjust your spread factor, bandwidth, and center frequency to match the meshtastic settings. You can also use the meshtastic_sdr library to decode these packets by enabling the ZMQ PUB Sinc block and using the script in that library.

https://gitlab.com/crankylinuxuser/meshtastic_sdr

info on meshtastic presets: https://meshtastic.org/docs/overview/radio-settings/

Then you can connect the a Heltec board and install the necessary development environment in the Arduino IDE.

Arduino IDE: https://docs.arduino.cc/software/ide-v2/tutorials/getting-started-ide-v2/

Heltec Quick-Start: https://docs.heltec.org/en/node/esp32/esp32_general_docs/quick_start.html

Then, you can upload the 'LoRaSender_heltecv3.ino' script. That will send repeating signals out at the required settings for our system to pick up. Make sure to change the settings back if you changed them to collect meshtastic packets.

You'll need to manually adjust the path name in the IQ Record block to properly store each sample data seperately.

I also included a bash script for randomly selecting a 600/400 split for training and test data sets. If you don't randomize this, it can lead to some overfitting in the training data and poor test accuracy. 

## Model Training & Testing

For this section, I use poetry to manage python related dependancies. You should either get poetry and use that, or you can look in 'pyproject.toml' and ensure you have the required dependancies installed through the package manager of your choice. 

https://github.com/python-poetry/poetry

If you do use poetry, run:`poetry install` to install required dependancies.

At a high level, 
- 'data.py' contains the dataset subclass neccessary to read the IQ samples on the filesystem and import them into a pytorch understandable format. It assumes that the data is organized as '<root>/<data_class>/<device>/<trainortest>' and stored as .cfiles. It will interpret the complex samples as a two channel array of floats, and perform any nessecary downsampling, windowing, etc.
- 'models.py' provides all the available models, built on the implentations from [1].
- 'train.py' provides a usefull helper to abstract the training loop.
- 'test.py' provides the same for testing.
- 'report.py' provides functionality to automatically generate pdf reports for each experiment, along with confusion, loss, and other plots and analysis. These reports will be generated in a subfolder named after the experiment. Typst is required to generate the full pdf, but you can still get everything else, and it will just fail at the very last step and leave the generated plots and analysis in 'results/experement_name/'. https://github.com/typst/typst
- 'experiments.py' provides all the experiments that I performed, pulling in functionality from all of these other modules. 

If you just want to run the experiments I did and reproduce these results, run:
`poetry run python models/experiments.py <experiment_name>`

You can run any number of experiments using multiple arguments to this command. Note that some experiments require a long runtime and they will always run in a set order regardless of the order of parameters.

I've also provided a couple of utilities which are not 'experiments' per-se, but executed in the same manner.

create_labelsets:
- Utility which when run will create and save new randomized labels for the data. it's recommended that you run this once (or use the included labels) and not again between trials.

typst:
- Utility which when enabled will call typst as a subprocess to compile the autogenerated reports. Requires you to have typst installed in '/opt/homebrew/bin/typst'. Change the path in report.py if needed.

The experiments available are as follows:

TODO

## Background Information

LoRa, which stands for 'Long Range', is a propriery radio communication platform and protocol made by semtech [6]. Lora's core promise is to provide long rang, low cost, and low power methods for device-to-device radio communication. Messages are bits (usually serialized data of some kind), and are encoded as signal through 'Chirp Spread Spectrum' [7].

LoRa has several applications, and is predominantly used in Internet of Things applications. It's also used in Meshtastic [8], which is how I personally came to be interested in the protocol. 

The core problem which this area of machine learning research is attempting to address is impersonation & security. Most LoRa systems use preshared keys for authentication, and those keys are stored in non-volatile memory. An attacker with physical access to the device even for a short time could easily obtain these keys, and then install a rogue device into the network [9].

$FIGURE

In order to address this problem, secondary authentication methods are introduced. One such method is so called 'device fingerprinting'. Each LoRa chip & transmission system carries imperfections at the hardware level which can be used as identifiers for the device. Existing methods of radio device fingerprinting involve bespoke system for each context, where experts in radio signal processing will manually extract features of interest [9].

In recent years, there has been a push to attempt to develop an approach using neural networks to classify radio signals. The main advantage of neural networks is that they do not require bespoke implementations, and can do feature extraction without direction. 

In my view, there are three main challenges such an approach must overcome before it's able to be useful in practice.
1. Performance - models must run (and likely train) on low power embedded systems.
2. Location & Environmental Dependance - Radio devices are often mobile and extremely sensitive to location and environmental changes. Even in one device in one location, the temperature of the crystal oscilator can cause CFO drift overtime, which can impact classification accuracy.
3. Closed / Open Set - In order to scale to new circumstances and prevent impersonation attacks, models must be able to generalize to new device sets with minimal retraining, and somehow handle novel devices.

At the time of writing, no literature that I've seen adiquatly addresses all three of these problems, but strides are being made on all three independantly.

Shen et al. (2024) attempts to address the open set problem by using a pretraining/fine-tuning split where the classification layer is stripped from the model after pretraining, and they achieved 90% accuracy when the model was transferred to a new set after fine tuning on a small dataset [10]

Gao et al. (2023) attempts to improve location independance in their models by manually extracting just the CFO and 'Singular Value Decomposition' to integrate that CFO with the existing input data. They were able to achieve 80% accuracy accross different locations and days between test and training datasets.

Both models used in these papers are relatively large, and both use FFT/Spectrogram input with additional preprocessing. 

The focus of the paper I reimplimented is performance improvement, and they do not consider the other two problems.



## Experiments & Results


# References

[1] C. Mex-Perera and M. Bazdresch, “ML-based RF Fingerprinting for LoRaWAN Device Identification,” in Western New York Image Processing Workshop, IEEE, 2023, pp. 1–4. doi: 10.1109/WNYISPW60588.2023.10349416
[2] todo Gnu-Radio
[3] hack-rf
[4] heltec v3
[5] tapparelj
[6] lora
[7] css
[8] meshtastic
[9] A. Ahmed, B. Quoitin, A. Gros, and V. Moeyaert, “A Comprehensive Survey on Deep Learning-Based LoRa Radio Frequency Fingerprinting Identification,” Sensors (Basel, Switzerland), vol. 24, no. 13, p. 4411, 2024, doi: 10.3390/s24134411
[10] G. Shen and J. Zhang, “Exploration of transferable deep learning-aided radio frequency fingerprint identification systems,” Security and Safety, vol. 3, p. 2023019, 2024, doi: 10.1051/sands/2023019
[11] J. Gao, H. Fan, Y. Zhao, and Y. Shi, “Leveraging Deep Learning for IoT Transceiver Identification,” Entropy (Basel, Switzerland), vol. 25,
no. 8, p. 1191, 2023, doi: 10.3390/e25081191
