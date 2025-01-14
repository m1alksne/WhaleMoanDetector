# WhaleMoanDetector: A fine-tuned faster-rCNN for detecting and classifying blue and fin whale moans in audio data

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

![Fin whale](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/blue_whale.jpeg)
Blue whales off the coast of San Diego.
Photo credit: Manuel Mendieta

## Overview 

The WhaleMoanDetector repository contains Python code required to train, test, and run inference using WhaleMoanDetector, a fine-tuned faster-rCNN model for blue and fin whale calls, on audio files. A brief tutorial highlighting the functions needed to deploy the trained model is included below. The workflow is meant to be straightforward for bioacousticains who are new to Python. 

![spectrogram](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/all_example.JPG)
WhaleMoanDetector is trained to identify blue whale A, B, and D calls, and fin whale 20 Hz and 40 Hz calls in 60 second windows. 


## WhaleMoanDetector Directory Structure: 

```
WhaleMoanDetector/
├── LICENSE
├── README.md          <- The top-level README for users.
├── code               <- All code required to run detector
│   ├── AudioDetectionDataset.py     <- Dataset class.
│   ├── AudioStreamDescriptor.py     <- Parses xwav and wav headers.
│   ├── call_context_filter.py       <- Context filters predictions based on duration and frequency.
│   ├── config.yaml                  <- User defined variables for running inference.
│   ├── custom_collate.py            <- Collate function.
│   ├── extract_wav_header.py	     <- Function to extract xwav and wav headers.
│   ├── freq_bbox.py		     <- Converts bounding boxes to frequency (Hz).
│   ├── inference_functions.py       <- Necessary functions to make audio file chunks and spectrograms, run inference,
│   │						and save out prediction spreadsheet and images
│   ├── inference_pipeline.py	     <- Wrapper to run inference_functions and modify file paths.
│   ├── make_annotations.py          <- Wrapper to make_spectrograms.py for training/validation/testing examples
│   ├── make_spectrograms.py         <- Necessary functions to make audio file chunks, spectrograms, and labels for
│   │						training, testing, and validation.
│   ├── modify_timestamp.py  	     <- Convert Triton LoggerPro files to "modified annotation" format with 
│   │						start and end times as # of seconds into start of wav file.
│   ├── modify_timestamp_function.py <- Necessary function for modify_timestamp.py.
│   ├── plot_groundtruth.py          <- Plots groundtruth annotations in Spyder software.
│   ├── PR_curve.py      	     <- Generates PR curve for test data.
│   ├── plot_predictions.py          <- Visualize WhaleMoanDetector predictions in Spyder.
│   ├── time_bbox.py                 <- Converts bounding boxes to time (sec into file).
│   ├── train.py                     <- Train WhaleMoanDetector.
│   ├── train_val_test_split_all.py  <- Make train/val/test splits using outputs of make_annotations.py.
│   ├── validation.py                <- Run validation script
│
├── figures               <- Graphics and figures to be used in reporting
├── labeled_data
│   │	└── CalCOFI
│   │	    └── modified_annotations <- Modified start and end time .csv files 
│   │	└── HARP
│   │	    └── modified_annotations <-Modified start and end time .csv files 
│   │
│   ├── spectrograms   <- The final spectrogram images used for model training and testing (not uploaded to GitHub)
│   │	└── CalCOFI 
│   │	└── HARP
│   └── wav           <- The original wav files (not uploaded to GitHub)
│
├── models             <- Trained models (not uploaded to GitHub)
   
```
## For new Python users:

1. [Download Python 3.9](https://www.python.org/downloads/release/python-390/)

2. [Download Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

3. [Miniconda tutorial](https://docs.anaconda.com/working-with-conda/environments/)
## WhaleMoanDetector Workflow

1. Clone the Repository:

    ```bash
    git clone https://github.com/m1alksne/WhaleMoanDetector.git
    cd WhaleMoanDetector
    ```

2. Create a Virtual Environment:

    ```bash
    conda create -n whalemoandetector python=3.9
    conda activate whalemoandetector
    ```

3. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run WhaleMoanDetector on wav files:

   a. Modify File Paths in the Configuration File:
      
	Example of `config.yaml`:

      ```yaml
      wav_directory: 'L:/CHNMS_audio/CHNMS_NO_01/CHNMS_NO_01_disk01_df100'
      txt_file_path: 'M:/Mysticetes/WhaleMoanDetector_outputs/CHNMS_NO_01/CHNMS_NO_01_raw_detections.txt'
      model_path: 'L:/WhaleMoanDetector/models/WhaleMoanDetector.pth'
      ```

    b. Exectute inference_pipeline.py

    ```bash
    python code/inference_pipeline.py
    ```

	Executing the inference pipeline will deploy WhaleMoanDetector on the directory of wav files given in ```wav_directory```. The ```inference_pipeline.py``` script preprocesses the audio data by generating 60 second non-overlapping audio segments. For each audio segment, a spectrogram is made using the [librosa](https://librosa.org/doc/latest/index.html) package with a hamming window and window length of 1 second with 90% overlap between consecutive frames. Spectrograms are truncated between 10-150 Hz and normalized individually between 0 and 1 for consistent input scaling:

![spectrogram](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/SOCAL_H_65_spectrogram.JPG)

WhaleMoanDetector will make predictions on each spectrogram and predictions are written to a .txt file stored in ```txt_file_path```. PNG images of spectrograms containing predictions will also be saved to the ```txt_file_path``` directory. An example of the ```txt_file_path``` directory is included below: 

![example](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/example_directory_structure.JPG)

5. Visualize WhaleMoanDetector predictions:

	```bash
	python code/plot_predictions.py
	```
![spectrogram](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/SOCAL_H_65_spectrogram_with_labels.JPG)
![spectrogram](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/SOCAL_H_65_spectrogram_with_labels2.JPG)
