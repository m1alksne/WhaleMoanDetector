# WhaleMoanDetector: Detecting Blue and Fin Whale Calls in Audio
*Michaela Noel Alksne and Shane Andres*

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning model for automatically detecting and classifying blue whale and fin whale calls in audio datasets.

![Fin whale](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/blue_whale.jpeg)
*Blue whales off the coast of San Diego. Photo credit: Manuel Mendieta*

## Overview

WhaleMoanDetector (WMD) is a specialized tool that identifies time and frequency bounds on whale vocalizations in audio data. It's particularly designed to detect:

- **Blue whale calls**: A, B, and D calls from Northeast Pacific populations
- **Fin whale calls**: 20 Hz and 40 Hz pulses

The system uses a [Faster R-CNN](https://arxiv.org/abs/1506.01497) object detection model trained on spectrograms. The tool is designed to interface with [WhaleMoanViz](https://github.com/m1alksne/WhaleMoanViz) to create an active learning pipeline. This allows users to incrementally improve the performance of their model on unlabeled data by providing it with human feedback.

This project is still in development, so please reach out with any issues or feature requests you might have!

![Example spectrogram with detections](https://github.com/m1alksne/WhaleMoanDetector/blob/main/figures/all_example.JPG)


## Getting Started

1. Download the Conda package manager with [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/). For more information, reference the [Miniconda tutorial](https://docs.anaconda.com/working-with-conda/environments/).

2. Download WMD through GitHub:
    - Click the green "Code" button at the top of this page
    - Select "Download ZIP"
    - Extract the ZIP file to a location you'll remember (like `C:\WhaleMoanDetector\` or `~/WhaleMoanDetector/`)

    Alternatively, if you have git installed run the following command:
    ```bash
    git clone https://github.com/m1alksne/WhaleMoanDetector.git
    cd WhaleMoanDetector
    ```

3. Open Anaconda Prompt (Windows) or Terminal (MacOS, Linux) and create a new Conda Python environment:
    ```bash
    conda create -n whalemoandetector python=3.9
    ```

4. If your computer has a CUDA-enabled GPU, install a [CUDA Driver](https://www.nvidia.com/en-us/drivers/). Then, install [PyTorch](https://pytorch.org/get-started/locally/) with a compatable CUDA version.

5. Navigate to the WhaleMoanDetector folder and install dependencies in the new environment:
    ```bash
    activate whalemoandetector
    pip install -r requirements.txt
    ```



## Running Inference with a Model

To run the model on an unlabeled audio dataset and generate call predictions:

### Step 1: Configure Project Settings

1. Navigate to the `code` folder in `WhaleMoanDetector`
2. Open `config.yaml` in a text editor (like Notepad++ or VS Code)
3. Modify these paths and parameters:

```yaml
categories: {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}  # Call type mappings

spectrogram:
  CalCOFI_flag: False                                       # Indicates whether AIS signals are present in audio

train:
  model_folder: `C:/path/to/your/model`                     # Where your models are stored

inference:
  wav_folder: 'C:/path/to/your/audio/files'                 # Where your .wav files are stored
  detections_folder: 'C:/path/to/save/results'              # Where to save detection results
  spectrogram_folder: 'C:/path/to/save/spectrograms'        # Where to save spectrogram images
```

  **Note that the rest of the `spectrogram` parameters must match those of the model being used.** Ensure that your model is stored in `model_folder`.


### Step 2: Configure Model Settings

Open `WhaleMoanDetector/code/inference_pipeline.py` and navigate to the `user input` section. Set the following:

```python
# !!! user input !!!
model_name = "your_model_name"          # Name of your trained model
model_constructor = RCNN_ResNet_50      # Must match your model's architecture
eval_epoch = 29                         # Which training epoch to use (usually the last one)
```

### Step 3: Run the detection pipeline

Navigate to the `code` folder and run the inference script:

```bash
python inference_pipeline.py
```

This will process all .wav and .x.wav files in `wav_folder` and save:
- `raw_detections.txt`, containing all predicted calls in the model output format (see Appendix)
- `raw_detections_context_filtered.txt`, a filtered version of the predictions after applying biological context rules
- Spectrogram images for each portion of the audio file where a whale call was predicted

### Step 4: Visualizing Results

To see the detections on spectrograms, run:

```bash
python plot_predictions.py "C:/path/to/your/detections_file.txt"
```

This will show each spectrogram with bounding boxes around detected calls, labeled by call type and confidence score.




## Testing Model Performance

If you have labeled data with known whale calls, you can evaluate your model's performance:

### Step 1: Configure Project Settings

Modify these paths and parameters in `config.yaml` for your project:

```yaml
categories: {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}  # Call type mappings

train: 
  model_folder: 'C:/path/to/your/models'                    # Where your model is stored
  labeled_data_folder: 'C:/path/to/annotations'             # Where model input detections are pulled from
  evaluation_folder: 'C:/path/to/eval'                      # Where performance metrics are saved
```

### Step 2: Configure Test Settings

Open `WhaleMoanDetector/code/test.py` and fill the `user input` section:

```python
# !!! user input !!!
model_name = "your_model_name"
model_constructor = RCNN_ResNet_50              # Must match your model's architecture
val_set_file = "test_annotations.txt"           # File with ground truth detections
eval_epoch = 29                                 # Which trained epoch to evaluate
iou_threshold = 0.1                             # Intersection over union threshold for matching predictions to ground truth
```

### Step 3: Run Test

Navigate to the `code` folder and run the test script:

```bash
python test.py
```

The evaluation will generate precision and recall metrics for each call type at different confidence thresholds. These will be saved in your evaluation folder with the format: `{val_set_name}_{model_name}_{int(100*iou_threshold)}_percent_iou.txt`



## Training a Model

This section describes how to train a model from scratch, assuming the user has training examples (annotations and spectrograms). Before training, you need:

1. Labeled training data in the model input format (see Appendix)
2. Spectrogram images corresponding to your annotations
3. Sufficient computational resources (GPU recommended)

### Step 1: Configure Project Settings

Modify these paths and parameters in `config.yaml` for your project:

```yaml
categories: {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}  # Call type mappings

train: 
  model_folder: 'C:/path/to/your/models'                    # Where new models are saved
  labeled_data_folder: 'C:/path/to/annotations'             # Where model input detections are pulled from
  evaluation_folder: 'C:/path/to/eval'                      # Where performance metrics are saved
```

Ensure that your input detections are stored in `labeled_data_folder`.`spectrogram` settings can be modified if desired.


### Step 2: Configure Model Settings 

Open `WhaleMoanDetector/code/train.py` and fill the `user input` section:

```python
    # !!! user input !!!

    model_name = "your_model_name"
    model_constructor = RCNN_ResNet_50              # Initialize model architecture (see `model_functions.py`)
    train_set_file = "train_annotations.txt"
    val_set_file = "val_annotations.txt"

    lr = 0.001                                      # Learning rate
    momentum = 0.9                                  # SGD momentum
    weight_decay = 0.0005                           # Weight decay for regularization
    num_epochs = 30                                 # Number of training epochs
    train_batch_size = 4                            # Batch size for training
    val_batch_size = 1                              # Batch size for validation


    model_log = {                                   # Metadata that is saved with the model. All fields are optional
        "model_name": model_name,
        "notes": "Testing out the code.",
        "dataset": "train_annotations",
        # ... additional metadata
```

### Step 3: Run Training

Navigate to the `code` folder and run the train script: 

```bash
python train.py
```

The training process will:

1. Create folders for your model inside of `model_folder` and `evaluation_folder`
2. Display progress with a real-time progress bar showing current epoch and batch
3. Save checkpoints for each epoch in `model_folder` with the naming pattern: `{model_name}_epoch_{epoch_number}.pth`
4. Log validation metrics for each epoch in `evaluation_folder`



## Troubleshooting

1. **"Module not found" errors**
   - Make sure you are running code from inside `WhaleMoanDetector/code`
   - Make sure your environment is activated: `conda activate whalemoandetector`

2. **CUDA/GPU errors**
   - The code will automatically use CPU if GPU isn't available
   - For GPU support, ensure you have compatible NVIDIA drivers

3. **File path errors**
   - Use forward slashes (`/`) in paths, even on Windows
   - Avoid spaces in folder names when possible



## Related Publications

Oleson, E., J. Calambokidis, W. Burgess, M. Mcdonald, C. A. Leduc and J. A. Hildebrand. 2007. Behavioral context of Northeast Pacific blue whale call production. Marine Ecology-progress Series - MAR ECOL-PROGR SER 330:269-284.
[https://www.int-res.com/abstracts/meps/v330/p269-284/](https://www.int-res.com/abstracts/meps/v330/p269-284/)

Širović, A., L. N. Williams, S. M. Kerosky, S. M. Wiggins and J. A. Hildebrand. 2013. Temporal separation of two fin whale call types across the eastern North Pacific. Marine Biology 160:47-57.
[https://link.springer.com/article/10.1007/s00227-012-2061-z](https://link.springer.com/article/10.1007/s00227-012-2061-z)



## Appendix

### Model Input Format

Training annotations should be tab-delimited text files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `spectrogram_path` | Path to the spectrogram image | `C:/spectrograms/audio1_20230801T120000.png` |
| `label` | Call type | `A`, `B`, `D`, `20Hz`, `40Hz` |
| `xmin`, `xmax`, `ymin`, `ymax` | Bounding box coordinates in the spectrogram image | `150`, `250`, `80`, `120` |

Rows with only `spectrogram_path` filled represent hard negative examples (spectrograms with no calls).

### Model Output Format

Prediction files are tab-delimited with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `wav_file_path` | Source audio file path | `C:/audio/audio1.wav` |
| `model_no` | Name of the model that made the detection | `my_model` |
| `image_file_path` | Path to the spectrogram image | `C:/spectrograms/audio1_20230801T120000.png` |
| `label` | Detected call type | `A`, `B`, `D`, `20Hz`, `40Hz` |
| `score` | Confidence score (0-1, higher is more confident) | `0.50` |
| `start_time_sec`, `end_time_sec`| Detection start / end time in seconds from file start | `15`, `34` |
| `start_time`, `end_time` | Absolute start / end timestamp | `2023-08-01 12:00:15`, `2023-08-01 12:00:34` |
| `min_frequency`, `max_frequency` | Minimum / maximum frequency of the detection (Hz) | `23`, `40` |
| `box_x1`, `box_x2`, `box_y1`, `box_y2` | Bounding box coordinates in the spectrogram image | `169.889`, `263.782`, `63.582`, `77.574` |



## WhaleMoanDetector Directory Structure 

```
WhaleMoanDetector/
├── LICENSE
├── README.md          <- The top-level README for users.
├── legacy             <- Legacy code from previous versions of WhaleMoanDetector.
├── code               <- All code required to run WhaleMoanDetector.
│   ├── AudioDetectionDataset.py     <- Dataset class.
│   ├── AudioStreamDescriptor.py     <- Parses xwav and wav headers.
│   ├── call_context_filter.py       <- Filters model predictions based on duration and frequency.
│   ├── config.yaml                  <- Contains project-specific file paths and settings.
│   ├── custom_collate.py            <- Collate function.
│   ├── inference_functions.py       <- Helper functions to run inference on a single audio file.
│   ├── inference_pipeline.py	     <- Runs inference and generates predictions for an audio dataset.
│   ├── make_new_examples.py         <- Converts validated predictions into new examples to train a model with.
│   ├── model_functions              <- Helper functions for loading in specific model architectures.
│   ├── plot_groundtruth.py          <- Plots all spectrograms and bounding boxes from a file in the model input format.
│   ├── plot_predictions.py          <- Plots all spectrograms and bounding box predictions from a file in the model output format.
│   ├── PR_curve.py      	         <- Generates PR curve for test data.
│   ├── spectrogram_functions.py     <- Helper functions to work with with audio file chunks and spectrograms.
│   ├── test.py                      <- Runs inference and generates performance metrics on an audio dataset.
│   ├── train.py                     <- Training loop for WhaleMoanDetector.
│   ├── validation.py                <- Function to perform validation during training.
```