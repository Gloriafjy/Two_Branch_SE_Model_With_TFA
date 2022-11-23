## Installation

First, install Python (recommended with Anaconda).

## Development

Clone this repository and install the dependencies. We recommend using
a fresh Conda environment.

```bash
git clone https://github.com/Gloriafjy/Two_Branch_SE_Model_With_TFA
cd code
pip install -r requirements_cuda.txt
```

## Train and evaluate

### 1. Data

Run `json_extract.py` to generate json files, which records the utterance file names for both training and validation set.

	# Run json_extract.py
	json_extract.py
	
	
### 2. Train
Training is simply done by launching the `mian_merge.py` script.
`solver_merge.py` and `train_merge.py` contain detailed training process.

	# Run main.py to begin network training 
	main_merge.py

### 3. Inference

The trained weights XXXXXX.pth.tar on VB dataset is also provided in BEST_MODEL. 

	# Run main.py to enhance the noisy speech samples.
	enhance.py 
