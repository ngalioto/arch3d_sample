# arch3d_sample
A sample of the code used to make the Hi-C foundation model ARCH3D.


## Installation
Clone the repository.
```bash
git clone https://github.com/ngalioto/arch3d_sample.git
cd arch3d
```

It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```

Install the repo and its requirements.
```bash
pip install ./
pip install -r requirements.txt
```

## Usage
### Data preprocessing
First, download your pre-training corpus as `.mcool` files. Then, to perform observed/expected normalization, run [`toeplitz_normalize`](https://github.com/ngalioto/arch3d_sample/arch3d/data/toeplitz_normalize.py) over each file. Make sure these files are saved into their own directory or directories.
```bash
python arch3d/data/toeplitz_normalize.py \
  "/path/to/mcool_file.mcool" \
  "/path/to/save_dir" \
  --weights_dir "/path/to/weights_dir" # this is optional. Use only if you want to use the `expected` weights later on.
```

### Configuration
Next, in the configuration file [`bert_large`](https://github.com/ngalioto/arch3d_sample/arch3d/conf/bert_large.yaml), change the list `datamodule['data_dir']` to contain the directories containing the Toeplitz-normalized Hi-C. Additionally, you can make changes to the model architecture or directory paths as desired.

### Pre-training
Then run [`pretrain`](https://github.com/ngalioto/arch3d_sample/arch3d/scripts/pretrain.py) with your updated configuration file.
```bash
python arch3d/scripts/pretrain.py \
  --config "arch3d/conf/bert_large.yaml"
```
The model will be saved to the directory `logger['save_dir']` specified in the configuration file.