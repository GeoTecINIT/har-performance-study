# Dataset

The dataset employed in this research is described in the following paper:

> Matey-Sanz, M., Casteleyn, S., & Granell, C. (2023). Dataset of inertial measurements of smartphones and smartwatches for human activity recognition. Data in Brief, 51, 109809. doi: 10.1016/j.dib.2023.109809

To reproduce the analyses, download the dataset and place its `DATA` folder in this directory. Then execute the data processing script, `01_data-processing.py` from `lib/pipeline`:

```bash
$ python 01_data-processing.py 
    --input_data_path <PATH_OF_RAW_DATA> 
    --windowed_data_path <PATH_TO_STORE_RESULTS>
```

**Parameters:**
- `--input_data_path`: directory of the raw data. For example, if the script is executde in `lib/pipeline`, use `../../01_DATA/01_DATASET/DATA`.
- `--windowed_data_path`: directory to store the processed data. For example, if the script is executde in `lib/pipeline`, use `../../01_DATA/01_DATASET/PROCESSED` to store the data in a folder named `PROCESSED` in this directory.

**Usage example** from `lib/pipeline`:

```bash
$ python 01_data-processing.py 
    --input_data_path ../../01_DATA/01_DATASET/DATA 
    --windowed_data_path ../../01_DATA/01_DATASET/PROCESSED
```