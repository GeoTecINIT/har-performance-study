# iLOSO results

These directories contain the raw iLOSO results. These raw results were obtained running the `03_incremental-loso.py` script in `lib/pipeline`:

```bash
$ python 03_incremental-loso.py 
    --data_dir <PATH_OF_DATA> 
    --reports_dir <PATH_TO_STORE_RECORDS>
    --model <MLP,CNN,LSTM,CNN-LSTM>
    --subject <EVALUATION_SUBJECT>
    --batch_size <BATCH_SIZE>
    --epochs <EPOCHS>
    --splits <SPLITS>
```

**Parameters:**
- `--data_dir`: directory of processed data (output of `01_data-processing.py` script).
- `--reports_dir`: directory to store the results.
- `--model`: one of the listed models.
- `--subject` (optional): if provided, perform iLOSO using only on the specified subject. Otherwise, performs iLOSO for every subject.
- `--batch_size` (optional): batch size for training the models. **Default value: 64**.
- `--epochs` (optional): epochs of training. **Default value: 50.**
- `--splits` (optional): number of models trained for each combination (evaluation subject - n train subjects): **Default value: 10**.

**Usage example** from `lib/pipeline`:

```bash
$ python 03_incremental-loso.py
        --data_dir ../../01_DATA/01_DATASET/PROCESSED
        --reports_dir ../../01_DATA/03_MODEL_REPORTS
        --model mlp
```

**Results**:

The results are analysed in `./1_training-data.ipynb`, `./2_data-sources.ipynb` and `./3_models.ipynb` notebooks.