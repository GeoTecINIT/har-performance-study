# Grid Search results

These directories contain the raw Grid Search results. These raw results were obtained running the `02_hyperparameter-optmization.py` script in `lib/pipeline`:

```bash
$ python 02_hyperparameter-optimization.py 
        --data_dir <PATH_OF_DATA> 
        --model <mlp,cnn,lstm,cnn-lstm>
        --phase <initial,extra-layers>
        --batch_size <BATCH_SIZE>
        --epochs <EPOCHS>
        --executions <EXECUTIONS>
```

**Parameters:**
- `--data_dir`: directory of processed data (output of `01_data-processing.py` script).
- `--model`: one of the listed models.
- `--phase`: `initial` for first phase and `extra-layers` for second phase.
- `--batch_size` (optional): batch size for training the models. **Default value: 64**.
- `--epochs` (optional): epochs of training. **Default value: 50.**
- `--executions` (optional): number of times each configuration is evaluated: **Default value: 5**.

**Usage example** from `lib/pipeline`:

```bash
$ python 02_hyperparameter-optimization.py 
        --data_dir ../../01_DATA/01_DATASET/PROCESSED
        --model mlp
        --phase initial
```

**Results**:

The results are visualized in table format in the `./0_grid-search.ipynb` notebook.