## Reproducing the Paper

If reproducing the experiments in the paper, we recommend creating a separate Conda environment:

```
git clone https://github.com/MLforHealth/predictive_checklists
cd predictive_checklists/
conda env create -f environment.yml
conda activate ip_checklists
```

Then, follow the instructions in the README to download and install [CPLEX Optimization Studio](https://www.ibm.com/ca-en/products/ilog-cplex-optimization-studio).



To reproduce the experiments in the paper which involve training grids of checklists using different methods, use `scripts/sweep.py` as follows:

```
python -m scripts.sweep launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `scripts/experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `scripts/launchers.py` (i.e. `slurm` or `local`).

We are not able to provide any data (other than UCI Heart) due to privacy reasons. Instructions for accessing the datasets used in the paper can be found in Appendix D.