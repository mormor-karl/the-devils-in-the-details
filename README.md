This is the repository for "The Devil's in the Details: the Detailedness of Classes Influences Personal Information Detection and Identification", a NoDaLiDa/Baltic-HLT 2025 submission. Since we cannot share the data due to privacy concerns, we only share the code (see details in the submission).

### Preparations
The token classification with Transformers is based on [this example](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification). Download this code into a folder called `bert` within the `experiments` folder.

### Previous code
This indicates which code was taken from a previous paper working with the same data.
- `bert/run_iob.sh`
- `data/preproc.sh`
- `analyze_output.py` - with our changes
- `reannotate_iob.py` - with our (major) changes

### Other code
- `analyze output.sh` is used to bulk calculate evaluation measures.
- `compare_predictions.py` is used to combine the predictions of all models into one for future analyses.
- `generate_splits.sh` is used to reannotate and split the data for fine-tuning.
- `visualize.py` is responsible for generating plots.
