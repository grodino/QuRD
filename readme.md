Dear reviewers, thank you for looking at the code. Here are instructions to run it and generate the
paper's figures.

# Setup 

1. Download the ModelDiff's benchmark models from https://github.com/yuanchun-li/ModelDiff
2. Download the SAC's benchmark models from https://github.com/guanjiyang/SAC
3. Install the modified version of `timm` using `pip install -e pytorch-image-models`

# Run the experiments

1. Use the `experiments/scripts/paper.py` to generate the script used to run the experiments of the
   paper
2. Execute the script

# Analyze the results

Run the `paper.ipynb` notebook. The results of the experiments used in the paper are in the
`generated` folder.