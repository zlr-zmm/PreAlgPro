# PreAlgPro Framework

This repository contains the code for the PreAlgPro framework, which is used for predicting allergenic protein. 

PDNAPred relies on large-scale pre-trained protein language model: ProtT5. These models are implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

# Usage

First, you need to download the model weights for ProtT5 from the provided Hugging Face URLs. Please visit the above links to download the respective weight files.

Save the downloaded weight files in your working directory and make sure you know their exact paths.

Next, you will use the provided `model_embedding.py scripts to generate embedding features for ProtT5 , respectively. In these scripts, you need to modify the file paths according to your needs.

For the `model_embedding.py` scripts, you need to run them separately to generate embedding features for ProtT5 . Run the following commands:

```
python model_embedding.py
```

This will generate the corresponding embedding features files.

Finally, you can proceed with model training and validation using the provided `train.py` script. Before running it, make sure you have prepared the training data and labels, and have downloaded the weight files and generated embedding feature files.

In the `train.py` script, you need to modify the file paths and other parameters according to your needs. Run the following command to start the model training and validation:

```
python train.py
```

The script will train the model and perform validation based on the data and parameters you provided, and it will save the output in the specified output directory.

Please note that the file paths and other parameters in the above steps need to be modified according to your own setup. Make sure you have installed the required dependencies properly, and follow the steps in the specified order.

# Contact 

In addition, we also provide Colab code, here is the code link: https://colab.research.google.com/drive/1aCgZbvrLxTuBaXPdzFfvyVWfIUJacv5K?usp=sharing

If you have any questions regarding the code, paper, or data, please feel free to contact Lingrong Zhang at [zlr_zmm@163.com](mailto:zlr_zmm@163.com).
