# PreAlgPro

PreAlgPro: A Neural Network Model for Allergenic Protein Identification through Pre-trained Protein Language Model 

Lingrong Zhang, Taigang Liu*

College of Information Technology, Shanghai Ocean University, Shanghai 201306, China

Abstract: Allergic reactions are caused by allergens, which can cause harm to life. At the same time, allergens also impact food, pharmaceuticals, and other fields, so accurately identifying allergenic proteins is significant. Nevertheless, identifying allergenic proteins through experimental means often entails considerable time and labor investments, which implies an urgent need for computational methods. The existing computational methods are either based on protein homology or use physicochemical characteristics of proteins for classification. In this study, we proposed a novel framework called PreAlgPro for allergenic protein prediction. First, we used pre-trained protein language models (PPLMs) to extract the embedding features of proteins and compared four different mainstream models to pick the best one. Second, we combine the attention mechanism and the convolutional neural network with residuals for feature extraction. Third, we use a five-fold cross-validation and independent test set to verify the model's effectiveness. Finally, we also conduct a case analysis to ascertain the robustness of the model. Based on the same independent test set, PreAlgPro showcased remarkable performance, attaining an accuracy of 99.01% and a recall of 98.17%. Our model achieved an outstanding overall precision of 99.86%, exhibiting a notable improvement of 6.55% to 14.72% compared to the existing state-of-the-art predictors. This indicated that PreAlgPro is an efficient method for allergenic protein identification. Our code and data are available at https://github.com/zlr-zmm/PreAlgPro.

Keywords: pre-trained protein language model; allergenic protein; deep learning; attention mechanism
