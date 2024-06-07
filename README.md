# FastGPR: Divide and conquer approach for Gaussian Process Regression (GPR)

This repository contains the implementation of the FastGPR algorithm as well as the scripts to perform the experiments mentioned in "FastGPR: divide-and-conquer technique in neuroimaging data shortens training time and improves accuracy"

## Abstract:
Gaussian process regression (GPR) has shown success in neuroimaging applications such as predicting subjects’ age from structural Magnetic Resonance Imaging (MRI) scans. An important drawback of GPR is the high computational demand during the training phase. The necessity of big datasets for training accurate models in combination with the inherent high dimensionality of neuroimaging data make the training of such models impractical for conventional computers. Although techniques, such as divide-and-conquer have been applied to address this issue, so far, they have not been validated in neuroimaging data. In this study we investigated the viability of training multiple GPR models on data splits when predicting chronological age using MRIs. We examined the performance and runtime for various sizes of splits of the training sets as well as the impact of the number of features. We found that creating an ensemble of models by repeatedly splitting the training data into two subsets reduces training time without compromising performance. This approach provides a practical way of overcoming the computational burden of the training phase of GPR models in neuroimaging applications. 

## Citing

This will be updated soon!