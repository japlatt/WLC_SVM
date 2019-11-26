Repository for the paper entitled "Machine Learning Classification Informed by a Functional Biophysical System" which can be found https://arxiv.org/abs/1911.08589 and has been submitted to PRL as of 11/25/2019.

Abstract:
We present a novel machine learning architecture for classification suggested by experiments on the insect olfactory system. The network separates odors via a winnerless competition network, then classifies objects by projection into a high dimensional space where a support vector machine provides more precision in classification. We build this network using biophysical models of neurons with our results showing high discrimination among inputs and exceptional robustness to noise. The same circuitry accurately identifies the amplitudes of mixtures of the odors on which it has been trained.

The main file is SVM.py which builds the WLC network, projects and plots the network activity and classifies various noisy perturbations.  This is the main result of the paper.

mixtures.py presents mixtures of 2 odors and creates the plot found in figure 4 of the paper.

The rest of the files are for support.

