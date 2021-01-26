# A-graph-convolutional-neural-network-for-addressing-small-scale-reaction-prediction
This is the code for "A-graph-convolutional-neural-network-for-addressing-small-scale-reaction-prediction" paper.
The preprint of this paper can be found in ChemRxiv with 
## Python 2.7.12
## Tensorflow 1.3.0
## Numpy 1.16.6
## Django 1.11.29
## RDkit 2017.09.1
# Dataset
The small dataset we used is Baeyer-Villiger oxidation reaction, which contains approximately 2071 chemical reactions. The data are extracted from the "Reaxys" database. After filtering irrelevant information and the simplified reaction dataset contains reactants and products only. We further canonicalize those reactions SMILES and apply the RXN Mapper to map the reaction data
# Train
Model use the nntrain_direct.py to train to find reaction center and use the nntrain_direct_useScores.py to train to rank candidate
# Test
Model use the nntest_direct.py and nntest_direct_useScores.py to start testing. 
