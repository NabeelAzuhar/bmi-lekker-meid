clc
clear all

load monkeydata_training.mat

num_neurons = trial(1,1).spikes;
print(num_neurons);