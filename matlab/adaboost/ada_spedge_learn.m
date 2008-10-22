function [error, theta, polarity] = ada_spedge_learn(i, TRAIN, WEAK, w)

training_labels = [TRAIN(:).class];

POS = TRAIN([TRAIN.class] == 1);
NEG = TRAIN([TRAIN.class] == 0);


err = 
