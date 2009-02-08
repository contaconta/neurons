function intmean = ada_intmean_define(IMSIZE)
%
%   intmean = ada_intmean_define(IMSIZE)
%
%   defines intmean weak learners
%
%
%


intmean(1).boundingbox  = [1 1 IMSIZE(1) IMSIZE(2)];
intmean(1).polarity     = single(1);
