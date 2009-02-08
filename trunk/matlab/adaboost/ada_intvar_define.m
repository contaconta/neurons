function intvar = ada_intvar_define(IMSIZE)
%
%   intvar = ada_intvar_define(IMSIZE)
%
%   defines intvar weak learners
%
%
%


intvar(1).boundingbox   = [1 1 IMSIZE(1) IMSIZE(2)];
intvar(1).polarity      = single(1);
