clear all; close all; clc;
%%
fid = fopen('sinergia-repeats.txt');
C = textscan(fid, '%s %s %s %s %s');




fclose(fid);