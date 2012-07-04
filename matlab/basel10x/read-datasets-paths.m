clear all; close all; clc;
%%
fid = fopen('sinergia-repeats.txt');

tline = fgetl(fid);
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
end

fclose(fid);