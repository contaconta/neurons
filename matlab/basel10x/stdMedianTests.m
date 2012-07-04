clear all; close all; 
%%
figure; hold on;
colormaps = jet(5);
for i =2:6
    [stdSeqRed, maxSeqRed, minSeqRed, meanSeqRed, medianSeqRed] = trkReadAndNormalizeImages2(97, ['/Users/feth/Documents/Work/Data/Sinergia/Olivier/10xtest/'  sprintf('%03d', i) '/green/']);
    plot(stdSeqRed, 'color', colormaps(i, :))
    keyboard;
end
for i =2:6
    [stdSeqGreen, maxSeqGreen, minSeqGreen, meanSeqGreen, medianSeqGreen] = trkReadAndNormalizeImages2(97, '/Users/feth/Documents/Work/Data/Sinergia/Olivier/10xtest/002/green/');
end
