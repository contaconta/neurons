function Cells = trkComputeIntensityHistograms(Cells, NUMBER_OF_BINS)

lastCell = Cells(end);
parfor i=1:numel(Cells)
    Cells(i).SomaHistGreen = histc(Cells(i).SomaGreenIntensities, ...
                                   linspace(lastCell.MinGreen, ...
                                            lastCell.MaxGreen, ...
                                            NUMBER_OF_BINS)          );%#ok
	% normalize the histogram to a probability
	Cells(i).SomaHistGreen = Cells(i).SomaHistGreen / sum(Cells(i).SomaHistGreen);
end

