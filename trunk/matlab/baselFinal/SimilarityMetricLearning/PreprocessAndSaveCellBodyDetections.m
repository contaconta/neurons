function [] = PreprocessAndSaveCellBodyDetections(Magnification, dataRootDirectory, RawRootDataDirectory, DetectionDirectory)



listOfGTSeq = dir(dataRootDirectory);

for i = 1:length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       inputSeqDirToProcess = [RawRootDataDirectory listOfGTSeq(i).name '/'];
       disp('---------------------------------------');
       disp(['processing ' inputSeqDirToProcess]);
       detectionsFileName = [DetectionDirectory listOfGTSeq(i).name];
       [Nuclei, Somata, Cells, CellsList] =  trkDetectNucleiSomataForEvaluation(inputSeqDirToProcess, Magnification); %#ok
       disp('saving ... ');
       save(detectionsFileName, 'Cells', 'CellsList');
       disp('=======================================');
   end
end