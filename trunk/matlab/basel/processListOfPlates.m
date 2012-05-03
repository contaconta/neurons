function processListOfPlates(datasets_paths_filename)



fid = fopen(datasets_paths_filename);
C = textscan(fid, '%s %s %s %s %s');
fclose(fid);

inputDataRoot      = '/raid/data/store/';
outputAnalisysRoot = '/raid/data/analysis/';

for i= 1:length(C{1})
   % process 
   %plateCode = C{1}(i);
   %DataSetType = C{2}(i);
   Sample	 = C{3}(i);
   Identifier = C{4}(i);
   Location = C{5}(i);
   keyboard;
   inputFolder = [inputDataRoot Location];
   resultsFolder = [outputAnalisysRoot 'original/' Identifier '/'];
   if( ~exist(resultsFolder, 'dir') )
       mkdir(resultsFolder);
   end
   processPlate(inputFolder , resultsFolder, Sample); 
end