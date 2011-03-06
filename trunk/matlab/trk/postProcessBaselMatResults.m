


dirIn  = '/net/cvlabfiler1/home/ksmith/Basel/Results/';
dirOut = '/net/cvlabfiler1/home/ksmith/Basel/PostResults/';

for nTrial =14:17
   matlabpool
   parfor nRun = 1:200
     matFileName = sprintf('%i-11-2010_%03i.mat', nTrial, nRun)
     if( exist([dirIn matFileName]) > 0)
       disp(matFileName);
        R = load([dirIn matFileName]);
        R = trkPostProcessing(R);
        save([dirOut matFileName], '-struct', 'R');
       disp([matFileName ' done']);
     end
   end
   matlabpool close
end

