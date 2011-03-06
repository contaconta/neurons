

function postProcessBasselMatresults(nRunBegin, nRunEnd)


dirIn  = '/net/cvlabfiler1/home/ksmith/Basel/Results/';
dirOut = '/net/cvlabfiler1/home/ksmith/Basel/PostResults/';


for nTrial =14:17
   for nRun = nRunBegin:1:nRunEnd
     matFileName = sprintf('%i-11-2010_%03i.mat', nTrial, nRun);
     disp(matFileName);
     if( exist([dirIn matFileName]) > 0)
       disp(matFileName);
        R = load([dirIn matFileName]);
        R = trkPostProcessing(R);
        save([dirOut matFileName], '-struct', 'R');
       disp([matFileName ' done']);
     end
   end
end

 quit
