function rvc = fnGermanNumNeuritesFreqExpansion(R, varargin)


% By default look at happy neurons
LookOnlyAtHappyNeurons = 0;

optargin = size(varargin,2);
if optargin >= 1
   LookOnlyAtHappyNeurons = varargin{1};
end


rvc = zeros(length(R.CellTimeInfo),1);
nPoints = 0;

for d = 1:length(R.CellTimeInfo)
    
   if ( isempty(R.trkSeq{d}) || R.D(R.trkSeq{d}(1)).ID==0 || ... % Check for validity of detection
      (LookOnlyAtHappyNeurons && R.D(R.trkSeq{d}(1)).Happy==0) )         % Shall we remove sad neurons from statistics?
       continue;
   end
   nPoints = nPoints+1;
   rvc(nPoints) = R.CellTimeInfo(d).GermanNumNeuritesFreqExpansion;
end

rvc = rvc(1:nPoints);
