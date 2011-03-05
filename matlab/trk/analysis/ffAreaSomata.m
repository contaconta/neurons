function rvc = ffAreaSomata(R, varargin)

LookOnlyAtHappyNeurons = 0;

optargin = size(varargin,2);
if optargin >= 1
   LookOnlyAtHappyNeurons = varargin{1};
end

rvc = zeros(length(R.D),1);
nPoints = 0;

for d = 1:length(R.D)
   if (isempty(R.D(d).ID)) || R.D(d).ID==0 || ... % Check for validity of detection
      (LookOnlyAtHappyNeurons && R.D(d).Happy==0) % Shall we remove sad neurons from statistics?    
       continue;
   end
   nPoints = nPoints+1;
   rvc(nPoints) = R.Soma(d).Area;
end

rvc = rvc(1:nPoints);