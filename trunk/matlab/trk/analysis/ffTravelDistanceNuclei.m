function rvc = ffTravelDistanceNuclei(R, varargin)


% By default look at happy neurons
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
   if(isempty(R.D(d).TravelDistance))
     disp('encountered an empty value in ffTravelDistanceNuclei');
     keyboard; 
   end
   nPoints = nPoints+1;

   rvc(nPoints) = R.D(d).TravelDistance;
end

rvc = rvc(1:nPoints);
