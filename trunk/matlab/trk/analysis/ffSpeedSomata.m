function rvc = ffSpeedSomata(R, varargin)

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
   if(isempty(R.Soma(d).Speed))
     disp('encountered an empty value in ffSpeedSomata');
     keyboard;
   end
   if(R.Soma(d).Time == 1)
     continue;
   end


   nPoints = nPoints+1;
   rvc(nPoints) = R.Soma(d).Speed;
end

rvc = rvc(1:nPoints);
