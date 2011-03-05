function rvc = ffSpeedNuclei(R, varargin)


rvc = zeros(length(R.D),1);
nPoints = 0;

for d = 1:length(R.D)
   if( isempty(R.D(d).ID) || R.D(d).ID==0)
       continue;
   end
   nPoints = nPoints+1;
   rvc(nPoints) = R.D(d).Speed;
end

rvc = rvc(1:nPoints);
