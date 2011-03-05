function rvc = ffTravelDistance(R, varargin)

rvc = zeros(length(R.D),1);

for d = 1:length(R.D)
   rvc(d) = R.D(d).TravelDistance;
end

