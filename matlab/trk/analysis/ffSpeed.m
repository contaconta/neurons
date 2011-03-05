function rvc = ffSpeed(R, varargin)

rvc = zeros(length(R.D),1);

for d = 1:length(R.D)
   d
   rvc(d) = R.D(d).Speed;
end

