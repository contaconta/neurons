function out = convert_to_absolutePositions(params)


out = zeros(size(params, 1), size(params, 2)-2);

for k = 1:size(params, 1)
   out(k, 1) = params(k, 3) + params(k, 5)*(-params(k, 2)) + params(k, 6)*(-params(k, 1));
   out(k, 2) = params(k, 4) + params(k, 7)*(-params(k, 2)) + params(k, 8)*(-params(k, 1));
   out(k, 3) = params(k, 5)+1;
   out(k, 4) = params(k, 6);
   out(k, 5) = params(k, 7);
   out(k, 6) = params(k, 8)+1;
end