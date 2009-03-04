function unitvec = unitvector(data)
%unitvec = unitvector(data)
%   u = unitvector(35);  % can accept angle arguments
%   u = unitvector([2 5]);  % can accept vector arguments
%   
%   angles in degrees
%  
%


% we have been given an angle
if length(data) == 1
    
    unitvec(1) = sind(data);
    unitvec(2) = cosd(data);
    
    
% we have been given a vector    
else 
    % ensure it is a column vector
    data = data(:);

    if isequal(data, [0 0]');
        % we this has no magnitude, pick a random angle
        ang = rand(1)*2*pi;
        unitvec(1) = sin(ang);
        unitvec(2) = cos(ang);
    else
        % normalize the vector
        unitvec = squeeze(l2norm(data));
    end
    
end