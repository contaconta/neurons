function f = test_faster(finds, fvals, II)
%ADA_FAST_HAAR_RESPONSE returns the haar feature response from and integral image
%
%   RESPONSE = ada_fast_haar_response(f,II) takes a row vector feature
%   'f' and a column vectorized integral image II (e.g. II = II(:)) and 
%   returns the haar-like feature response of the feature to integral image.  
%   The response is defined as the difference in intensity between dark and 
%   white regions.  Responses of multiple features can be computed by
%   adding rows to the feature vector.  Responses to multiple integral
%   images can be computed by adding columns to II.  Returns a row vector
%   of feature responses corresponding to each feature or integral image.
%
%   f =  [ f e a t u r e ]   II = [ I   I
%                                   n   n
%                                   t   t
%                                   e   e
%                                   g   g
%                                   r   r
%                                   a   a
%                                   l   l
%                                   i   i
%                                   m   m
%                                   1   2
%                                  (:) (:)]
%
%   Copyright © 2008 Kevin Smith
%
%   See also INTEGRAL_IMAGE, ADA_CLASSIFY_STRONG

%f = sum( fvals.*II(finds,:), 1);

f = fvals * II(finds,:);

% function f = test_faster(fasterinds, fastervals, II)
% 
% 
% f = sum([fastervals{:}]'.*II([fasterinds{:}]));
% 
% 








% 
% function f = test_faster(faster, II)
% faster = nonzeros(faster);
% 
% f = sum(faster(:,2).*II(faster(:,1)));
