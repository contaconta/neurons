function RESPONSE = vj_fast_haar_response(f,II)
%VJ_FAST_HAAR_RESPONSE returns the haar feature response from and integral image
%
%   RESPONSE = vj_fast_haar_response(f,II) takes a row vector feature
%   'f' and a column vectorized integral image II (e.g. II = II(:)) and 
%   returns the haar-like feature response of the feature to integral image.  
%   The response is defined as the difference in intensity between dark and 
%   white regions.  Responses of multiple features can be computed by
%   adding rows to the feature vector.  Responses to multiple integral
%   images can be computed by adding columns to II.  Returns a row vector
%   of feature responses corresponding to each feature or integral image.
%
%   f =  [ f e a t u r e ]   II = [ I
%                                   n
%                                   t
%                                   e
%                                   g
%                                   r
%                                   a
%                                   l
%                                   i
%                                   m
%                                  (:) ]
%
%   Copyright © 2008 Kevin Smith
%
%   See also INTEGRAL_IMAGE, VJ_CLASSIFY_STRONG

RESPONSE = (f * II);

if size(RESPONSE,1) ~=1
    RESPONSE = RESPONSE';
end
