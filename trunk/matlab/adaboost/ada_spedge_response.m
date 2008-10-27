function f = ada_spedge_response(index, DATA)
%ADA_SPEDGE_RESPONSE returns the spedge feature response
%
%
%
%
%



%% if we have many images, but a single location, angle
if ~iscell(index)
   
    f = DATA(index,:);
    
%% if we have a single image, but many feature types
else
    
    f = cellfun(@(x) DATA(x), index); 
end







%% response for non-vectorized spedges

% %% if we have many images, but a single location, angle
% if ~iscell(descriptor)
%     
%     angle   = descriptor(1);
%     r       = descriptor(2);
%     c       = descriptor(3);
%     f = arrayfun(@(x)x.sp(angle,r,c), DATA, 'uniformoutput', false);
%     f = cell2mat(f);
% 
% %% if we have a single image, but many feature types
% else
%     
%     f = cellfun(@(x) DATA(x(1),x(2),x(3)), descriptor);   
% end
