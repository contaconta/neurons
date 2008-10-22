function f = ada_spedge_response(descriptor, DATA)
%ADA_SPEDGE_RESPONSE returns the spedge feature response
%
%
%
%
%


%% if we have many images, but a single location, angle
if ~iscell(descriptor)
    
    angle   = descriptor(1);
    r       = descriptor(2);
    c       = descriptor(3);
    f = cell2mat(arrayfun(@(x)x.sp(angle,r,c), DATA, 'uniformoutput', false));

%% if we have a single image, but many feature types
else
    
    f = cellfun(@(x) DATA(x(1),x(2),x(3)), descriptor);   
end
