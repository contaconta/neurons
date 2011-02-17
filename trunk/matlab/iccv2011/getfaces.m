function [c x y]= getfaces(A, name, ind)


inds = find(strcmp(A.textdata, name));

D = A.data(inds,:);

SCALE_FACTOR = 1.3;  %1.4

%filename left-eye right-eye nose left-corner-mouth center-mouth right-corner-mouth

c = []; x = []; y = [];

for i = 1:size(D,1)
    leyex = D(i,1);
    leyey = D(i,2);
    reyex = D(i,3);
    reyey = D(i,4);
    nosex = D(i,5);
    nosey = D(i,6);
    lmoux = D(i,7);
    lmouy = D(i,8);
    cmoux = D(i,9);
    cmouy = D(i,10);
    rmoux = D(i,11);
    rmouy = D(i,12);
    
    x(i,:) = [leyex reyex nosex lmoux cmoux rmoux];
    y(i,:) = [leyey reyey nosey lmouy cmouy rmouy];
    
    % x or col
    %c(i,1) = round(mean([leyex reyex nosex lmoux cmoux rmoux]));
    c(i,1) = round(mean([leyex reyex]));
    
    % y or row
    c(i,2) = round(mean([leyey reyey nosey ]));  %cmouy
    c(i,2) = round(mean([leyey reyey nosey cmouy rmouy]));  %cmouy
    
    xmin = min([leyex reyex nosex lmoux cmoux rmoux]);
    xmax = max([leyex reyex nosex lmoux cmoux rmoux]);
    
    ymin = min([leyey reyey nosey lmouy cmouy rmouy]);
    ymax = max([leyey reyey nosey lmouy cmouy rmouy]);
    
    dx = abs(xmax - xmin);
    dy = abs(ymax - ymin);
    
    %d = mean([dx dy]);
    d = dy;
    
    %d = min([d size(I,2)-c(i,1) size(I,1)-c(i,2) c(i,1) c(i,2)  ]);
    
    % width
    c(i,3) = round(SCALE_FACTOR*d);
    
    c(i,4) = round(SCALE_FACTOR*d);
end

%keyboard;