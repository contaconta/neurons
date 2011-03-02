function J=trkRegionGrow2(I,M,distThresh)


MAX_ITER = 5000;
MAX_N = 10000;
MAX_AREA = 5000;

J = 2*double(M);
SIZE = size(J);

muI = mean(I(M));
area = sum(sum(M));



pixdist=0; % Distance of the region newest pixel to the regio mean

% Neighbor locations (footprint)
neigb=[-1 0; 1 0; 0 -1;0 1];


P = bwperim(M);
[r c] = find(P);

N = zeros(MAX_N,3);
%plist = zeros(MAX_N,1);
plist = [];
for i = 1:length(r)
    N(i,:) = [r(i) c(i) I(r(i),c(i))];
    plist = [plist i];
    pmax = i;
end

r = r(1);
c = c(1);

iter = 0;

while(pixdist<distThresh && area<numel(I) && iter<MAX_ITER)

    if iter ~= 0
        % Add new neighbor pixels
        for j=1:4,
            % Calculate the neighbour coordinate
            rn = r +neigb(j,1); cn = c +neigb(j,2);

            % Check if neighbour is inside or outside the image
            ins=(rn>=1)&&(cn>=1)&&(rn<=SIZE(1))&&(cn<=SIZE(2));

            % Add neighbor if inside and not already part of the segmented area
            if(ins && (J(rn,cn)==0) ) 
                pmax = pmax+1;
                %plist(pmax) = 1;
                plist = [plist pmax];
                N(pmax,:) = [rn cn I(rn,cn)]; 
                J(rn,cn)=1;
            end
        end
    end    
    
    
    % distance calculation
    dist = abs(N(plist,3) - muI);
    
    % find the pixel with the min distance
    [pixdist, pindex] = min(dist);
    index = plist(pindex);
    J(r,c)=2; 
    %area=area+1;
    
    
    % Calculate the new mean of the region
    %reg_mean= (reg_mean*reg_size + neg_list(index,3))/(reg_size+1);
    
    
    % Save the x and y coordinates of the pixel (for the neighbour add proccess)
    r = N(index,1); 
    c = N(index,2);
    
    %disp(['[' num2str(index) ' ' num2str(pindex) ' ' num2str(r) ' ' num2str(c) ']']);
    
    
    % Remove the pixel from the neighbour (check) list
    plist(pindex) = [];
    iter = iter + 1;
    
    %imagesc(J);
    %pause(0.05);
    %keyboard;
end

J = J > 1;
%keyboard;



