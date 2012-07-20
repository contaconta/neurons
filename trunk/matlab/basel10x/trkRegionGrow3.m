function J=trkRegionGrow3(I,M,THRESH,r,c)

WEIGHT_I = 800; %400;     % controls influence of intensity vs distance from nuclues
                    % higher = more inetensity influence, lower = more distance


WIN = 100;
MAX_ITER = 5000;
MAX_N = 10000;


J = 2*double(M);
SIZE = size(J);

muI = mean(I(M));
area = sum(sum(M));
MAX_AREA = 2.25*area;


% create a distance map
D = Inf * ones(size(M));
rmin = max(1, r - WIN); 
rmax = min(size(D,1), r + WIN);
cmin = max(1, c - WIN); 
cmax = min(size(D,2), c + WIN);

Dwin = M(rmin:rmax,cmin:cmax);
Dwin = bwdist(Dwin,'quasi-euclidean');
D(rmin:rmax,cmin:cmax) = Dwin;




pixdist=0; % Distance of the region newest pixel to the regio mean

% Neighbor locations (footprint)
neigb=[-1 0; 1 0; 0 -1;0 1];


P = bwperim(M);
[r c] = find(P);

N = zeros(MAX_N,4);
%plist = zeros(MAX_N,1);
plist = [];
for i = 1:length(r)
    N(i,:) = [r(i) c(i) I(r(i),c(i))  D(r(i),c(i))];
    plist = [plist i]; %#ok<AGROW>
    pmax = i;
end

r = r(1);
c = c(1);

iter = 0;

while(pixdist<THRESH && area<MAX_AREA && iter<MAX_ITER)

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
                N(pmax,:) = [rn cn I(rn,cn) D(rn,cn)]; 
                J(rn,cn)=1;
            end
        end
    end    
    
    
    % distance calculation
    distI = abs(N(plist,3) - muI);
    dist = WEIGHT_I.*distI + N(plist,4);
    
    % find the pixel with the min distance
    [pixdist, pindex] = min(dist);
    index = plist(pindex);
    J(r,c)=2; 
    area=area+1;
    
    
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

% Ir = I;
% Ig = I;
% Ib = I;
% Ir(J) = 0;
% Ig(J) = 1;
% Ib(J) = 0;
% 
% K = Ir;
% K(:,:,2) = Ig;
% K(:,:,3) = Ib;
% 
% imshow(K);
% 
% keyboard;



