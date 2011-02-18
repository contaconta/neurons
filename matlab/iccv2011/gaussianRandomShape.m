function  [X Y W SIGMA] = gaussianRandomShape(IMSIZE, sigmas, k, Npos, Nneg, REFLECT, CROSS, sigmaO)
% Npos = # of pos paths
% Nneg = # of neg paths
% k = # of gaussians per path

DISPLAY = 0;
warning('off','MATLAB:polyfit:PolyNotUnique');

xmax = IMSIZE(2);
ymax = IMSIZE(1);
numScales = numel(sigmas);
M = true(IMSIZE);
X = []; Y = []; W = []; SIGMA = [];

if ~exist('sigmaO', 'var')
    sigmaO = pi/6;
end

if ~exist('CROSS', 'var')
    CROSS = .1;
end

polaritylist = [ones(1,Npos), -1*ones(1,Nneg)];
polaritylist = polaritylist( randperm(numel(polaritylist)));

for i = 1:Npos+Nneg
    
    if i > 1
        if rand < REFLECT
            % reflect the previous curve
            [M,x,y,w,s]  = reflectPath(x,y,s,w,IMSIZE,M);
            %disp('reflected');
        elseif (rand < CROSS) && (numel(x) > 1)
            [M,x,y,w,s]  = crossPath(x,y,s,w,IMSIZE,M);
            %disp('crossed');
        else
            % generate a new curve
            [M,x,y,w,s] = generatecurve(M, sigmas, k, xmax, ymax, numScales, IMSIZE, sigmaO);
            %disp('not refl');
        end
        
    else
        % we must generate a new curve on 1st iteration
        [M, x, y, w, s] = generatecurve(M, sigmas, k, xmax, ymax, numScales, IMSIZE, sigmaO);     
    end
    
    %[M, x, y, w, s] = generatecurve(M, sigmas, k, xmax, ymax, numScales, IMSIZE); %#ok<NASGU,ASGLU>
    w = w * polaritylist(i);
    X = [X; x-1]; %#ok<*AGROW>
    Y = [Y; y-1]; %!!!!! IMPORTANT zero-indexing for c++ !!!!!!!
    W = [W; w];
    SIGMA = [SIGMA; s];
end


% BAD!!! mean center according to area (assume 2*sigma radius)
% winds = (W > 0);
% binds = (W <= 0);
% warea = sum( 4*pi*SIGMA(winds).^2);
% barea = sum( 4*pi*SIGMA(binds).^2);
% W(winds) = W(winds)/warea;
% W(binds) = W(binds)/barea;

%[X Y W SIGMA]



% ================== DISPLAY =========================
if DISPLAY
    R = reconstruction(IMSIZE, X, Y, W, SIGMA);
    imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray;
    drawnow;
end
%pause;




















function [M, x, y, w, sigma] = generatecurve(M, sigmas, k, xmax, ymax, numScales, IMSIZE, orientationvariance)

%orientationvariance = pi/8;
%orientationvariance = pi/6;
MAXITER = 100;

scaleprobs = 1 ./ ( (sigmas+1).^2);
%scaleprobs(sigmas < 1) = median(scaleprobs);
%scaleprobs(sigmas < 1) = scaleprobs(sigmas == 1);

%keyboard;

x = zeros(k,1); y = x; s = x; O = x; sigma = x; w = x;

% first, decide an initial point
i = 1;
validinds = find(M);
if ~isempty(validinds)
    ind = randsample(validinds, 1);
else
    disp('error: no valid indexes remain!');
    x = []; y = []; w = []; sigma = []; 
    return;
end
[y(i) x(i)] = ind2sub(IMSIZE, ind);
%x(i) = randi(xmax,1);
%y(i) = randi(ymax,1);
s(i) = randsample(1:numScales,1, 'true', scaleprobs);
O(i) = pi*2*rand(1);
sigma(1) = sigmas(s(1));

% assign weights accoring to area
w(i) = 1/scaleprobs(s(i));
w(i) = 1;

M = updatemask(M, x(i),y(i),sigma(i));
%disp(['x: ' num2str(x(i)) ' y: ' num2str(y(i)) ' s: ' num2str(sigma(i))]);    
%imagesc(M); colormap gray;
%keyboard;

for i = 2:k
    
    % first, select a scale at random prom prob distribution
    idxlist = [s(i-1)-1 s(i-1) s(i-1)+1];
    idxlist(idxlist == 0) = 1;
    idxlist(idxlist > numScales) = numel(sigmas);
    s(i) = randsample(idxlist,1);
    sigma(i) = sigmas(s(i));
    
    % assign weights accoring to area
    w(i) = 1/scaleprobs(s(i));
    w(i) = 1;
    
    %keyboard;
    
    iter = 1;
    while iter < MAXITER
    
        % randomly select an angle
        o = orientationvariance*randn(1) + O(i-1);        
        
        % find the point at s(i-1) + s(i)
        d = (sigma(i-1) + sigma(i));   % NOTE: maybe put 2*
        d = max([1 d]);
        u = x(i-1) + round(cos(o)*d);
        v = y(i-1) + round(sin(o)*d);
        
        
        if (u > 0) && (u <= xmax) && (v > 0) && (v <= ymax)
            valid = M(v,u);
            if valid
                x(i) = u;
                y(i) = v;
                O(i) = o;
                %disp(['orientation = ' num2str(rad2deg(o))]);
                break;
            end
        else
            %disp('tried to generate an invalid point!');
        end
        iter = iter + 1;
        
        if iter == MAXITER
            x = x(1:i-1);
            y = y(1:i-1);
            sigma = sigma(1:i-1);
            w = w(1:i-1);
            %disp('returned because no valid point found!');
            w = w./sum(w);
            return;
        end
    end
    
    % update the mask
    M = updatemask(M, x(i),y(i),sigma(i));
    
    %disp(['x: ' num2str(x(i)) ' y: ' num2str(y(i)) ' s: ' num2str(sigma(i))]);
    
    %imagesc(M); colormap gray;
    

end
w = w./sum(w);

    
%keyboard;






function M = updatemask(M, x, y, s)

[i j] = meshgrid(1:size(M,1), 1:size(M,2));

d = sqrt((i - x).^2 + (j - y).^2);

M(d < s) = false;




function TF = enoughInside(x2,y2, IMSIZE)

TF = 0;
ok = (x2 >= 1) & (x2 <= IMSIZE(2)) & (y2 >= 1) & (y2 <= IMSIZE(1));

if (length(find(ok)) / numel(x2)) > .5
    TF = 1;
end


function TF = isInside(x,y, IMSIZE)

TF = (x >= 1) & (x <= IMSIZE(2)) & (y >= 1) & (y <= IMSIZE(1));




%keyboard;


function [M,X,Y,W,S]  = reflectPath(x,y,s,w,IMSIZE,M)

reflectSigma = 3;

xc = mean(x);
yc = mean(y);          
smean = mean(s);

x = x - xc;
y = y - yc;

d = 2*smean + reflectSigma*smean*abs(randn(1));
d = max(1, d);

p = polyfit(x,y,1);
slope = -(1/p(1));
dy = randsample([-1 1],1)*sqrt( d^2 / (1 + (1/slope)^2));
dx = dy/slope;

x = x + xc;
y = y + yc;
x2 = x + dx;
y2 = y + dy;

iter = 1;
while ~enoughInside(x2,y2, IMSIZE)     
    x2 = x + dx;
    y2 = y + dy;

    d = 2*smean + reflectSigma*smean*abs(randn(1)); d = max(1,d);
    angle = 2*pi*rand(1);
    dx = cos(angle)*d;
    dy = sin(angle)*d;
    %disp('randomly selected a new angle');
    iter = iter + 1;
    if iter > 1000
        disp('too many iterations to reflect!');
        %keyboard;
        break;
    end
    if iter > 100
        smean = 2;
    end
end
% figure(1); clf; hold on;
% axis([0 25 0 25]);
% plot(x,y,'bs');
% plot(x2,y2,'ro');
%             f = polyval(p,1:25); 
%             plot(1:25, f, 'g.');

count = 1;
for i = 1:length(x2)
    if isInside(round(x2(i)),round(y2(i)), IMSIZE)
        X(count) = round(x2(i));
        Y(count) = round(y2(i));
        S(count) = s(i);
        W(count) = abs(w(i));
        M = updatemask(M, X(count),Y(count),S(count));
        count = count + 1;
    end
end

if ~isempty(W)
    W = W./sum(W);
    W = W(:);
    S = S(:);
    X = X(:);
    Y = Y(:);
else
    X = [];
    Y = [];
    S = [];
    W = [];
end




function [M,X,Y,W,S]  = crossPath(x,y,s,w,IMSIZE,M)


xc = mean(x);
yc = mean(y);          

x0 = x - xc;
y0 = y - yc;

%Theta = pi* randi([0 7]) /2;
%validangles = [pi/4 pi/2 .75*pi 1.25*pi 1.5*pi 1.75*pi];
validangles = [pi/2 1.5*pi];  
Theta = randsample(validangles, 1);

R = [cos(Theta) -sin(Theta); sin(Theta) cos(Theta)];

for i =1:length(x0)
    a = R*[x0(i); y0(i)];
    xR(i) = a(1);
    yR(i) = a(2);
end

x2 = xR + xc;
y2 = yR + yc;

iter = 1;
while ~enoughInside(x2,y2, IMSIZE)     

    x0 = x - xc;
    y0 = y - yc;

    %Theta = pi* randi([0 7]) /2;
    Theta = randsample([pi/4 pi/2 .75*pi 1.25*pi 1.5*pi 1.75*pi], 1);

    R = [cos(Theta) -sin(Theta); sin(Theta) cos(Theta)];

    for i =1:length(x0)
        a = R*[x0(i); y0(i)];
        xR(i) = a(1);
        yR(i) = a(2);
    end

    x2 = xR + xc;
    y2 = yR + yc;

    %disp('randomly selected a new angle');
    iter = iter + 1;
    if iter > 1000
        %disp('too many iterations to cross!');
        %keyboard;
        break;
    end
end
% figure(1); clf; hold on;
% axis([0 25 0 25]);
% plot(x,y,'bs');
% plot(x2,y2,'ro');
%          

count = 1;
for i = 1:length(x2)
    if isInside(round(x2(i)),round(y2(i)), IMSIZE)
        X(count) = round(x2(i));
        Y(count) = round(y2(i));
        S(count) = s(i);
        W(count) = abs(w(i));
        M = updatemask(M, X(count),Y(count),S(count));
        count = count + 1;
    end
end

if ~isempty(W)
    W = W./sum(W);
    W = W(:);
    S = S(:);
    X = X(:);
    Y = Y(:);
else
    X = [];
    Y = [];
    S = [];
    W = [];
end




