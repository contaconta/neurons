function create_violajonesGaussians(IMSIZE, filename, ROTATABLE)

N       = 16;   % max number of gaussians per row
M       = 16;   % max num of gaussians per column
Kmax    = 32;   % max total number of gaussians

AspectMax = 4;
Sigmas = [.5 1:8];

cnt = 1;


fid = fopen(filename, 'w');
disp(['...writing to ' filename]);


for sind = 1:length(Sigmas)
    
    x0 = 1;
    y0 = 1;
    
    d = 2*Sigmas(sind);
    
    
    
    % rank 2 horiz
    for n = 1:N+1
        for m = 1:M+1
            count = 1;
            x = []; y = []; s = []; w = [];

            % check that we have a valid aspect ratio
            aspect = n/m;
            if ((aspect > 1/AspectMax ) && (aspect < AspectMax)) && ((n > 4) || (m > 4))
                disp('bad aspect ratio!');
                continue;
            end
            
            % create the white part
            for r = 1:n
                for c = 1:m
                    x(count) = x0 + (c-1)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = 1;
                    count = count + 1;
                end
            end
            
            % create the black part
            xb = x;
            yb = y + n*d;
            sb = s;
            wb = -1*w;
            
            x = [x xb]; y = [y yb]; s = [s sb]; w = [w wb];

            xc = mean(x);
            yc = mean(y);
            
            if numel(x) > Kmax
                disp('too many gaussians!');
                continue;
            end
            if ~checkisvalid(x,y,IMSIZE)
                disp('out of bounds!');
                continue;
            end
            
            % scan the feature over the image
            for r = 0:(IMSIZE(1)-max(y))
                for c = 0:(IMSIZE(2)-max(x))
                 	X = x+c-1;
                    Y = y+r-1;
                    XC = xc+c-1;
                    YC = yc+r-1;
                    writeToFeatureList(X,Y,s,w,XC,YC,fid,ROTATABLE);
                    displayfunction(X,Y,s,w,XC,YC,IMSIZE,cnt);
                    cnt = cnt + 1;
                end
            end
            
        end
    end
       
    
    % rank 2 vert
    for n = 1:N+1
        for m = 1:M+1
            count = 1;
            x = []; y = []; s = []; w = [];

            % check that we have a valid aspect ratio
            aspect = n/m;
            if ((aspect > 1/AspectMax ) && (aspect < AspectMax)) && ((n > 4) || (m > 4))
                disp('bad aspect ratio!');
                continue;
            end
            
            % create the white part
            for r = 1:n
                for c = 1:m
                    x(count) = x0 + (c-1)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = 1;
                    count = count + 1;
                end
            end
            
            % create the black part
            xb = x + m*d;
            yb = y;
            sb = s;
            wb = -1*w;
            
            x = [x xb]; y = [y yb]; s = [s sb]; w = [w wb];

            xc = mean(x);
            yc = mean(y);
            
            if numel(x) > Kmax
                disp('too many gaussians!');
                continue;
            end
            if ~checkisvalid(x,y,IMSIZE)
                disp('out of bounds!');
                continue;
            end
            
          	% scan the feature over the image
            for r = 0:(IMSIZE(1)-max(y))
                for c = 0:(IMSIZE(2)-max(x))
                    X = x+c-1;
                    Y = y+r-1;
                    XC = xc+c-1;
                    YC = yc+r-1;
                    writeToFeatureList(X,Y,s,w,XC,YC,fid,ROTATABLE);
                    displayfunction(X,Y,s,w,XC,YC,IMSIZE,cnt);
                    cnt = cnt + 1;
                end
            end
        end
    end
    
    
    % rank 3 horiz
    for n = 1:N+1
        for m = 1:M+1
            count = 1;
            x = []; y = []; s = []; w = [];

            % check that we have a valid aspect ratio
            aspect = n/m;
            if ((aspect > 1/AspectMax ) && (aspect < AspectMax)) && ((n > 4) || (m > 4))
                disp('bad aspect ratio!');
                continue;
            end
            
            % create the white part
            for r = 1:n
                for c = 1:m
                    x(count) = x0 + (c-1)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = 1;
                    count = count + 1;
                end
            end
            
            % create the black part
            xb = x;
            yb = y + n*d;
            sb = s;
            wb = -2*w;
            
            % create another white part
            xw2 = x;
            yw2 = y + 2*n*d;
            sw2 = s;
            ww2 = 1*w;
            
            x = [x xb xw2]; y = [y yb yw2]; s = [s sb sw2]; w = [w wb ww2];

            xc = mean(x);
            yc = mean(y);
            
            if numel(x) > Kmax
                disp('too many gaussians!');
                continue;
            end
            if ~checkisvalid(x,y,IMSIZE)
                disp('out of bounds!');
                continue;
            end
            
            % scan the feature over the image
            for r = 0:(IMSIZE(1)-max(y))
                for c = 0:(IMSIZE(2)-max(x))
                    X = x+c-1;
                    Y = y+r-1;
                    XC = xc+c-1;
                    YC = yc+r-1;
                    writeToFeatureList(X,Y,s,w,XC,YC,fid,ROTATABLE);
                    displayfunction(X,Y,s,w,XC,YC,IMSIZE,cnt);
                    cnt = cnt + 1;
                end
            end
        end
    end
    
    
    % rank 3 vert
    for n = 1:N+1
        for m = 1:M+1
            count = 1;
            x = []; y = []; s = []; w = [];

            % check that we have a valid aspect ratio
            aspect = n/m;
            if ((aspect > 1/AspectMax ) && (aspect < AspectMax)) && ((n > 4) || (m > 4))
                disp('bad aspect ratio!');
                continue;
            end
            
            % create the white part
            for r = 1:n
                for c = 1:m
                    x(count) = x0 + (c-1)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = 1;
                    count = count + 1;
                end
            end
            
            % create the black part
            xb = x + m*d;
            yb = y;
            sb = s;
            wb = -2*w;
            
            % create another white part
            xw2 = x + 2*m*d;
            yw2 = y;
            sw2 = s;
            ww2 = 1*w;
            
            x = [x xb xw2]; y = [y yb yw2]; s = [s sb sw2]; w = [w wb ww2];

            xc = mean(x);
            yc = mean(y);
            
            if numel(x) > Kmax
                disp('too many gaussians!');
                continue;
            end
            if ~checkisvalid(x,y,IMSIZE)
                disp('out of bounds!');
                continue;
            end
            
            % scan the feature over the image
            for r = 0:(IMSIZE(1)-max(y))
                for c = 0:(IMSIZE(2)-max(x))
                    X = x+c-1;
                    Y = y+r-1;
                    XC = xc+c-1;
                    YC = yc+r-1;
                    writeToFeatureList(X,Y,s,w,XC,YC,fid,ROTATABLE);
                    displayfunction(X,Y,s,w,XC,YC,IMSIZE,cnt);
                    cnt = cnt + 1;
                end
            end
        end
    end
    
    
    % rank 4 
    for n = 2:2:N+1
        for m = 2:2:M+1
            count = 1;
            x = []; y = []; s = []; w = [];

            % check that we have a valid aspect ratio
            aspect = n/m;
            if ((aspect > 1/AspectMax ) && (aspect < AspectMax)) && ((n > 4) || (m > 4))
                disp('bad aspect ratio!');
                continue;
            end
            
            % create the white part
            for r = 1:n/2
                for c = 1:m/2
                    x(count) = x0 + (c-1)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = 1;
                    count = count + 1;
                    x(count) = x0 + (c-1)*d + (m/2)*d;
                    y(count) = y0 + (r-1)*d;
                    s(count) = Sigmas(sind);
                    w(count) = -1;
                    count = count + 1;
                end
            end
            
            % create the reverse part
            xb = x;
            yb = y + (n/2)*d;
            sb = s;
            wb = -1*w;
            
            x = [x xb]; y = [y yb]; s = [s sb]; w = [w wb];

            xc = mean(x);
            yc = mean(y);
            
            if numel(x) > Kmax
                disp('too many gaussians!');
                continue;
            end
            if ~checkisvalid(x,y,IMSIZE)
                disp('out of bounds!');
                continue;
            end
            
            % scan the feature over the image
            for r = 0:(IMSIZE(1)-max(y))
                for c = 0:(IMSIZE(2)-max(x))
                    X = x+c-1;
                    Y = y+r-1;
                    XC = xc+c-1;
                    YC = yc+r-1;
                    writeToFeatureList(X,Y,s,w,XC,YC,fid,ROTATABLE);
                    displayfunction(X,Y,s,w,XC,YC,IMSIZE,cnt);
                    cnt = cnt + 1;
                end
            end
        end
    end
    
    
    % center-surround


    %% display

    


end


disp('=======================================');
disp(' ');
disp('IMPORTANT!!!!');
disp(' ');
disp(['WRITE THE NUMBER OF FEATURES (' num2str(cnt -1) ') AT THE TOP OF']);
disp(filename);
disp(' ');
disp('=======================================');

fclose(fid);







function writeToFeatureList(x,y,s,w,xc,yc,fid,ROTATABLE)

appendSparseFeature(fid,x,y,w,s,xc,yc,ROTATABLE,0);

% n = length(x);
% f = [num2str(n) sprintf(' %f %f', xc, yc)];
% for k = 1:n        
%     f = [f sprintf(' %f %f %d %d',  w(k), s(k), x(k), y(k) )]; %#ok<*AGROW>
% end
% fprintf(fid, [f '\n']);



function displayfunction(x,y,s,w,xc,yc,IMSIZE,count)

if mod(count, 500) == 0
    %[x(:) y(:) s(:) w(:)]    
    R1 = reconstruction_coarse(IMSIZE,x,y,w,s,0);
    %subplot(1,2,1); 
    imagesc(R1, [-max(abs(R1(:))) max(abs(R1(:)))]);  colormap gray; title('5sigma + 1');
    hold on;
    plot(x(w > 0)+1, y(w > 0)+1, 'rs');
    plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
    plot(xc+1,yc+1, 'mo'); hold off;

    %R2 = reconstruction(IMSIZE,x,y,w,s);  % analytic
    %R2 = reconstruction_coarse(IMSIZE,x,y,w,s);
    %subplot(1,2,2); imagesc(R2, [-max(abs(R2(:))) max(abs(R2(:)))]);  colormap gray; title('2sigma + 1');
    drawnow;
    %pause;
end


function ok = checkisvalid(x,y,IMSIZE)

%ok = 0;

ok = (x >= 1) & (x <= IMSIZE(2)) & (y >= 1) & (y <= IMSIZE(1));

if ~isempty(find(~ok, 1))
    ok = 0;
else
    ok = 1;
end
