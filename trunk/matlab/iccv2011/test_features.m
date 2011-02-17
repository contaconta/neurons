function test_features(filename, randstop)

%filename = 'violajones24x24.list';

if ~exist('randstop', 'var')
    randstop = 0;
end

IMSIZE = [24 24];
BLANK = zeros(IMSIZE); figure(1); imagesc(BLANK); colormap gray;
STOPRATE = 10000;

fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);

h = figure(1); colormap gray;
pos = get(h, 'Position');
set(h, 'Position', [pos(1), pos(2), 125 125]);


if randstop
    stop = randi(STOPRATE,1);
else
    stop = 0;
end

count = 0;
for i = 1:nfeatures
    tline = fgetl(fid);

    f = str2num(tline); %#ok<ST2NM>
    BW = rectRender(f, IMSIZE, BLANK);
    
    %figure(1);
    
    if f(2) == 0
        imagesc(BW); 
    else
        cla;
        imagesc(BW);
    end
    
    if i == stop
        str = ['feature ' num2str(stop-1) ': [' tline ']'];
        disp(str);
        disp('');
        
        RANK = f(1);
        p = 2;
        for k = 1:RANK
            T = f(p);
            if T == 0
                w = f(p+1);
                x0 = f(p+2);
                y0 = f(p+3);
                x1 = f(p+4);
                y1 = f(p+5);

                pol = sign(w);

                disp(['w_' num2str(k-1) ' = ' num2str(w)]);
                disp(['x0_' num2str(k-1) ' = ' num2str(x0)]);
                disp(['y0_' num2str(k-1) ' = ' num2str(y0)]);
                disp(['x1_' num2str(k-1) ' = ' num2str(x1)]);
                disp(['y1_' num2str(k-1) ' = ' num2str(y1)]);
                disp(' ');
                p = p + 6;
            else
                disp('45 degree feature');
                w = f(p+1);
                pol = sign(w);
                A = [ f(p+2)+1 f(p+3)+1];
                B = [ f(p+4)+1 f(p+5)+1];
                C = [ f(p+6)+1 f(p+7)+1];
                D = [ f(p+8)+1 f(p+9)+1];
                
                disp(['w_' num2str(k-1) ' = [' num2str(w) ' ]' ]);
                disp(['A_' num2str(k-1) ' = [' num2str(A) ' ]' ]);
                disp(['B_' num2str(k-1) ' = [' num2str(B) ' ]' ]);
                disp(['C_' num2str(k-1) ' = [' num2str(C) ' ]' ]);
                disp(['D_' num2str(k-1) ' = [' num2str(D) ' ]' ]);
                disp(' ');
                p = p + 10;
                
                
            end
        end
        disp('----------------------');
        
        clf;
        figure(1);
        imagesc(BW); 
        BW = rectRender(f, IMSIZE, BLANK);
        %pause;
        keyboard;

        stop = stop + randi(STOPRATE,1);
    end
    
    drawnow;
    
    
end

fclose(fid);