clear;


SIGMA = 4.5;


folder_GT = '/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/GTLudo/';
folder_det = '/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/Detections/';

folder_GTobj =  '/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/GTLudo_objPerCell/';
folder_detobj = '/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/Detections_objPerCell/';

folder_result = '/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/Results/';


d = dir('/home/ksmith/Dropbox/ExamplesPruning/NeuritesEvaluation/GTLudo/');
d = d(3:end);

f = figure; set(f, 'Position', [431 122 1087 826]);
set(gca, 'Position', [0 0 1 1]);



RESULTS = [];


% loop through the images
for i = 1:numel(d)

    % get the image
    folder = sprintf('%s%s/', folder_GT, d(i).name);
    filename = [folder 'green.tif'];
    fprintf('loading %s\n', filename);
    
    I = imread(filename);
    I = double(I);
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    
    
    hold off; cla;
    

    N = [];
    removelist = [];
    NMAX = 36;
    N(NMAX).GT_r = [];
    N(NMAX).GT_c = [];
    N(NMAX).DET_r = [];
    N(NMAX).DET_c = [];
    N(NMAX).GT_t = [];
    N(NMAX).GT_s = [];
    N(NMAX).DET_t = [];
    N(NMAX).DET_s = [];
    N(NMAX).FN_1 = [];
    N(NMAX).FN_2 = [];
    N(NMAX).FP_1 = [];
    N(NMAX).FP_2 = [];
    N(NMAX).name = [];
    
    
    
    % loop through all possible cells
    for n = 1:NMAX
        
        N(n).name = sprintf('%03d', n);
        
        % ground truth
        filename = [folder_GTobj sprintf('%03dID%03d.obj',i,n)];
        if exist(filename, 'file')
            obj = read_wobj(filename);
            clist = obj.vertices(:,1);
            rlist = obj.vertices(:,2);
                       
            N(n).GT_r = rlist+1;
            N(n).GT_c = clist+1;   
        end
        
        
        % detection 
        filename = [folder_detobj sprintf('%03dID%03d.obj',i,n)];
        if exist(filename, 'file')
            obj = read_wobj(filename);
            clist = obj.vertices(:,1);
            rlist = obj.vertices(:,2);
                       
            N(n).DET_r = rlist+1;
            N(n).DET_c = clist+1;
        end
        
        
        % compute the metrics for this cell
        [t s m1 m2] = evaluation_metric(N(n).GT_r, N(n).GT_c, N(n).DET_r, N(n).DET_c, SIGMA);
        N(n).GT_t = t;
        N(n).GT_s = s;
        N(n).FN_1 = m1;
        N(n).FN_2 = m2;
        [t s m1 m2] = evaluation_metric(N(n).DET_r, N(n).DET_c, N(n).GT_r, N(n).GT_c, SIGMA);
        N(n).DET_t = t;
        N(n).DET_s = s;
        N(n).FP_1 = m1;
        N(n).FP_2 = m2;
        
        

    end
    
    % remove cells with no GT or DET
%     for n = 1:NMAX
%         if isempty(N(n).GT_r) && isempty(N(n).DET_r)
%             removelist = [removelist; n];            
%         end
%     end
     for n = 1:NMAX
        if isempty(N(n).GT_r)
            removelist = [removelist; n];            
        end
    end
    N(removelist) = [];
    
    
    
    % render this frame
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    imshow(I);
    hold on;
    
    
    
    
    
    for n = 1:length(N)
        %fprintf('rendering n = %s (%d/%d)', N(n).name, n, length(N));

        % get the patch for the GT
        if ~isempty(N(n).GT_r)
            B = zeros(size(Ir));
            for k = 1:size(N(n).GT_r,1)
                B(N(n).GT_r(k), N(n).GT_c(k)) = 1;
            end
            D = bwdist(B);
            T = D < SIGMA;
            Ig(T) = .75 * Ig(T);
            Ib(T) = .75 * Ib(T); 
            %fprintf(' red ');
        end



       for k = 1:size(N(n).GT_r,1)
            Ir(N(n).GT_r(k), N(n).GT_c(k)) = 1;
            Ig(N(n).GT_r(k), N(n).GT_c(k)) = 0;
            Ib(N(n).GT_r(k), N(n).GT_c(k)) = 0;
        end
        
        % render
        for k = 1:size(N(n).DET_r,1)
            Ir(N(n).DET_r(k), N(n).DET_c(k)) = 0;
            Ig(N(n).DET_r(k), N(n).DET_c(k)) = 1;
            Ib(N(n).DET_r(k), N(n).DET_c(k)) = 1;
        end
        
        
        %fprintf('\n');


    end
    
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    imshow(I);
    
   
    for n = 1:length(N)
            % text
        if ~isempty(N(n).GT_r)
            xloc = quantile(N(n).GT_c, .80);
            yloc = quantile(N(n).GT_r, .80);
        else
            xloc = quantile(N(n).DET_c, .80);
            yloc = quantile(N(n).DET_r, .80);
        end
        %str1 = sprintf('FP = %0.2f  %0.2f\nFN = %0.3f  %0.3f', N(n).FP_1, N(n).FP_2, N(n).FN_1, N(n).FN_2);
        
        str0 = sprintf('%s', N(n).name);
        str1 = sprintf('FP = %0.2f  %0.2f', N(n).FP_1, N(n).FP_2);
        str2 = sprintf('FN = %0.2f  %0.2f', N(n).FN_1, N(n).FN_2);
        t0 = text(xloc, yloc-12, str0);
        t1 =text(xloc,yloc, str1);
        t2 = text(xloc,yloc+12, str2);
        set(t0, 'Color', [.3 .3 .3]);
        set(t2, 'Color', [1 .2 .2]);
        set(t1, 'Color', [.2 1 1]);
        
    end
    
    mean_FN_1 = mean([N(:).FN_1]);    
    mean_FN_2 = mean([N(:).FN_2]);  
    mean_FP_1 = mean([N(:).FP_1]);    
    mean_FP_2 = mean([N(:).FP_2]); 
    
    % add cells to the results
    for n = 1:length(N)
        RESULTS(end+1, :) = [N(n).FN_1 N(n).FN_2 N(n).FP_1 N(n).FP_2];
    end
    
    % text output
    for n = 1:length(N)
        fprintf('%s   FN = %0.3f   %0.3f    FP = %0.3f   %0.3f\n', N(n).name, N(n).FN_1, N(n).FN_2, N(n).FP_1, N(n).FP_2);        
    end    
    fprintf('mean FN = %0.3f   %0.3f      mean FP = %0.3f   %0.3f\n', mean_FN_1, mean_FN_2, mean_FP_1, mean_FP_2);

    
    % write the image file result
    F = getframe(gcf);
    I = F.cdata;
    filename = sprintf('%s%03d.png', folder_result, i);
    fprintf('writing %s\n', filename);
    imwrite(I, filename);
    
    keyboard;
    
end













     
%         % get the patch for the GT
%         if ~isempty(N(n).GT_r)
%             B = zeros(size(Ir));
%             for k = 1:size(N(n).GT_r,1)
%                 B(N(n).GT_r(k), N(n).GT_c(k)) = 1;
%             end
%             D = bwdist(B);
%             T = D < SIGMA;
%             BW = bwboundaries(T, 8, 'noholes');            
%             x1 = BW{1}(:,2);
%             y1 = BW{1}(:,1);
%             [x2, y2] = poly2cw(x1, y1);
%             p = patch(x2, y2, 1, 'FaceColor', [1 0 0], 'FaceAlpha', .5, 'EdgeColor', [1 0 0], 'EdgeAlpha', 1, 'LineWidth', .5);
%             fprintf(' red ');
%         end
          
        
%          % get the patch for the GT
%         if ~isempty(N(n).GT_r)
%             B = zeros(size(Ir));
%             for k = 1:size(N(n).GT_r,1)
%                 B(N(n).GT_r(k), N(n).GT_c(k)) = 1;
%             end
%             BW = bwboundaries(B, 8, 'noholes');            
%             x1 = BW{1}(:,2);
%             y1 = BW{1}(:,1);
%             [x2, y2] = poly2cw(x1, y1);
%             p = patch(x2, y2, 1, 'FaceColor', [1 0 0], 'FaceAlpha', .5, 'EdgeColor', [1 0 0], 'EdgeAlpha', 1, 'LineWidth', .5);
%         end
            
        
%          % get the patch for the DET
%         if ~isempty(N(n).DET_r)
%             B = zeros(size(Ir));
%             for k = 1:size(N(n).DET_r,1)
%                 B(N(n).DET_r(k), N(n).DET_c(k)) = 1;
%             end
%             BW = bwboundaries(B, 8, 'noholes');            
%             x1 = BW{1}(:,2);
%             y1 = BW{1}(:,1);
%             [x2, y2] = poly2cw(x1, y1);
%             p = patch(x2, y2, 1, 'FaceColor', [0 1 0], 'FaceAlpha', .5, 'EdgeColor', [0 1 0], 'EdgeAlpha', 1, 'LineWidth', .5);
%             fprintf(' green ');
%         end
        

        
%         for j = 1:numel(N(n).GT_r)
%             plot(N(n).GT_c(j), N(n).GT_r(j), 'r.');
%         end
%         for j = 1:numel(N(n).DET_r)
%             plot(N(n).DET_c(j), N(n).DET_r(j), 'g.');
%         end



% 
% 
%             % render
%         for k = 1:size(N(n).GT_r,1)
%             Ir(N(n).GT_r(k), N(n).GT_c(k)) = 1;
%             Ig(N(n).GT_r(k), N(n).GT_c(k)) = 0;
%             Ib(N(n).GT_r(k), N(n).GT_c(k)) = 0;
%         end
%         
%         % render
%         for k = 1:size(N(n).DET_r,1)
%             Ir(N(n).DET_r(k), N(n).DET_c(k)) = 0;
%             Ig(N(n).DET_r(k), N(n).DET_c(k)) = 1;
%             Ib(N(n).DET_r(k), N(n).DET_c(k)) = 0;
%         end
%         
%         I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
%         imshow(I);
%         N(n)
%     
%     
%     
%     % get Ludo's ground truth
%     d1 = dir([folder_GTobj sprintf('%03dID*.obj', i)]);
%     
%     % loop through detected cells
%     for j = 1:numel(d1)
%         filename = [folder_GTobj d1(j).name];
%         obj = read_wobj(filename);
%         clist = obj.vertices(:,1);
%         rlist = obj.vertices(:,2);
%         
%         GT(i).neurite(j).r = rlist+1;
%         GT(i).neurite(j).c = clist+1;
%          
%      
%         for k = 1:size(obj.vertices,1)
%             Ir(GT(i).neurite(j).r(k), GT(i).neurite(j).c(k)) = 1;
%             Ig(GT(i).neurite(j).r(k), GT(i).neurite(j).c(k)) = 0;
%             Ib(GT(i).neurite(j).r(k), GT(i).neurite(j).c(k)) = 0;
%         end
%         
%         
%         I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
% 
% 
% 
%     end
% 
%     
%                          imshow(I);
%          pause(0.1); hold on;
%          
% 
% end
% 
% 
% 
% 
% 
