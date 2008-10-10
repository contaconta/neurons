function detections = ada_face_detect(I, DETECTOR, DELTA, SCALE_FACTOR, D)
%
%
%
%
%
%

if strcmp(class(I), 'uint8')
    I = mat2gray(I);
end

if strcmp(DETECTOR(1).type, 'CASCADE');
    FACESIZE = DETECTOR(1).CLASSIFIER.IMSIZE;
    classify = @ada_classify_cascade;
else
    FACESIZE = CLASSIFIER.IMSIZE;
    classify = @ada_classify_strong;
end


% decide if we want to display or not
D =1;


MAX_SCALE = max([size(I,1)/FACESIZE(1) size(I,2)/FACESIZE(2)]);
SCALES = 1;
while SCALES(length(SCALES)) < MAX_SCALE
    SCALES(length(SCALES) + 1) = SCALES(length(SCALES)) * SCALE_FACTOR;
end
SCALES = 1./SCALES;

%I = imadjust(I);
%I = adapthisteq(I, 'NumTiles', round(size(I)/25), 'NBins', 300);
detector_hits = zeros(10000,4); count = 0;

tic;


for s = SCALES
    I2 = imresize(I, s);
    s_actual = size(I2,1) / size(I,1);
    II = integral_image(I2);  
    disp(['scale = ' num2str(1/s_actual)]);
    W = size(I2,2);  H = size(I2,1);
    
    base_points = [1 1; 1 size(I,2); size(I,1) 1; size(I)];
    input_points = [1 1; 1 size(I2,2); size(I2,1) 1; size(I2)];
    %t = cp2tform(input_points,base_points,'projective');
    t = maketform('projective',input_points,base_points);
    
    for x = 1:max(1,round(DELTA*s_actual)):W - FACESIZE(1)
        for y = 1:max(1,round(DELTA*s_actual)):H - FACESIZE(2)
            
            IA = I2(y:y+FACESIZE(2)-1, x:x + FACESIZE(1) -1);
            IA = imnormalize('image', IA);
            II = integral_image(IA);
            II = II(:);
            C = classify(DETECTOR,  II, [x-1 y-1]);
            
            %C = classify(DETECTOR,  II, [x-1 y-1]);
           
            
            
           % C = 0;
%             if (x == 1) && (y == 1)
%                 x2 = x + FACESIZE(2);
%                 y2 = y + FACESIZE(1);
%                 [i1 j1] = tformfwd(t, x, y);
%                 [i2 j2] = tformfwd(t, x2, y2);
% 
%                 count = count + 1;
%                 detector_hits(count,:) = [i1 j1 j2 j2];                    
%             end
%             
%             
%             if (x == W - FACESIZE(1)) && (y == H - FACESIZE(2))
%                 x2 = x + FACESIZE(2);
%                 y2 = y + FACESIZE(1);
%                 %disp(['x2 y2 = ' num2str([x2 y2]) '  size(I2) = ' num2str(size(I2) ) ]);
% 
%                 [i1 j1] = tformfwd(t, x, y);
%                 [i2 j2] = tformfwd(t, x2, y2);
%                 disp(['x2 y2 = ' num2str(x2) ' ' num2str(y2)  '   i2 j2 = ' num2str(i2) ' ' num2str(j2) '  size(I2) = ' num2str(size(I2) ) ]);
%                 tforminv(t,size(I,2), size(I,1))
%                 
%                 count = count + 1;
%                 detector_hits(count,:) = [i1 j1 i2 j2];
%             end
            
            if C ==1
                x2 = x + FACESIZE(2);
                y2 = y + FACESIZE(1);
                [i1 j1] = tformfwd(t, x, y);
                [i2 j2] = tformfwd(t, x2, y2);
                count = count + 1;
                detector_hits(count,:) = [i1 j1 i2 j2];
            end
        end
    end
end
toc
if count > 1
    detector_hits = detector_hits(1:count,:);
    detections = group_detections(detector_hits);
elseif count == 1
    detections = detector_hits(1,:);
else
    detections = [];
end



if D
    figure; imshow(I);
    for i = 1:count
        x = detector_hits(i,1); y = detector_hits(i,2); x2 = detector_hits(i,3); y2 = detector_hits(i,4);
        a(i) = line([x x2 x2 x2 x2 x x x], [y y y y2 y2 y2 y2 y]);
        set(a(i), 'Color', [0 0 1], 'LineWidth', 0.5, 'LineStyle', '-.');
    end
    
    for i = 1:size(detections,1)
        x = detections(i,1); y = detections(i,2); x2 = detections(i,3); y2 = detections(i,4);
        b(i) = line([x x2 x2 x2 x2 x x x], [y y y y2 y2 y2 y2 y]);
        set(b(i), 'Color', [1 0 0], 'LineWidth', 2);
    end
end





function grouped_detections = group_detections(d)

OVERLAP_PERCENTAGE = .75;

detections.mean = d(size(d,1),:)';
detections.list = d(size(d,1),:);


for i = size(d,1)-1:-1:1
    grouped = 0;
    
    for j = 1:length(detections)
        if overlap(d(i,:), detections(j).mean) > (OVERLAP_PERCENTAGE * (d(i,3) - d(i,1)) * (d(i,4) - d(i,2)))
            detections(j).list(size(detections(j).list, 1)+1,:) = d(i,:);
            detections(j).mean = mean(detections(j).list,1)';
            grouped = 1;
            break;
        end
    end
    
    if grouped == 0
        detections(length(detections)+1).mean = d(i,:)';
        detections(length(detections)).list = d(i,:);
    end
    
end

grouped_detections = [detections(:).mean]';

