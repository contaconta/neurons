function [] = generateViperFileFromFirstFrame(folder_n, resultsFolder, exp_num, ReferenceFile)


%% some parameters
sigma_red = 2;
minArea   = 50;
maxArea   = 155;


Rfolder = [folder_n 'red/'];
Rfiles = dir([Rfolder '*.TIF']);

TMAX = length(Rfiles);
%%
Rt = imread([folder_n 'red/' Rfiles(1).name]);
Rt = mat2gray(double(Rt));


Rblur = imgaussian(Rt, sigma_red);
I = Rblur;
I = uint8(255*mat2gray(I));

M = vl_mser(I, 'MinDiversity', minArea/maxArea,...
    'MaxVariation', 0.25,...
    'MinArea', minArea/numel(I), ...
    'MaxArea', maxArea/numel(I), ...
    'BrightOnDark', 1, ...
    'Delta',2) ;

mm = zeros(size(Rt));
for x = M'
    s = vl_erfill(I, x);
    mm(s) = mm(s)+1;
end
M = mm > 0;
M  	= imfill(M, 'holes');
M = bwlabel(M);

Circles = [];
count = 1;
detections_t = regionprops(M, 'Area', 'Centroid', 'Eccentricity');  %#ok<*MRPBW>
% add some measurements, create a list of detections
if ~isempty(detections_t)
    for i = 1:length(detections_t)
        if detections_t(i).Eccentricity < 0.90
            detections_t(i).Radius = sqrt(detections_t(i).Area/ pi);
            Circles{count} = detections_t(i);%#ok
            count = count + 1;
        end
    end
end
%%
ST = xml2struct(ReferenceFile);
outputAnnotFile = [resultsFolder exp_num];


%% first change the associated filenames

ST.viper.data.sourcefile{1}.Attributes.filename = ['./' exp_num 'OriginalGreen.mpg'];
ST.viper.data.sourcefile{2}.Attributes.filename = ['./' exp_num 'OriginalRed.mpg'];



for i =1:2
    ST.viper.data.sourcefile{i}.object = cell(size(Circles));
    for j =1:length(Circles)
        ST.viper.data.sourcefile{i}.object{j}.Attributes.id         = int2str(j-1);
        ST.viper.data.sourcefile{i}.object{j}.Attributes.name       = 'Cell';
        ST.viper.data.sourcefile{i}.object{j}.Attributes.framespan  = ['1:' int2str(TMAX)];
        
        ST.viper.data.sourcefile{i}.object{j}.attribute.Attributes.name  = 'Nucleus';
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle  = cell(1);
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle{1}.Text = '';
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle{1}.Attributes.framespan = ['1:' int2str(TMAX)];
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle{1}.Attributes.radius = num2str(round(Circles{j}.Radius));
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle{1}.Attributes.x = num2str(round(Circles{j}.Centroid(1)));
        ST.viper.data.sourcefile{i}.object{j}.attribute.data_colon_circle{1}.Attributes.y = num2str(round(Circles{j}.Centroid(2)));
        
    end
    
end

struct2xml(ST, outputAnnotFile);