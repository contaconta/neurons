function Segmentation = SGAC(InputImage, ImageSpacing, center, radius, SigmoidWeight)
%
% Segmentation using the Geodesic Active Contour model
%
% Given an intial seed point and a radius, a sphere is shrinked until some
% convergence criterion is reached. radius should be large enough so the
% sphere includes completely the object of interest.
%
% The crutial part of this function is the stopping metric function.
% The stopping metric function is a sigmoid function of the image
% intensities. First, an estimate of sigmoid function parameters alpha and
% beta is done using the image intesities inside the given initial sphere.
% $SigmoidWeight \in [0, 1]$ controles beta. if the obtained segmentation
% overfloews out of the object of interest, SigmoidWeight should be
% increased. A default value to start with might be 0.7 
%
%
% Author: F. Benmansour 2011, CVLab, EPFL


sz = size(InputImage);
% Pad the image if needed
% on x direction
XminPad = 0;
XmaxPad = 0;
if (center(1)-radius/ImageSpacing(1)) < 3
    XminPad = radius/ImageSpacing(1)-center(1) + 4;
end

if (center(1)+radius/ImageSpacing(1)) > (size(InputImage, 1) - 3)
    XmaxPad = center(1) + radius + 5 - size(InputImage, 1);
end

% on y direction
YminPad = 0;
YmaxPad = 0;
if (center(2)-radius/ImageSpacing(2)) < 3
    YminPad = radius/ImageSpacing(2)-center(2) + 4;
end

if (center(2)+radius/ImageSpacing(2)) > (size(InputImage, 3) - 3)
    YmaxPad = center(2) + radius + 5 - size(InputImage, 2);
end

% on z direction

ZminPad = 0;
ZmaxPad = 0;
if (center(3)-radius/ImageSpacing(3)) < 3
    ZminPad = radius/ImageSpacing(3)-center(3) + 4;
end

if (center(3)+radius/ImageSpacing(3)) > (size(InputImage, 3) - 3)
    ZmaxPad = center(3) + radius + 5 - size(InputImage, 3);
end

% compute the level set function that such that its zero
% levelset is a sphere of the given center and radius
% here, just to get the padding value
InitialLevelSet = matitk('SFM', 1e9, ones(size(InputImage)), [], center, ImageSpacing);
InitialLevelSet = InitialLevelSet - radius;
meanIntensityOutsideSphere = mean(InputImage(InitialLevelSet > 0));
meanIntensityInsideSphere  = mean(InputImage(InitialLevelSet <= 0));

PaddedImage = meanIntensityOutsideSphere*ones(size(InputImage) + ...
                                round([XminPad+XmaxPad, YminPad+YmaxPad, ZminPad+ZmaxPad]));
PaddedImage(round(XminPad+1):round(XminPad+sz(1)), ...
            round(YminPad+1):round(YminPad+sz(2)), ...
            round(ZminPad+1):round(ZminPad+sz(3))) = InputImage;

% first smooth the image witha sigma equal to the minimal image spacing
smoothingSigma = min(ImageSpacing);
SmoothedImage = matitk('FGA', [smoothingSigma, 5], PaddedImage, [], [], ImageSpacing);



% Estimating the sigmoid function parameters

InitialLevelSet = matitk('SFM', 1e9, ones(size(PaddedImage)), [], center -1 + [round(XminPad), round(YminPad), round(ZminPad)], ImageSpacing);
InitialLevelSet = InitialLevelSet - radius;
sigBeta = (255.0*SigmoidWeight) *( meanIntensityInsideSphere /  max(SmoothedImage(:)) );
sigAlpha = -1.0;

% compute the image intensity-baqsed sigmoid function
ObjectDetector = matitk('FSN', [0, 1, sigAlpha, sigBeta], 255.0*(double(SmoothedImage)/max(SmoothedImage(:))) );

% run the geodesic active contour segmentation
GAC_params = [-1.0, 0.1, 1.5, 0.01, 500];
Segmentation = matitk('SGAC', GAC_params, ObjectDetector, InitialLevelSet, [], ImageSpacing);


% crop the padded image
Segmentation = Segmentation(round(XminPad+1):round(XminPad+sz(1)), ...
            round(YminPad+1):round(YminPad+sz(2)), ...
            round(ZminPad+1):round(ZminPad+sz(3)));

% get rid of all the small connected componnents
CC = bwconncomp(Segmentation);
numPixelsPerComp = cellfun(@numel,CC.PixelIdxList);
[~,idx] = max(numPixelsPerComp);

for i =1:length(numPixelsPerComp)
   if(i ~= idx)
       Segmentation(CC.PixelIdxList{i}) = 0;
   end
end

return;


% function Segmentation = SGAC(InputImage, ImageSpacing, center, radius, SigmoidWeight)
% %
% % Segmentation using the Geodesic Active Contour model
% %
% % Given an intial seed point and a radius, a sphere is shrinked until some
% % convergence criterion is reached. raidus should be large enough so the
% % sphere includes completely the object of interest.
% %
% % The crutial part of this function is the stopping metric function.
% % The stopping metric function is a sigmoid function of the image
% % intensities. First, an estimate of sigmoid function parameters alpha and
% % beta is done using the image intesities inside the given initial sphere.
% % $SigmoidWeight \in [0, 1]$ controles beta. if the obtained segmentation
% % overfloews out of the object of interest, SigmoidWeight should be
% % increased. A default value to start with might be 0.7 
% %
% %
% % Author: F. Benmansour 2011, CVLab, EPFL
% 
% % first smooth the image witha sigma equal to the minimal image spacing
% smoothingSigma = min(ImageSpacing);
% SmoothedImage = matitk('FGA', [smoothingSigma, 5], InputImage, [], [], ImageSpacing);
% 
% % compute the level set function that such that its zero
% % levelset is a sphere of the given center and radius
% InitialLevelSet = matitk('SFM', 1e9, ones(size(InputImage)), [], center, ImageSpacing);
% InitialLevelSet = InitialLevelSet - radius;
% 
% % Estimating the sigmoid function parameters
% meanIntensityInsideSphere = mean(SmoothedImage(InitialLevelSet <=0));
% sigBeta = (255.0*SigmoidWeight) *( meanIntensityInsideSphere /  max(SmoothedImage(:)) );
% sigAlpha = -1.0;
% 
% % compute the image intensity-baqsed sigmoid function
% ObjectDetector = matitk('FSN', [0, 1, sigAlpha, sigBeta], 255.0*(double(SmoothedImage)/max(SmoothedImage(:))) );
% 
% % run the geodesic active contour segmentation
% GAC_params = [-1.0, 0.1, 1.5, 0.01, 500];
% Segmentation = matitk('SGAC', GAC_params, ObjectDetector, InitialLevelSet, [], ImageSpacing);
% 
% return;