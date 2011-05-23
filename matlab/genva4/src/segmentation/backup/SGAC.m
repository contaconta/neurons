function Segmentation = SGAC(InputImage, ImageSpacing, center, radius, sigAlpha, sigBeta)
%
% Segmentation using the Geodesic Active Contour model
%
% Given an intial seed point and a radius, a sphere is shrinked until some
% convergence criterion is reached. raidus should be large enough so the
% sphere includes completely the object of interest.
%
% The crutial part of this function is the stopping metric function.
% The stopping metric function is a sigmoid function of the image
% intensities using sigAlpha and sigBeta.
% - If the object ot detect it darker than the background then alpha should
% be negative. sigAlpha = -1.0 for instance. sigBeta, sould be more or less
% equal to half the lowest image intensity of the object to detect.
%
% - If the object is brighterm, alpha should be positive, for instance =
% 1.0. same rule for beta.
%
% Author: F. Benmansour 2011, CVLab, EPFL

% first smooth the image witha sigma equal to the minimal image spacing
smoothingSigma = min(ImageSpacing);
SmoothedImage = matitk('FGA', [smoothingSigma, 5], InputImage, [], [], ImageSpacing);

% compute the level set function that such that its zero
% levelset is a sphere of the given center and radius
InitialLevelSet = matitk('SFM', 1e9, ones(size(InputImage)), [], center, ImageSpacing);
InitialLevelSet = InitialLevelSet - radius;


% compute the image intensity-baqsed sigmoid function
ObjectDetector = matitk('FSN', [0, 1, sigAlpha, sigBeta], 255.0*(double(SmoothedImage)/max(SmoothedImage(:))) );

% run the geodesic active contour segmentation
GAC_params = [-1.0, 0.1, 1.5, 0.01, 500];
Segmentation = matitk('SGAC', GAC_params, ObjectDetector, InitialLevelSet, [], ImageSpacing);

return;