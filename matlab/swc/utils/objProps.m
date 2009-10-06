
annotation_dir = '/home/alboot/usr/work/EM/annotation/'

ldir = dir([annotation_dir '*.png']);

% average values
areas = [];
perimeters = [];
eccentricities = [];
numberOfBlobs = 0;

for l=1:length(ldir)
  filename = [annotation_dir ldir(l).name]
  binaryImage=imread(filename);
  binaryImage=binaryImage(:,:,3);
  labeledImage = bwlabel(binaryImage, 8); % Label each blob so can do calc on it
  blobMeasurements = regionprops(labeledImage, 'all'); % Get all the blob properties.

  numberOfBlobs = numberOfBlobs + size(blobMeasurements, 1);
  areas = [areas blobMeasurements.Area];
  perimeters = [perimeters blobMeasurements.Perimeter];
  eccentricities = [perimeters blobMeasurements.Eccentricity];
  
end

disp(['numberOfBlobs: ' num2str(numberOfBlobs)]);
disp(['Area : m=' num2str(mean(areas)) ', std=' num2str(std(areas))]);
disp(['Perimeter : m=' num2str(mean(perimeters)) ', std=' num2str(std(perimeters))]);
disp(['Eccentricity : m=' num2str(mean(eccentricities)) ', std=' num2str(std(eccentricities))]);

