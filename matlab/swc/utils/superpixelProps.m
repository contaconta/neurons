
annotation_dir = '/home/alboot/usr/work/EM/annotation/'

% average values
aNbSuperpixels = 0;
nbBlobs = 0;

ldir = dir([annotation_dir '*.png']);
for l=1:length(ldir)
  filename = [annotation_dir ldir(l).name];
  binaryImage=imread(filename);
  binaryImage=binaryImage(:,:,3);
  labeledImage = bwlabel(binaryImage, 8); % Label each blob so can do calc on it
  blobMeasurements = regionprops(labeledImage, 'PixelIdxList'); % Get all the blob properties.

  % Load file containing superpixel labels
  id=regexp(filename,'\D*(\d*)\D*','tokens');
  superpixel_file = ['../temp/labels/FIBSLICE' num2str(id{1}{1}) '.dat'];

  if exist(superpixel_file,'file')
    L = readRKLabel(superpixel_file, [1536 2048])';
    
    % Count the number of superpixels per blob
    for k = 1:size(blobMeasurements, 1)
      labels = [];
      for p=blobMeasurements(k).PixelIdxList
        labels = [labels L(p)];
      end
      aNbSuperpixels = aNbSuperpixels + length(unique(labels));
    end
    nbBlobs = nbBlobs + size(blobMeasurements, 1);
  end
end

aNbSuperpixels = aNbSuperpixels/nbBlobs;
disp(['Number of Superpixels for a mitochondria : ' num2str(aNbSuperpixels)]);
