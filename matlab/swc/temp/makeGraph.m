% Create binary files containing a list of neighbors for each superpixel

addpath('../utils');
addpath('../bin');
labelPath = 'labels/';
imagePath = '/localhome/aurelien/usr/share/Data/LabelMe/Images/FIBSLICE/';
imageExt = '*.png';
labelExt = '*.dat';
%imageFilenm = [imagePath 'FIBSLICE0002.png'];
%labelFilenm = [labelPath 'FIBSLICE0002.dat'];

imgs = dir([imagePath imageExt]);
dats = dir([labelPath labelExt]);

for i=2:length(dats)
  
  prefixFilenm = regexp(imgs(i).name, '(.*)png', 'tokens');
  outFilenm = [prefixFilenm{1}{1} 'dat']
  
  % Read image
  imageFilenm = [imagePath imgs(i).name]
  I = imread(imageFilenm);

  % Read files containing the labels
  labelFilenm = [labelPath dats(i).name]
  L = readRKLabel(labelFilenm, [1536 2048])';

  [G0, G0list] = adjacency(L);

  %keyboard;

  % Output file
  fid = fopen(outFilenm, 'w');
  for j=1:length(G0list)
    fprintf(fid, '%d ', j);
    for k=1:length(G0list{j})
      fprintf(fid, '%d ', G0list{j}(k));
    end
    fprintf(fid, '\n');
  end
  fclose(fid);

end
