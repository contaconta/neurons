addpath '/localhome/aurelien/src/EM/neurons/matlab/toolboxes/LabelMeToolbox'

HOMEIMAGES = '/localhome/aurelien/work/EM/datasets/LabelMe/Images/';
HOMEANNOTATIONS = '/localhome/aurelien/work/EM/datasets/LabelMe/Annotations/';
inputdir = 'FIBSLICE_400/';
outputpath = 'Masks/'

ldirs = dir([HOMEANNOTATIONS inputdir '*.xml']);
for i=1:length(ldirs)
  filename=[HOMEANNOTATIONS inputdir ldirs(i).name]
  xml2mask(filename, HOMEIMAGES, outputpath);
end

