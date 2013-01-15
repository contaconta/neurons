function Im = load_image_from_sequence(inputSeqDirToProcess, channel, image_idx)

list = dir(fullfile([inputSeqDirToProcess '/' channel], '*.TIF'));
names = {list.name};
sorted_filenames = sort_nat(names);

filename = [inputSeqDirToProcess '/' channel '/' sorted_filenames{image_idx}];

Im = imread(filename);

