labelPath = '../full_size_10img_seg_plus_labels/';
imagePath = '/localhome/aurelien/Documents/EM/raw_mitochondria2/originals/';
imageFilenm = [imagePath 'FIBSLICE0002.png'];
labelFilenm = [labelPath 'FIBSLICE0002.dat'];

I = imread(imageFilenm);

fid = fopen(labelFilenm,'r');
L = fread(fid,[size(I,2) size(I,1)],'int32');
L = double(L);
L = L+1;


fclose(fid);


list = unique(L(:))';

for l = list

	BW = L==l;
	A = zeros(size(BW));
	BW = bwmorph(BW,'dilate',1) - BW;
	BW(BW < 0) = 0;
	BW = logical(BW);
	neighbors = L(BW);
	neighbors = setdiff(unique(neighbors),l)';
	
	for n = neighbors
		A(L==n) = find(neighbors ==n);		
	end

	%imagesc(A);
	%pause(.0001);
	%refresh;

end
