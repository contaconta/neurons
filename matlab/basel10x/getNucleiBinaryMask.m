% get a binary mask containing nuclei
function B = getNucleiBinaryMask(LAPL, LAPL_THRESH, NUC_MIN_AREA)


Blap = LAPL <  LAPL_THRESH;
Blap = bwareaopen(Blap, NUC_MIN_AREA);
Bprop = regionprops(Blap, 'Eccentricity', 'PixelIdxList');
for i = 1:length(Bprop)
    if Bprop(i).Eccentricity > .85
        Blap(Bprop(i).PixelIdxList) = 0;
    end
end
B = Blap;
B = imfill(B,'holes');              % fill holes
