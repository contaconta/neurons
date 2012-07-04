% get a binary mask containing nuclei
function B = getNucleiBinaryMask(LAPL, J, LAPL_THRESH, NUC_INT_THRESH, NUC_MIN_AREA)


Blap = LAPL <  LAPL_THRESH;
Blap = bwareaopen(Blap, NUC_MIN_AREA);
Bprop = regionprops(Blap, 'Eccentricity', 'PixelIdxList');
for i = 1:length(Bprop)
    if Bprop(i).Eccentricity > .85
        Blap(Bprop(i).PixelIdxList) = 0;
    end
end
Bint = J > NUC_INT_THRESH;
B = Blap | Bint;
B = bwareaopen(B, NUC_MIN_AREA);    % get rid of small components
B = imfill(B,'holes');              % fill holes
