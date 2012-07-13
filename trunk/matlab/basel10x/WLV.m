function R = WLV(h, Im, epsilon)

hpos = max(h, 0);
hneg = min(h, 0);

if sum(hpos(:)) > 0
    hpos = hpos / sum(hpos(:)) ;
end

if sum(hneg(:)) < 0
    hneg = hneg / sum(hneg(:)) ;
end

% the following operations apply just because the fitler h is symmetric
Mu_pos = imfilter(Im, hpos, 'replicate');
WLVpos = imfilter(Im.^2, hpos, 'replicate') - Mu_pos.^2;

Mu_neg = imfilter(Im, hneg, 'replicate');
WLVneg = imfilter(Im.^2, hneg, 'replicate') - Mu_neg.^2;

R = (Mu_pos - Mu_neg) ./ (min(sqrt(WLVpos), sqrt(WLVneg)) + epsilon);
