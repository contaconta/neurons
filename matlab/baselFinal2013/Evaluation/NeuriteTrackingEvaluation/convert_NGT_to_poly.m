function NGT = convert_NGT_to_poly(NGT, polywidth)
fprintf('converting NGT lines to polys');

for i = 1:numel(NGT)
    fprintf('.');
    for t = 1:numel(NGT(i).P)
        switch NGT(i).P(t).status
            case 'OK'
                for n = 1:numel(NGT(i).P(t).N)
                    NGT(i).P(t).N(n).linex = NGT(i).P(t).N(n).x;
                    NGT(i).P(t).N(n).liney = NGT(i).P(t).N(n).y;
                    [xp, yp] = line2poly(NGT(i).P(t).N(n).linex, NGT(i).P(t).N(n).liney, polywidth);
                    NGT(i).P(t).N(n).x = xp;
                    NGT(i).P(t).N(n).y = yp;
                end
            otherwise
                % do nothing
        end
    end
end
fprintf('\n');
    