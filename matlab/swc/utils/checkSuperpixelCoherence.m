path = '~/src/EM/Cpp/SuperPixelCodeForAurelien/FIBSLICE0002.dat';
imsize = [1536 2048];
l=readRKLabel(path,imsize);

for x=1:size(l,1)
    for y=1:size(l,2)
        count = 0;
        lxy = l(x,y);
        for u=max(1,x-1):min(size(l,1),x+1)
            for v=max(1,y-1):min(size(l,2),y+1)
                if lxy == l(u,v)
                   count = count + 1;
                end
            end
        end
        if(count == 0)
           disp ['(x,y)' num2str(x) num2str(y)]    
        end
    end
end