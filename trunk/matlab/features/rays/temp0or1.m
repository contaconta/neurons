function temp0or1(I, L, filenm, path)

N = 3000;

fid = fopen([path filenm], 'a');

r = 0 + .2.*randn(N,1);

for s = 1:N
    
    fprintf(fid, '%d ', 0);
    

        fprintf(fid, '%d:%g\n', 1, r(s));

    
    
end

r = 2 + .2.*randn(N,1);

for s = 1:N
    
    fprintf(fid, '%d ', 2);
    
    
        fprintf(fid, '%d:%g\n', 1, r(s));

    
    
end

fclose(fid);