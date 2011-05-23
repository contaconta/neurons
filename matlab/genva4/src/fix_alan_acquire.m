function V = fix_alan_acquire(V,DATAMAX)

%DATAMAX = 8288;  % weird max value in alan's data

V = uint8(  V * ((2^8)/DATAMAX));