function [R,C,Z] = bresenham_line(unitvector,r1,c1,z1, V)

%function [c,r,Z] = bresenham_line3d(P1, P2, precision)


dc = unitvector(2);
dr = unitvector(1);
dz = unitvector(3);

ac = abs(dc)*2;
ar = abs(dr)*2;
az = abs(dz)*2;

sc = sign(dc);
sr = sign(dr);
sz = sign(dz);

c = c1;
r = r1;
z = z1;
idc = 1;

if(ac>=max(ar,az))			% c dominant
  rd = ar - ac/2;
  zd = az - ac/2;

  while(1)
     R(idc) = r; %#ok<*AGROW>
     C(idc) = c;
     Z(idc) = z;
     idc = idc + 1;

     if  (r<=0) || (c<=0) || (z<=0) || (r >= size(V,1)+1) || (c >= size(V,2)+1) || (z >= size(V,3)+1) || (V(r,c,z)==1)
%         R = R(1:length(R)-1);
%         C = C(1:length(C)-1);
%         Z = Z(1:length(Z)-1);
        break;
     end

     if(rd >= 0)		% move along r
        r = r + sr;
        rd = rd - ac;
     end

     if(zd >= 0)		% move along z
        z = z + sz;
        zd = zd - ac;
     end

     c  = c  + sc;		% move along c
     rd = rd + ar;
     zd = zd + az;
  end
elseif(ar>=max(ac,az))		% r dominant
  cd = ac - ar/2;
  zd = az - ar/2;

  while(1)
     R(idc) = r;
     C(idc) = c;
     Z(idc) = z;
     idc = idc + 1;

     if  (r<=0) || (c<=0) || (z<=0) || (r >= size(V,1)+1) || (c >= size(V,2)+1) || (z >= size(V,3)+1) || (V(r,c,z)==1)
     %if (r<=1) || (c<=1) || (z<=1) || (r >= size(V,1)) || (c >= size(V,2)) || (z >= size(V,3)) || (V(r,c,z)==1)
%         R = R(1:length(R)-1);
%         C = C(1:length(C)-1);
%         Z = Z(1:length(Z)-1);
         break;
     end

     if(cd >= 0)		% move along c
        c = c + sc;
        cd = cd - ar;
     end

     if(zd >= 0)		% move along z
        z = z + sz;
        zd = zd - ar;
     end

     r  = r  + sr;		% move along r
     cd = cd + ac;
     zd = zd + az;
  end
elseif(az>=max(ac,ar))		% z dominant
  cd = ac - az/2;
  rd = ar - az/2;

  while(1)
     R(idc) = r;
     C(idc) = c;
     Z(idc) = z;
     idc = idc + 1;

     if  (r<=0) || (c<=0) || (z<=0) || (r >= size(V,1)+1) || (c >= size(V,2)+1) || (z >= size(V,3)+1) || (V(r,c,z)==1)
     %if (r<=1) || (c<=1) || (z<=1) || (r >= size(V,1)) || (c >= size(V,2)) || (z >= size(V,3)) || (V(r,c,z)==1)
%         R = R(1:length(R)-1);
%         C = C(1:length(C)-1);
%         Z = Z(1:length(Z)-1);
        break;
     end

     if(cd >= 0)
        c = c + sc;
        cd = cd - az;
     end

     if(rd >= 0)		% move along r
        r = r + sr;
        rd = rd - az;
     end

     z  = z  + sz;		% move along z
     cd = cd + ac;
     rd = rd + ar;
  end
end



return;					% bresenham_line3d