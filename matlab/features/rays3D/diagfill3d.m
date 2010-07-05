function E2 = diagfill3d(E)


E2 = E;

pixelval = 255;

for r = 2:size(E,1)-1
    for c = 2:size(E,2)-1
        for z = 2:size(E,3)-1
            
            % is E(r,c,z) an edge?
            if E(r,c,z) ==pixelval
            
                %disp('hello');
                
                %case 1
                if E(r+1,c+1,z+1) == pixelval
                   E2(r-1,c,z) = pixelval;
                   E2(r-1,c,z+1) = pixelval;                    
                
                % case 3
                elseif E(r+1, c+1, z) == pixelval
                    E2(r-1,c,z) = pixelval;
                end
                
                % case 2
                if E(r-1, c-1, z-1) == pixelval
                   E2(r+1,c,z) = pixelval;
                   E2(r+1,c,z+1) = pixelval;
                
                %case 4   
                elseif E(r-1, c+1,z) == pixelval
                    E2(r+1,c,z) = pixelval;
                end

            end
        end
    end
end