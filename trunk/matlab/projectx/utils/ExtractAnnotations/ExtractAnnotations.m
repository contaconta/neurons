
addpath('../LabelMeToolbox');
addpath('../../matlab/toolboxes/kevin');
HOMEANNOTATIONS = 'Annotations/FIBSLICE';
HOMEIMAGES = 'Images/FIBSLICE';
folder = '';
type = 'mitochondria'
%sub_type = 'interior';
sub_type = 'membrane';
%sub_type = 'whole';
force_xml = true
size_bb = 30;
max_bb_overlap = floor(((size_bb*2+1)^2)*0.05)

dir_name = [HOMEIMAGES folder '/']
files = dir(dir_name)

for i=1:length(files)
  filename = files(i).name
  
  if (strcmp(filename,'.')==1 || strcmp(filename,'..')==1)
    continue
  end
  
  % Extract image id
  patterns = '\d*';
  img_idc = regexp(filename, patterns, 'match');
  img_id = img_idc{1};
  write_XML = true
  
  % Check if XML annotation file already exists
  xml_filename = [HOMEANNOTATIONS folder '/FIBSLICE' img_id '.xml']
  
  if ((force_xml==true) || (exist(xml_filename) > 0))
    full_filename = [dir_name filename]
    I = imread(full_filename);
    %BW = im2bw(I, graythresh(I));
    %img_id = sscanf(filename,'FIBSLICE%s/.png')
    
    annotation_filename = ['annotation' img_id '.png']
    BW = imread(annotation_filename);
    BW = BW(:,:,1);
    %imshow(label2rgb(L, @jet, [.5 .5 .5]))
    %hold on

    % Metadata for XML file
    clear xml
    xml.annotation.filename = filename;
    xml.annotation.folder = folder;

    if(strcmp(sub_type,'whole'))
      % Export list of boundary points
      [B,L] = bwboundaries(BW,'noholes');
      for obj = 1:length(B)
        boundary = B{obj};
        %plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);

        % Metadata for the object
        xml.annotation.object(obj).name = [type ' ' sub_type];
        xml.annotation.object(obj).deleted = 0;
        xml.annotation.object(obj).verified = 1;
        xml.annotation.object(obj).id = obj;

        for i = 1:length(boundary)
          xml.annotation.object(obj).polygon(1).pt(i).y = boundary(i,1);
          xml.annotation.object(obj).polygon(1).pt(i).x = boundary(i,2);
        end
      end
    elseif(strcmp(sub_type,'interior'))
      % get pixels that belong to the object
      sPixels = regionprops(bwlabel(BW), 'PixelList');
      % TODO : Subtract pixels that lie on the boundary
      
      obj_id = 1;
      for obj=1:length(sPixels)
        %size(sPixels(obj).PixelList)
        % Sample the list
        pixelList = sPixels(obj).PixelList(1:length(sPixels(obj).PixelList)/50:length(sPixels(obj).PixelList),:);
        %center_bb = sPixels(obj).PixelList(1,:); % center of the patch
        center_bb = pixelList(1,:); % center of the patch
        last_bb = [center_bb-size_bb center_bb+size_bb];
        
        % Metadata for the object
        xml.annotation.object(obj_id).name = [type ' ' sub_type];
        xml.annotation.object(obj_id).deleted = 0;
        xml.annotation.object(obj_id).verified = 1;
        xml.annotation.object(obj_id).id = obj_id;
        
        % Export the first patch for the membrane
        %xml.annotation.object(obj).polygon(1).pt(1).x = center_bb(1);
        %xml.annotation.object(obj).polygon(1).pt(1).y = center_bb(2);
        
        xml.annotation.object(obj_id).polygon(1).pt(1).x = center_bb(1)-size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(1).y = center_bb(2)-size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(2).x = center_bb(1)-size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(2).y = center_bb(2)+size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(3).x = center_bb(1)+size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(3).y = center_bb(2)+size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(4).x = center_bb(1)+size_bb;
        xml.annotation.object(obj_id).polygon(1).pt(4).y = center_bb(2)-size_bb;
        obj_id = obj_id + 1;
        
        %if obj_id > 100
        %  break
        %end
        
        % Loop over the pixels belonging to obj
        for p=2:length(pixelList)
          
          center_bb = pixelList(p,:); % center of the patch
          bb = [center_bb-size_bb center_bb+size_bb];
          
          %overlap(bb,last_bb)
          if(overlap(bb,last_bb)<max_bb_overlap)
            %disp('Take it')
            
            % Metadata for the object
            xml.annotation.object(obj_id).name = [type ' ' sub_type];
            xml.annotation.object(obj_id).deleted = 0;
            xml.annotation.object(obj_id).verified = 1;
            xml.annotation.object(obj_id).id = obj_id;
            
            % Export center coordinate
            %xml.annotation.object(obj).polygon(1).pt(1).x = center_bb(1);
            %xml.annotation.object(obj).polygon(1).pt(1).y = center_bb(2);

            xml.annotation.object(obj_id).polygon(1).pt(1).x = center_bb(1)-size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(1).y = center_bb(2)-size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(2).x = center_bb(1)-size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(2).y = center_bb(2)+size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(3).x = center_bb(1)+size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(3).y = center_bb(2)+size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(4).x = center_bb(1)+size_bb;
            xml.annotation.object(obj_id).polygon(1).pt(4).y = center_bb(2)-size_bb;
            
            last_bb = bb;
            obj_id = obj_id + 1;
            
          end
        end
      end
    elseif(strcmp(sub_type,'membrane'))
      
      % Export patches along the boundary
      [B,L] = bwboundaries(BW,'noholes');
      obj_id = 1;
      for obj = 1:length(B)
        boundary = B{obj};
        % Resample boundary
        boundary=boundary(1:length(boundary)/50:length(boundary),:);
        
        %plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);

        for i = 1:length(boundary)
          
          % Metadata for the object
          xml.annotation.object(obj_id).name = [type ' ' sub_type];
          xml.annotation.object(obj_id).deleted = 0;
          xml.annotation.object(obj_id).verified = 1;
          xml.annotation.object(obj_id).id = obj_id;
          
          %xml.annotation.object(obj_id).polygon(1).pt(1).y = boundary(i,1);
          %xml.annotation.object(obj_id).polygon(1).pt(1).x = boundary(i,2);
          
          xml.annotation.object(obj_id).polygon(1).pt(1).y = boundary(i,1)-size_bb;
          xml.annotation.object(obj_id).polygon(1).pt(1).x = boundary(i,2)-size_bb;
          xml.annotation.object(obj_id).polygon(1).pt(2).y = boundary(i,1)-size_bb;
          xml.annotation.object(obj_id).polygon(1).pt(2).x = boundary(i,2)+size_bb;          
          xml.annotation.object(obj_id).polygon(1).pt(3).y = boundary(i,1)+size_bb;
          xml.annotation.object(obj_id).polygon(1).pt(3).x = boundary(i,2)+size_bb;          
          xml.annotation.object(obj_id).polygon(1).pt(4).y = boundary(i,1)+size_bb;
          xml.annotation.object(obj_id).polygon(1).pt(4).x = boundary(i,2)-size_bb;          
          
          obj_id = obj_id + 1;
        end
      end
    else
        disp('Error : unknown type');
        write_XML = false;
    end
  end

  if write_XML
    writeXML(xml_filename,xml);
  end
end

% Show annotations
D = LMdatabase(HOMEANNOTATIONS);
LMplot(D, 1, HOMEIMAGES);
%LMdbshowscenes(D, HOMEIMAGES);
