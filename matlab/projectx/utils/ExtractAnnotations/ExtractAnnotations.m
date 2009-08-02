addpath('../LabelMeToolbox');
%addpath('../../toolboxes/kevin');
folder = 'FIBSLICE';
HOMEANNOTATIONS = 'Annotations';
HOMEIMAGES = 'Images';
type = 'mitochondria'
sub_type = {'interior'};
%sub_type = 'membrane';
%sub_type = 'whole';
sub_type = {'interior','membrane','whole'};
%sub_type = {'interior','whole'};
force_xml = true;
size_bb = 24;
max_bb_overlap = floor(((size_bb*2+1)^2)*(3/4))
shrinking_factor = round(size_bb/2);
export_center = true; % Export center pixel. If false, it exports a
                      % bounding box around the pixel instead

dir_name = [HOMEIMAGES '/' folder '/'];
files = dir(dir_name);

for i=1:length(files)
  filename = files(i).name;
  
  if (strcmp(filename,'.')==1 || strcmp(filename,'..')==1)
    continue
  end
  
  % Extract image id
  patterns = '\d*';
  img_idc = regexp(filename, patterns, 'match');
  img_id = img_idc{1};
  write_XML = false;
  
  % Check if XML annotation file already exists
  xml_filename = [HOMEANNOTATIONS '/' folder '/FIBSLICE' img_id '.xml']
  
  if ((force_xml==true) || (exist(xml_filename) > 0))
    full_filename = [dir_name filename]
    I = imread(full_filename);
    I = I(:,:,1);
    %BW = im2bw(I, graythresh(I));
    
    obj_id = 1;
    sampled_points = logical(zeros(size(I),'uint8'));
    
    annotation_filename = ['annotation' img_id '.png']
    BW = imread(annotation_filename);
    BW = BW(:,:,1);

    % Metadata for XML file
    clear xml;
    xml.annotation.filename = filename;
    xml.annotation.folder = folder;

    if(contains(sub_type,'whole'))
      full_type = [type ' whole'];
      write_XML = true;
      % Export list of boundary points
      [B,L] = bwboundaries(BW,'noholes');
      for obj = 1:length(B)
        boundary = B{obj};
        %plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);

        % Metadata for the object
        xml.annotation.object(obj).name = full_type;
        xml.annotation.object(obj).deleted = 0;
        xml.annotation.object(obj).verified = 1;
        xml.annotation.object(obj).id = obj;

        for i = 1:length(boundary)
          xml.annotation.object(obj).polygon(1).pt(i).y = boundary(i,1);
          xml.annotation.object(obj).polygon(1).pt(i).x = boundary(i,2);
        end
      end
      obj_id = length(B) + 1;
    end
    if(contains(sub_type,'interior'))
      full_type = [type ' interior'];
      write_XML = true;
      % shrink the binary mask
      % BW2 = bwmorph(BW,'shrink',shrinking_factor);
      
      se = strel('ball',5,5);
      BW2 = imerode(BW,se);
      
      % get pixels that belong to the object
      sPixels = regionprops(bwlabel(BW2), 'PixelList');
      
      for obj=1:length(sPixels)
        
        %sPixels(obj).PixelList(1:length(sPixels(obj).PixelList)/50:length(sPixels(obj).PixelList),:);
        pixelList = sPixels(obj).PixelList;        
        numsamples = round(size(pixelList,1));
        samps = randsample(size(pixelList,1),numsamples);
        pixelList = pixelList(samps,:);
        
        % Loop over the pixels belonging to obj
        %W = wristwatch('start', 'end', length(pixelList), 'every', 100);
        for p=1:size(pixelList,1)
          
          center_bb = pixelList(p,:); % center of the patch
          
          if (center_bb(1) <= size_bb) || (center_bb(2) <= size_bb) || ...
                (center_bb(1) > size(I,2)-size_bb) || (center_bb(2) > size(I,1)-size_bb)
            center_bb
            continue; % discard point
          end
          
          overlap=sum(sum(sampled_points(center_bb(2)-size_bb:center_bb(2)+size_bb,center_bb(1)-size_bb:center_bb(1)+size_bb)));
          
          if(overlap<max_bb_overlap)            
            
            % Metadata for the object
            xml.annotation.object(obj_id).name = full_type;
            xml.annotation.object(obj_id).deleted = 0;
            xml.annotation.object(obj_id).verified = 1;
            xml.annotation.object(obj_id).id = obj_id;
            
            % Export coordinates
            if export_center
              xml.annotation.object(obj_id).polygon(1).pt(1).x = center_bb(1);
              xml.annotation.object(obj_id).polygon(1).pt(1).y = center_bb(2);
            else
              xml.annotation.object(obj_id).polygon(1).pt(1).x = center_bb(1)-size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(1).y = center_bb(2)-size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(2).x = center_bb(1)-size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(2).y = center_bb(2)+size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(3).x = center_bb(1)+size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(3).y = center_bb(2)+size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(4).x = center_bb(1)+size_bb;
              xml.annotation.object(obj_id).polygon(1).pt(4).y = center_bb(2)-size_bb;
            end
            obj_id = obj_id + 1;         
            
            sampled_points(center_bb(2)-size_bb:center_bb(2)+size_bb,center_bb(1)-size_bb:center_bb(1)+size_bb)=ones(size_bb*2+1);
          end
          
          %W = wristwatch(W, 'update', p, 'text', '...how long will it take? ');
          
        end
      end
    end
    if(contains(sub_type,'membrane'))
      full_type = [type ' membrane'];
      write_XML = true;
      % Export patches along the boundary
      [B,L] = bwboundaries(BW,'noholes');
      for obj = 1:length(B)
        boundary = B{obj};
        % Resample boundary
        boundary=boundary(1:round(length(boundary)/100):length(boundary),:);
        
        %plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);

        for i = 1:length(boundary)
          
          % Metadata for the object
          xml.annotation.object(obj_id).name = full_type;
          xml.annotation.object(obj_id).deleted = 0;
          xml.annotation.object(obj_id).verified = 1;
          xml.annotation.object(obj_id).id = obj_id;
          
          xml.annotation.object(obj_id).polygon(1).pt(1).y = boundary(i,1);
          xml.annotation.object(obj_id).polygon(1).pt(1).x = boundary(i,2);
          
          %xml.annotation.object(obj_id).polygon(1).pt(1).y = boundary(i,1)-size_bb;
          %xml.annotation.object(obj_id).polygon(1).pt(1).x = boundary(i,2)-size_bb;
          %xml.annotation.object(obj_id).polygon(1).pt(2).y = boundary(i,1)-size_bb;
          %xml.annotation.object(obj_id).polygon(1).pt(2).x = boundary(i,2)+size_bb;          
          %xml.annotation.object(obj_id).polygon(1).pt(3).y = boundary(i,1)+size_bb;
          %xml.annotation.object(obj_id).polygon(1).pt(3).x = boundary(i,2)+size_bb;          
          %xml.annotation.object(obj_id).polygon(1).pt(4).y = boundary(i,1)+size_bb;
          %xml.annotation.object(obj_id).polygon(1).pt(4).x = boundary(i,2)-size_bb;          
          
          obj_id = obj_id + 1;
        end
      end
    end
  end

  if write_XML
    disp(['trying to write xml file ' xml_filename]);
    obj_id
    writeXML(xml_filename,xml);
  end
end

obj_id

% Show annotations
D = LMdatabase(HOMEANNOTATIONS);
LMplot(D, 1, HOMEIMAGES);
%LMdbshowscenes(D, HOMEIMAGES);
