function Rectangle_List=generate_shape1(W_p,H_p,C,sigma,max_aspect_ratio)
global W;
global H;
W=W_p;
H=H_p;

mean_width=W/C;
mean_height=H/C;

%Will store all constituent rectangles of shape
global Rectangle_List;
%keep track of shape graphically
global Canvas;

%Every row is a rectangle [x_min, y_min, width, height, polarity]
Rectangle_List=[];
%Initialization----------------------------------------------------
Canvas=zeros(H,W);
%Get a rectangle in the [W,H] plane uniformaly at random
x_min=round(rand()*(W-1)) + 1;
y_min=round(rand()*(H-1)) + 1;

%aspect=max_aspect_ratio;
%while aspect>=max_aspect_ratio
    width=-1;
    while(width<0 || width > (W-x_min))
        width=round(randn(1)*sigma)+min(mean_width,W-x_min);%rand()*(W-x_min);
    end
    height=-1;
    while(height<0 || height > (H-y_min))
        height=round(randn(1)*sigma)+min(mean_height,H-y_min);%rand()*(H-y_min);
    end
    
    
    %aspect=max(width,height)/min(width,height);
%end


%Add to list and draw
Rectangle_List(1,:)=round([x_min,y_min,width,height,round(rand())]);
draw_rectangle(Rectangle_List(1,:),1);
if(sum(Canvas==0)==0)
    Rectangle_List=[];
    return;
end
%imagesc(Canvas,[0 C])
%pause

c=size(Rectangle_List,1);
%Add C-1 shapes
while(c<C)
    
   

    
    valid_point=0;
    while(valid_point==0)
        %Pick a rectangle at random
        anchor_rect=randi(c,1);
        %Pick a side at random top,left,bottom,right->1,2,3,4
        anchor_side=randi(4,1);
        %Pick a point at random
        switch anchor_side
            case 1
                anchor_y=Rectangle_List(anchor_rect,2);
                anchor_x=randi(Rectangle_List(anchor_rect,3)+1,1)-1+Rectangle_List(anchor_rect,1);
                %Move Anchor off top
                anchor_y=anchor_y-1;
            case 2    
                anchor_x=Rectangle_List(anchor_rect,1)+Rectangle_List(anchor_rect,3);
                anchor_y=randi(Rectangle_List(anchor_rect,4)+1,1)-1+Rectangle_List(anchor_rect,2);
                %Move Anchor off right
                anchor_x=anchor_x+1;
            case 3
                anchor_y=Rectangle_List(anchor_rect,2)+Rectangle_List(anchor_rect,4);
                anchor_x=randi(Rectangle_List(anchor_rect,3)+1,1)-1+Rectangle_List(anchor_rect,1);
                %Move Anchor off bottom
                anchor_y=anchor_y+1;
            case 4
                anchor_x=Rectangle_List(anchor_rect,1);
                anchor_y=randi(Rectangle_List(anchor_rect,4)+1,1)-1+Rectangle_List(anchor_rect,2);
                %Move Anchor off left
                anchor_x=anchor_x-1;
        end
        
        if (anchor_x >=1 && anchor_x <=W && anchor_y >=1 && anchor_y<=H)
            if Canvas(anchor_y,anchor_x)==0
                valid_point=1;
            end
        end
    end
    
    %anchor_x
    %anchor_y
   
       
    %Have a valid anchor, determine constraints on height, width given
    %height
    
    h_minus=height_constraint(anchor_x,anchor_y,0,c);
    h_plus=height_constraint(anchor_x,anchor_y,1,c);
    
    h_minus=abs(anchor_y-h_minus);
    h_plus=abs(anchor_y-h_plus);
    
    aspect=max_aspect_ratio;
    while aspect >= max_aspect_ratio
    
        %h_minus_sampled=randi(abs(anchor_y-h_minus)+1,1)-1;
        %h_plus_sampled=randi(abs(anchor_y-h_plus)+1,1)-1;
    
        
       
        
        if (h_minus >= mean_height/2 && h_plus >= mean_height/2)
            mean_h_minus=mean_height/2;
            mean_h_plus=mean_height/2;
        elseif(h_minus >= mean_height/2 && h_plus < mean_height/2)
            mean_h_plus=h_plus;
            mean_h_minus=min(mean_height/2+(mean_height/2-h_plus),h_minus);
        elseif(h_minus < mean_height/2 && h_plus >= mean_height/2)
            mean_h_minus=h_minus;
            mean_h_plus=min(mean_height/2+(mean_height/2-h_minus),h_plus);
        else
            mean_h_minus=h_minus;
            mean_h_plus=h_plus;
        end
        
        
        
        %h_minus
        %h_plus
        %mean_height
        %mean_h_minus
        %mean_h_plus
        
        h_minus_sampled=-1;
        while(h_minus_sampled<0 || h_minus_sampled > h_minus)
            h_minus_sampled=round(randn(1)*sigma+mean_h_minus);
        end
        
        %h_minus_sampled
        
        h_plus_sampled=-1;
        while(h_plus_sampled<0 || h_plus_sampled > h_plus)
            h_plus_sampled=round(randn(1)*sigma+mean_h_plus);
        end
        
        %h_plus_sampled
        
        
    
        w_minus=width_constraint_given_height(anchor_x,anchor_y-h_minus_sampled,h_plus_sampled+h_minus_sampled,0,c);
        w_plus=width_constraint_given_height(anchor_x,anchor_y-h_minus_sampled,h_plus_sampled+h_minus_sampled,1,c);
        
        w_minus=abs(anchor_x-w_minus);
        w_plus=abs(anchor_x-w_plus);
        
        
       if (w_minus >= mean_width/2 && w_plus >= mean_width/2)
            mean_w_minus=mean_width/2;
            mean_w_plus=mean_width/2;
        elseif(w_minus >= mean_width/2 && w_plus < mean_width/2)
            mean_w_plus=w_plus;
            mean_w_minus=min(mean_width/2+(mean_width/2-w_plus),w_minus);
        elseif(w_minus < mean_width/2 && w_plus >= mean_width/2)
            mean_w_minus=w_minus;
            mean_w_plus=min(mean_width/2+(mean_width/2-w_minus),w_plus);
        else
            mean_w_minus=w_minus;
            mean_w_plus=w_plus;
        end
        
        
        %w_minus
        %w_plus
        %mean_width
        %mean_w_minus
        %mean_w_plus
        
        
        w_minus_sampled=-1;
        while(w_minus_sampled<0 || w_minus_sampled > w_minus)
            w_minus_sampled=round(randn(1)*sigma+mean_w_minus);
        end
        
        %w_minus_sampled
        
        
        w_plus_sampled=-1;
        while(w_plus_sampled<0 || w_plus_sampled > w_plus)
            w_plus_sampled=round(randn(1)*sigma+mean_w_plus);
        end
        
        %w_plus_sampled
    
    
        aspect=max(w_minus_sampled+w_plus_sampled+1,h_minus_sampled+h_plus_sampled+1)/min(w_minus_sampled+w_plus_sampled+1,h_minus_sampled+h_plus_sampled+1 );
    end
    
    
   candidate_rect=round([anchor_x-w_minus_sampled anchor_y-h_minus_sampled w_plus_sampled+w_minus_sampled h_plus_sampled+h_minus_sampled round(rand())]); %randi(2,1)-1])
   
   
   
   
   
   %%Merge candidate with all possible rectangles
   merged_with=[];
   merging=1;
   while(merging==1)
       merging=0;
       for i=1:1:(c)
       
           if(sum(i==merged_with)==0)
               if(candidate_rect(5)==Rectangle_List(i,5))
               
                   
               if(candidate_rect(3)==Rectangle_List(i,3)) %Same width
                  
                   if(candidate_rect(1)==Rectangle_List(i,1)) %Same x_position
                     
                       if(candidate_rect(2)==Rectangle_List(i,2)+Rectangle_List(i,4)+1) %candidate is mergeable at bottom
                           
                           merged_with=[merged_with i];
                           candidate_rect=[Rectangle_List(i,1) Rectangle_List(i,2) Rectangle_List(i,3) Rectangle_List(i,4)+candidate_rect(4)+1 Rectangle_List(i,5)];
                           draw_rectangle(candidate_rect,i);
                           if(sum(Canvas==0)==0)
                               Rectangle_List=[]; 
                               return;
                           end
                           merging=1;
                           break;
                       elseif(candidate_rect(2)+candidate_rect(4)+1==Rectangle_List(i,2)) %candidate is mergeable on top
                           merged_with=[merged_with i];
                           candidate_rect=[candidate_rect(1) candidate_rect(2) candidate_rect(3) Rectangle_List(i,4)+candidate_rect(4)+1 Rectangle_List(i,5)];
                           draw_rectangle(candidate_rect,i);
                           if(sum(Canvas==0)==0)
                               Rectangle_List=[]; 
                               return;
                           end
                           merging=1;
                           break;
                       end 
                   end
               end
               
               if(candidate_rect(4)==Rectangle_List(i,4)) %Same height
                   if(candidate_rect(2)==Rectangle_List(i,2)) %Same y_position
                       if(candidate_rect(1)==Rectangle_List(i,1)+Rectangle_List(i,3)+1) %candidate is mergeable on right
                           merged_with=[merged_with i];
                           candidate_rect=[Rectangle_List(i,1) Rectangle_List(i,2) Rectangle_List(i,3)+candidate_rect(3)+1 Rectangle_List(i,4) Rectangle_List(i,5)];
                           draw_rectangle(candidate_rect,i);
                           if(sum(Canvas==0)==0)
                               Rectangle_List=[];
                               return;
                           end
                           merging=1;
                           break;
                       elseif(candidate_rect(1)+candidate_rect(3)+1==Rectangle_List(i,1)) %candidate is mergeable on left
                           merged_with=[merged_with i];
                           candidate_rect=[candidate_rect(1) candidate_rect(2) Rectangle_List(i,3)+candidate_rect(3)+1 candidate_rect(4) Rectangle_List(i,5)];
                           draw_rectangle(candidate_rect,i);
                           if(sum(Canvas==0)==0)
                               Rectangle_List=[]; 
                               return;
                           end
                           merging=1;
                           break;
                       end 
                   end
               end
               
               
               end
           end
           
       
       
       end
       
   end
   
  
   
   if(sum(merged_with)==0)
      Rectangle_List(c+1,:)=candidate_rect;
      draw_rectangle(candidate_rect,c+1);
      if(sum(Canvas==0)==0)
          Rectangle_List=[]; 
          return;
      end
   else
       Rectangle_List(merged_with,:)=[];
       Rectangle_List(size(Rectangle_List,1)+1,:)=candidate_rect;
   end
        
    %imagesc(Canvas,[0 C]);
    %Rectangle_List
    %pause
    
    c=size(Rectangle_List,1);
   
     
        
end

for i=1:C
    draw_rectangle(Rectangle_List(i,:),i);
end


%Have a shape of C boxes, could be all white or all black. Make sure it does not
%happen
if(sum(Rectangle_List(:,5))==C)
    %all 1;
    %Number of boxes to switch
    num=randi(C-1,1);
    num = max(num,1);
    %which to swtich, can repeat...too bad
    ind=randsample(C,num);
    Rectangle_List(ind,5)=0;
    %disp('fixed all 1');
   
elseif (sum(Rectangle_List(:,5))==0)
    %all 0;
    %Number of boxes to switch
    num=randi(C-1,1);
    num = max(num,1);
    %which to swtich, can repeat...too bad
    ind=randsample(C,num);
    Rectangle_List(ind,5)=1;
    %disp('fixed all 0');
end

% figure(2);
% imagesc(Canvas,[0 C]);
%     %Rectangle_List
% drawnow; pause(0.002);
% % pause;
% Rectangle_List
%     keyboard;


%imagesc(Canvas);


function y=height_constraint(anchor_x, anchor_y,up_down,number_rectangles)
global Rectangle_List;
global H;
h_plus=[H+1];
h_minus=[0];

if (up_down==0)
    
    for r=1:1:number_rectangles
        if ( Rectangle_List(r,2) + Rectangle_List(r,4) < anchor_y &&  Rectangle_List(r,1) <= anchor_x && Rectangle_List(r,1)+Rectangle_List(r,3) >= anchor_x)
            h_minus=[h_minus Rectangle_List(r,2) + Rectangle_List(r,4)];
        end
    end
    %maximum valid y to top
    y=max(h_minus)+1;
    
    
elseif (up_down==1)
    
    for r=1:1:number_rectangles
        if ( Rectangle_List(r,2) > anchor_y &&  Rectangle_List(r,1) <= anchor_x && Rectangle_List(r,1)+Rectangle_List(r,3) >= anchor_x)
            h_plus=[h_plus Rectangle_List(r,2)];
        end
    end
    %minimum valid y to bottom
    y=min(h_plus)-1;
    
end

function y=width_constraint(anchor_x, anchor_y,left_right,number_rectangles)
global Rectangle_List;
global W;
w_plus=[W+1];
w_minus=[0];

if (left_right==0)
    
    for r=1:1:number_rectangles
        if ( Rectangle_List(r,1) + Rectangle_List(r,3) < anchor_x &&  Rectangle_List(r,2) <= anchor_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= anchor_y)
            w_minus=[w_minus Rectangle_List(r,1) + Rectangle_List(r,3)];
        end
    end
    %maximum valid x to left
    y=max(w_minus)+1;
    
    
elseif (left_right==1)
    
    for r=1:1:number_rectangles
        if ( Rectangle_List(r,1) > anchor_x &&  Rectangle_List(r,2) <= anchor_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= anchor_y)
            w_plus=[w_plus Rectangle_List(r,1)];
        end
    end
    %minimum valid x to right
    y=min(w_plus)-1;
    
end


function y=width_constraint_given_height(min_x,min_y,sampled_height,left_right,number_rectangles)
global Rectangle_List;
global W;
w_plus=[W+1];
w_minus=[0];

if (left_right==0)
    
    for r=1:1:number_rectangles
        if(Rectangle_List(r,1) < min_x && ((Rectangle_List(r,2)>=min_y && Rectangle_List(r,2) <= min_y+sampled_height)||...
                                           (Rectangle_List(r,2)+Rectangle_List(r,4)>=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) <= min_y+sampled_height)||...
                                           (Rectangle_List(r,2)<=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= min_y)||...
                                           (Rectangle_List(r,2)+Rectangle_List(r,4)<=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= min_y)))
            w_minus=[w_minus Rectangle_List(r,1)+Rectangle_List(r,3)];
        end
    end
    %maximum valid width to left
    y=max(w_minus)+1;

elseif(left_right==1)
    
    for r=1:1:number_rectangles
        if(Rectangle_List(r,1) > min_x && ((Rectangle_List(r,2)>=min_y && Rectangle_List(r,2) <= min_y+sampled_height)||...
                                           (Rectangle_List(r,2)+Rectangle_List(r,4)>=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) <= min_y+sampled_height)||...
                                           (Rectangle_List(r,2)<=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= min_y)||...
                                           (Rectangle_List(r,2)+Rectangle_List(r,4)<=min_y && Rectangle_List(r,2)+Rectangle_List(r,4) >= min_y)))
            w_plus=[w_plus Rectangle_List(r,1)];
        end
    end
    %minimum valid width to right
    y=min(w_plus)-1;

end







    
    


function draw_rectangle(A,C)
global Canvas;
%top
Canvas(A(2),A(1):A(1)+A(3))=1;
%bottom
Canvas(A(2)+A(4),A(1):A(1)+A(3))=1;
%right
Canvas(A(2):A(2)+A(4),A(1))=1;
%left
Canvas(A(2):A(2)+A(4),A(1)+A(3))=1;

Canvas(A(2):A(2)+A(4),A(1):A(1)+A(3))=C;

