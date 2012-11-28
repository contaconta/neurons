function peak=fpeak(x,y,s,Range)

%Brief introduction:Find peak value of datas.
%Variable:  s is the sensitivity of the function.
%           Range is the peak value's range
%-------------------------------------------------
%Make sure x and y are with same form.
[rx,cx]=size(x);
[ry,cy]=size(y);
if  rx==1
    x=x';
    rx=length(x);
end
if  ry==1;
    y=y';
    ry=length(y);
end
if  rx~=ry
    fprintf('%s','Vector element must agree!');
    return
end

numP=1;
Data=sortrows([x,y]);
for i=1:rx
    isP=getPeak(Data,i,s);
    if  sum(isnan(isP))==0
        peak(numP,:)=isP;
        numP=numP+1;
    end
end
if nargin == 4
    peak=peak(find(peak(:,1)>=Range(1) & peak(:,1)<=Range(2)),:);
    peak=peak(find(peak(:,2)>=Range(3) & peak(:,2)<=Range(4)),:);
end

%-------------------------------------------
function p=getPeak(Data,i,s)
%English:Select points by sensitivity
%Chinese:���������ѡ��Ҫ�����жϵĵ�
if i-s<1
    top=1;
else
    top=i-s;
end
y=Data(:,2);
if i+s>length(y)
    bottom=length(y);
else
    bottom=i+s;
end

%Add solution as an input arguments,then you can write your own solution. 
%Solution 1��All datas in the range of senstitivity must be <= or >= peak
%value
%Solution 1��Ҫ�������������ȷ�Χ�ڵĵ㶼<=��>=��ֵ
%If tP=1, it's top peak
%tP�ж��Ƿ�Ϊ�ϼ�ֵ�㣬tP=1Ϊ�ϼ�ֵ��
%If bP=1, it's bottom peak
%bP�ж��Ƿ�Ϊ�¼�ֵ�㣬bP=1Ϊ�¼�ֵ��
tP=(sum(y(top:bottom)>=y(i))==1);
bP=(sum(y(top:bottom)<=y(i))==1);
if tP==1
    p=Data(i,:);
else
    p=[nan,nan];
end

%Solution 2��������������