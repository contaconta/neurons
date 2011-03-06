function [exp_date, label_txt, num_txt] = trkGetDateAndLabel(folder)


datepat = '\d*-\d*-\d\d\d\d';
exppat  = '/\d\d\d/';


exp_date = regexp(folder, datepat, 'match');
exp_label = regexp(folder, exppat, 'match');

exp_date = exp_date{1};
exp_label = exp_label{1};
num_txt = exp_label(2:4);
exp_label = str2double(exp_label(2:4));


switch exp_date
    case '14-11-2010'
        if exp_label <= 10
            label_txt = 'Not_kd';
        elseif exp_label <= 20
            label_txt = 'Not_Targ';
        elseif exp_label <= 30
            label_txt = 'RhoA_siRNA1';  
        elseif exp_label <= 40
            label_txt = 'RhoA_siRNA2';      
        elseif exp_label <= 50
            label_txt = 'RhoA_siRNA3'; 
        elseif exp_label <= 60
            label_txt = 'Net_siRNA';     
        elseif exp_label <= 70
            label_txt = 'Trio_siRNA2';       
        elseif exp_label <= 80
            label_txt = 'Trio_siRNA1';       
        elseif exp_label <= 90
            label_txt = 'SrGAP2_siRNA3';   
        elseif exp_label <= 100
            label_txt = 'SrGAP2_siRNA2';       
        elseif exp_label <= 110
            label_txt = 'Map2K7_siRNA1';      
        elseif exp_label <= 120
            label_txt = 'Map2K7_siRNA2'; 
        elseif exp_label <= 130
            label_txt = 'Map2K7_siRNA3';     
        elseif exp_label <= 140
            label_txt = 'SrGAP2_siRNA1'; 
        else
            error('unknown experiment number!');
        end
 

    case '15-11-2010'
        if exp_label <= 10
            label_txt = 'Not_kd';
        elseif exp_label <= 20
            label_txt = 'Not_Targ';
        elseif exp_label <= 30
            label_txt = 'RhoA_siRNA1';  
        elseif exp_label <= 40
            label_txt = 'RhoA_siRNA2';      
        elseif exp_label <= 50
            label_txt = 'RhoA_siRNA3'; 
        elseif exp_label <= 60
            label_txt = 'Net_siRNA';     
        elseif exp_label <= 71
            label_txt = 'Trio_siRNA2';       
        elseif exp_label <= 81
            label_txt = 'Trio_siRNA1';       
        elseif exp_label <= 94
            label_txt = 'SrGAP2_siRNA3';   
        elseif exp_label <= 104
            label_txt = 'SrGAP2_siRNA2';       
        elseif exp_label <= 114
            label_txt = 'Map2K7_siRNA1';      
        elseif exp_label <= 124
            label_txt = 'Map2K7_siRNA2'; 
        elseif exp_label <= 134
            label_txt = 'Map2K7_siRNA3';     
        elseif exp_label <= 143
            label_txt = 'SrGAP2_siRNA1'; 
        else
            error('unknown experiment number!');
        end
        
        
    case '16-11-2010'
        if exp_label <= 10
            label_txt = 'Not_kd';
        elseif exp_label <= 19
            label_txt = 'Not_Targ';
        elseif exp_label <= 29
            label_txt = 'RhoA_siRNA1';  
        elseif exp_label <= 39
            label_txt = 'RhoA_siRNA2';      
        elseif exp_label <= 49
            label_txt = 'RhoA_siRNA3'; 
        elseif exp_label <= 59
            label_txt = 'Net_siRNA';     
        elseif exp_label <= 69
            label_txt = 'Trio_siRNA2';       
        elseif exp_label <= 78
            label_txt = 'Trio_siRNA1';       
        elseif exp_label <= 88
            label_txt = 'SrGAP2_siRNA3';   
        elseif exp_label <= 98
            label_txt = 'SrGAP2_siRNA2';       
        elseif exp_label <= 107
            label_txt = 'Map2K7_siRNA1';      
        elseif exp_label <= 117
            label_txt = 'Map2K7_siRNA2'; 
        elseif exp_label <= 127
            label_txt = 'Map2K7_siRNA3';     
        elseif exp_label <= 137
            label_txt = 'SrGAP2_siRNA1'; 
        else
            error('unknown experiment number!');
        end
 
    case '17-11-2010'
        if exp_label <= 10
            label_txt = 'Not_kd';
        elseif exp_label <= 20
            label_txt = 'Not_Targ';
        elseif exp_label <= 30
            label_txt = 'RhoA_siRNA1';  
        elseif exp_label <= 40
            label_txt = 'RhoA_siRNA2';      
        elseif exp_label <= 50
            label_txt = 'RhoA_siRNA3'; 
        elseif exp_label <= 60
            label_txt = 'Net_siRNA';     
        elseif exp_label <= 70
            label_txt = 'Trio_siRNA2';       
        elseif exp_label <= 80
            label_txt = 'Trio_siRNA1';       
        elseif exp_label <= 90
            label_txt = 'SrGAP2_siRNA3';   
        elseif exp_label <= 100
            label_txt = 'SrGAP2_siRNA2';       
        elseif exp_label <= 110
            label_txt = 'Map2K7_siRNA1';      
        elseif exp_label <= 120
            label_txt = 'Map2K7_siRNA2'; 
        elseif exp_label <= 130
            label_txt = 'Map2K7_siRNA3';     
        elseif exp_label <= 140
            label_txt = 'SrGAP2_siRNA1'; 
        else
            error('unknown experiment number!');
        end
        
        
        
    otherwise
        error('unknown date!');
end


