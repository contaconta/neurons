n = regexp(datestr(now), '\w+', 'match');
n1 = strcat('Copyright © ', ' ', n(3));

INFO.appname     = 'ProjectX - Boosted Microscopy Detection';
INFO.institution = 'École Polytechnique Fédérale de Lausanne (CVLab)';
INFO.version     = '0.1';
INFO.author      = 'Kevin Smith and Aurelien Lucchi';
INFO.email       = 'kevin.smith@epfl.ch, aurelien.lucchi@epfl.ch';
INFO.copyright   = n1{1};



disp( '-------------------------------------------------------------------'); 
disp(['  ' INFO.appname ', version ' INFO.version ]);
disp(['    by ' INFO.author ]);
disp(['    ' INFO.email]);
disp(['    ' INFO.institution ]);
disp (' ');
disp(['  started on: ' datestr(now) ]);
disp(['  DATA SET:   ' DATASETS.filelist]);
disp (' ');
disp(['  ' INFO.copyright ]);
disp( '-------------------------------------------------------------------'); 
disp( ' ');