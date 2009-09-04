n = regexp(datestr(now), '\w+', 'match');
n1 = strcat('Copyright © ', ' ', n(3), ' All rights reserved.');

INFO.appname     = 'ProjectX - Boosted Microscopy Detection';
INFO.institution = 'École Polytechnique Fédérale de Lausanne (CVLab)';
INFO.version     = '0.1';
INFO.author      = 'Kevin Smith and Aurelien Lucchi';
INFO.email       = 'kevin.smith@epfl.ch, aurelien.lucchi@epfl.ch';
INFO.copyright   = n1{1};  clear n n1;

disp(' ');
disp( '-------------------------------------------------------------------------'); 
disp(['  ' INFO.appname ', version ' INFO.version ]);
disp(['    by ' INFO.author ]);
disp(['    ' INFO.email]);
disp(['    ' INFO.institution ]);
disp (' ');
disp(['  started on:  ' datestr(now) ]);
disp(['  + query:     ' DATASETS.pos_query]);
disp(['  - query:     ' DATASETS.neg_query]);
disp(['  LEARNERS:    ' strcat(LEARNERS.types{:}) ]);
disp(['  TRAIN:       ' num2str(DATASETS.TRAIN_POS) '+ examples, ' num2str(DATASETS.TRAIN_NEG) '- examples']);
disp(['  VALIDATION:  ' num2str(DATASETS.VALIDATION_POS) '+ examples, ' num2str(DATASETS.VALIDATION_NEG) '- examples']);

disp (' ');
disp(['  ' INFO.copyright ]);
disp(' ');
disp('  This program is free software; you can redistribute it and/or modify it');
disp('  under the terms of the GNU General Public License version 2 (or higher)'); 
disp('  as published by the Free Software Foundation.');
disp(' ');                                                                     
disp('  This program is distributed WITHOUT ANY WARRANTY; without even the ');
disp('  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR ');
disp('  PURPOSE.  See the GNU General Public License for more details.');
disp( '-------------------------------------------------------------------------'); 
disp(' ');