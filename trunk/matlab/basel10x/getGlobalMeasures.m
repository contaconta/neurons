function Experiment = getGlobalMeasures(date_txt,label_txt, tracks, Dlist, num_txt)

Experiment.Date = date_txt;
Experiment.Label = label_txt;
Experiment.NumberOfCells = max(tracks);
Experiment.Length = length(Dlist);
Experiment.AssayPosition = num_txt;