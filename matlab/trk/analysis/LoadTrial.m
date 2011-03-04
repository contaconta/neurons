function TRIAL = LoadTrial(path, naming_scheme)

TRIAL.ExperimentNames = containers.Map();
numberExperiments = 0;

schm = [path '/' naming_scheme];
for nFile = 1:1:200
   name = sprintf(schm, nFile);
   % if the file exists
   if(exist(name)>0)
       disp(['loading ' name ]);
       R = load(name);
       RType = R.GlobalMeasures.Label;
       % If there is already another run of the same experiment
       if( TRIAL.ExperimentNames.isKey(RType) )
          nExp = TRIAL.ExperimentNames(RType);
          TRIAL.EXPERIMENTS(nExp).RUNS(...
              length(TRIAL.EXPERIMENTS(nExp).RUNS)+1) = R;
       else
           numberExperiments = numberExperiments+1;
           TRIAL.ExperimentNames(RType) = numberExperiments;           
           TRIAL.EXPERIMENTS(numberExperiments).RUNS(1) = R;
       end
       
   end % File Exists
end %File loop