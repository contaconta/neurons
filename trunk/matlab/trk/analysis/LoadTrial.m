function TRIAL = LoadTrial(path, naming_scheme, run_begin, run_end, run_step)

TRIAL.ExperimentNames = containers.Map();
numberExperiments = 0;

if(exist('run_begin','var')==0), run_begin=1; end
if(exist('run_end','var')==0), run_end=200; end
if(exist('run_end','var')==0), run_step=1; end


schm = [path '/' naming_scheme];
for nFile = run_begin:run_step:run_end
   name = sprintf(schm, nFile);
   % if the file exists
   if(exist(name)>0)
       disp(['loading ' name ]);
       R = load(name);
       R = HappyNeuronVector(R);
       R.FIL = [];
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