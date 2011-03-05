close all;
clear all;

load '/net/cvlabfiler1/home/ksmith/Basel/Results/14-11-2010_001.mat';

%%
interestingCells = zeros(1,length(trkSeq));

HAPPY_THRESHOLD = 100;

% By default neurons are unhappy
happyVector = zeros(size(tracks));

%% We only care for cells that have a given number of filaments
for nTrack = 1:length(trkSeq)
   if isempty(trkSeq{nTrack})
       continue
   end
   track = trkSeq{nTrack};
   mean_neurite_t = zeros(size(track));
   mean_number_d = zeros(size(track));

   for nDtc = 1:length(track)
       nd = track(nDtc);
       nmbr_dendrites = max(FILAMENTS(nd).NeuriteID);
       dendrite_lengths = zeros(1,nmbr_dendrites);
       for nddr = 1:nmbr_dendrites
           dendrite_lengths(nddr) = size(find(FILAMENTS(nd).NeuriteID == nddr),1);
       end
       if(isempty(nmbr_dendrites) || nmbr_dendrites ==0 )
         mean_number_d(nDtc) = 0;
         mean_neurite_t(nDtc) = 0;       
       else
        mean_number_d(nDtc) = nmbr_dendrites;
        
        mean_neurite_t(nDtc) = mean(dendrite_lengths);
       end
   end
   % if they are not many, the are long and variate, they have fun
   happy_factor = mean(mean_neurite_t)*std(mean_neurite_t)/(mean(mean_number_d)+1);
   
   if(happy_factor > HAPPY_THRESHOLD)
      happyVector(track) = 1;
   end
   
end


