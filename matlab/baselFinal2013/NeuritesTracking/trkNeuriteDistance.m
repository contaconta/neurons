function WD = trkNeuriteDistance(d1,d2)

weightCableLength = 50;
weightSomaContact = 10;
weightCentroidShift = 5;
%% TotalCableLength

m = abs((d1.TotalCableLength - d2.TotalCableLength) / (d1.TotalCableLength + d2.TotalCableLength));

%% DistToSomaMedian

dm = abs((d1.ExtremeLength_q_50 - d2.ExtremeLength_q_50) / (d1.ExtremeLength_q_50 + d2.ExtremeLength_q_50));

%% LengthBrancheMedia

ds = abs((d1.LengthBranches_q_50 - d2.LengthBranches_q_50) / (d1.LengthBranches_q_50 + d2.LengthBranches_q_50));

%% shape distance

shape_d = m+dm+ds;

centroid_d = sqrt( (d1.CentroidOffset(1) - d2.CentroidOffset(1))^2 + (d1.CentroidOffset(2) - d2.CentroidOffset(2))^2);
soma_d = sqrt( (d1.SomaContact(1) - d2.SomaContact(1))^2 + (d1.SomaContact(2) - d2.SomaContact(2))^2);

CableLength_d = m;


WD = weightCableLength*CableLength_d + weightCentroidShift*centroid_d + weightSomaContact*soma_d;

if isnan(WD)
    keyboard;
end