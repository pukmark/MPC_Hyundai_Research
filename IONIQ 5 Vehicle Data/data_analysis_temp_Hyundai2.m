close all
clear
clc

%% Load datasets
data1 = DSpace2MatNG('rec1_410.mat');
data2 = DSpace2MatNG('rec1_411.mat');
data3 = DSpace2MatNG('hyundai_data1.mat');
data4 = DSpace2MatNG('hyundai_data2.mat');
data5 = DSpace2MatNG('hyundai_level2_3_101725.mat');
data6 = DSpace2MatNG('hyundai_data_11212025.mat');

plot_chunks = 0;
%% Analyze for mode = 1
meanGrade = []; meanPower = [];
[meanGrade1, meanPower1] = analyzeModeChunks(data1, 'Data1', 1, 1000, plot_chunks); meanGrade = [meanGrade', meanGrade1']; meanPower = [meanPower, meanPower1'];
[meanGrade2, meanPower2] = analyzeModeChunks(data2, 'Data1', 1, 1000, plot_chunks); meanGrade = [meanGrade, meanGrade2']; meanPower = [meanPower, meanPower2'];
[meanGrade3, meanPower3] = analyzeModeChunks(data3, 'Data1', 1, 1000, plot_chunks); meanGrade = [meanGrade, meanGrade3']; meanPower = [meanPower, meanPower3'];
[meanGrade4, meanPower4] = analyzeModeChunks(data4, 'Data1', 1, 1000, plot_chunks); meanGrade = [meanGrade, meanGrade4']; meanPower = [meanPower, meanPower4'];
[meanGrade5, meanPower5] = analyzeModeChunks(data5, 'Data1', 2, 1000, plot_chunks); meanGrade = [meanGrade, meanGrade5']; meanPower = [meanPower, meanPower5'];
[meanGrade6, meanPower6] = analyzeModeChunks(data6, 'Data1', 1, 150, 1); 
% meanGrade = [meanGrade, meanGrade6']; meanPower = [meanPower, meanPower6'];

% meanPower6(2) = meanPower6(2)*1.2;
% meanPower6(7) = meanPower6(7)*1.5;
% meanPower6(11) = meanPower6(11)*2;

% for k = 1:length(meanGrade5_2)
%     meanGrade5_3(k) = meanGrade5_2(k)* (1+(rand(1)-0.5)*0.25);
%     meanPower5_3(k) = meanPower5_2(k)* (1+(rand(1)-0.5)*0.25);
% end

i_ZeroGarde = find(abs(meanGrade)<2);
MeanPowerAtZeroGrade = mean(meanPower(i_ZeroGarde));


% Scatter: Energy vs Grade
% Fit a quadratic polynomial (degree 2)
p = polyfit(meanGrade, meanPower, 2);
x_p = linspace(min(meanGrade), max(meanGrade), 200);
y_p = polyval(p, x_p);

figure(1000)
scatter(meanGrade, meanPower/1e6, 'filled', 'LineWidth',3);
hold on
scatter(meanGrade6, meanPower6/1e6, 'k', 'filled', 'LineWidth',3);
% scatter(meanGrade5_2, meanPower5_2/1e6,'r', 'filled', 'LineWidth',3);
% scatter(meanGrade5_3, meanPower5_3/1e6,'g', 'filled', 'LineWidth',3);

plot(x_p, y_p/1e6, 'r-.', 'LineWidth',3);
hold on
p_mod = p; p_mod(1:end-1) = (1500+150)/1250* 14/10*p_mod(1:end-1);
plot(x_p, polyval(p_mod, x_p)/1e6, 'g-.', 'LineWidth',3);

xlabel('Mean Pitch [Deg]'); title('Mean Energy per 1[KM] driving (MJ)');
ylabel('Energy [MJ]');
grid on
legend('Old Data Points','New Data Points', 'Nominal Energy Model', 'Energy Model adapted for mass change')

pred_eng =  polyval(p_mod, meanGrade6);
err = mean(abs(((pred_eng-meanPower6))./pred_eng) *100)



% Scatter: Energy vs Grade
mean_rel_power = (meanPower-MeanPowerAtZeroGrade)/MeanPowerAtZeroGrade*100;
p = polyfit(meanGrade, mean_rel_power, 2);
y_p = polyval(p, x_p);
figure(1001)
scatter(meanGrade, (meanPower-MeanPowerAtZeroGrade)/MeanPowerAtZeroGrade*100, 'filled');
hold on
plot(x_p, y_p, '-.', 'LineWidth',3);
xlabel('Mean Pitch [Deg]'); title('Relative (To Zero Grade) Mean Power Change (%)');
ylabel('Rel. Power [%]');
hold on
grid on


%% ------------------------------------------------------------------------
function [meanGrade, meanPower] = analyzeModeChunks(data, label, modeVal, max_time, plot_chunks)
% Splits data into contiguous intervals where
% data.Gway_Cluster_DrivingMode == modeVal, then:
%  • plots battery power, normalized altitude, pitch, wheel power,
%    wheel velocity, cumulative energy, trajectory (lat vs lon),
%    and traveled distance
%  • computes & scatters total energy vs mean road grade vs mean speed
%  • for Data1, splits chunk 7 into two at ~80 s

%-- Cast the key signals to double once
data.Gway_BatCur             = double(data.Gway_BatCur);
data.Gway_BatVol             = double(data.Gway_BatVol);
data.Gway_WhlTq              = double(data.Gway_WhlTq);
data.Gway_Wheel_Velocity_FL  = double(data.Gway_Wheel_Velocity_FL);
data.Speed3D                 = double(data.Speed3D);

%-- Identify mode-chunks
isMode = (data.Gway_Cluster_DrivingMode == modeVal);
edges = diff([0; isMode(:); 0]);
starts = find(edges==1);
ends   = find(edges==-1)-1;
nSeg   = numel(starts);

added_seg = 0;
% ensure column vectors once at the top of the function
starts = starts(:);
ends   = ends(:);

k = 1;
while k <= numel(starts)
    s0  = starts(k);
    e0  = ends(k);
    idx = s0:e0;

    tBat = data.GWAY3_time(idx);
    tRel = tBat - tBat(1);

    if tRel(end) > max_time
        cutTimes  = max_time : max_time : tRel(end);
        cutIdxRel = arrayfun(@(ct) find(tRel >= ct, 1, 'first'), cutTimes);

        % absolute cut indices (column vector)
        cutIdxAbs = s0 + cutIdxRel(:) - 1;

        % new segments within [s0, e0] (column vectors)
        newStarts = [s0;         cutIdxAbs + 1];
        newEnds   = [cutIdxAbs;  e0];

        % vertically concatenate pieces (all columns)
        starts = [starts(1:k-1); newStarts; starts(k+1:end)];
        ends   = [ends(1:k-1);   newEnds;   ends(k+1:end)];

        % skip past the subsegments we just inserted
        k = k + numel(newStarts);
    else
        k = k + 1;
    end
end

% for k = 1:nSeg
%     idx  = starts(k+added_seg):ends(k+added_seg);                  % indices for this segment
%     tBat = data.GWAY3_time(idx);              % time for this segment
% 
%     % elapsed time from start of this segment
%     tRel = tBat - tBat(1);
% 
%     if tRel(end) > max_time
%         % times at which we want to cut (max_time, 2*max_time, 3*max_time, ...)
%         cutTimes = max_time : max_time : tRel(end);
% 
%         % indices in this segment where elapsed time first reaches each cut time
%         cutIdxRel = arrayfun(@(ct) find(tRel >= ct, 1, 'first'), cutTimes);
% 
%         % relative start/end indices for all subsegments
%         ends   = [ends(1:k-1)', starts(k)+cutIdxRel,         ends(k:end)']';
%         starts = [starts(1:k)', ends(k)+1,          starts(k+1:end)']';
%         added_seg = added_seg+1;
%     end
% end

%-- For Data1, split chunk 7 at ~80s into two
% if strcmp(label,'Data1') && nSeg>=7
%   % indices of chunk 7
%   i0 = starts(7);
%   i1 = ends(7);
%   t7 = data.GWAY3_time(i0:i1);
%   t7_rel = t7 - t7(1);
%   % find split point where relative time >= 80 s
%   splitRel = 80;
%   j = find(t7_rel >= splitRel, 1, 'first');
%   if ~isempty(j) && j < numel(t7_rel)
%     splitIdx = i0 + j - 1;
%     % rebuild starts/ends with split
%     starts = [starts(1:6); i0; splitIdx+1; starts(8:end)];
%     ends   = [ends(1:6); splitIdx; i1; ends(8:end)];
%     nSeg = numel(starts);
%   end
% end

% Preallocate stats
totalEnergy_time = zeros(nSeg,1);
meanPower = zeros(nSeg,1);
meanGrade   = zeros(nSeg,1);
meanSpeed   = zeros(nSeg,1);
meanAC   = zeros(nSeg,1);

%-- Loop through each segment
ileg = 0;
nSeg   = numel(starts);
meanGrade = zeros(nSeg,1); meanPower = zeros(nSeg,1);
for k = 1:nSeg
    idx = starts(k):ends(k);

    % Battery power (W)
    tBat = data.GWAY3_time(idx);
    Pbat = data.Gway_BatCur(idx) .* data.Gway_BatVol(idx);

    % Altitude (m) and normalize to zero at start
    tAlt = data.Altitude_time(idx);
    alt  = data.PosAlt(idx);
    tAlt0 = tAlt - tAlt(1);
    alt0  = alt  - alt(1);

    % Road grade (°)
    tPitch = data.HeadingPitchRoll_time(idx);
    grade  = data.AnglePitch(idx);

    % Wheel power (W)
    tWhl = data.GWAY1_time(idx);
    Pwhl = data.Gway_WhlTq(idx) .* data.Gway_Wheel_Velocity_FL(idx);

    % Wheel velocity (FL)
    wheelVel = data.Gway_Wheel_Velocity_FL(idx);

    % Cumulative energy (J)
    cumEnergy = cumtrapz(tBat, Pbat);

    % Latitude & Longitude
    lat = data.PosLat(idx);
    lon = data.PosLon(idx);

    % Speed (m/s) and traveled distance
    tSpeed = tWhl;
    speed  = data.Gway_Wheel_Velocity_FL(idx) * 10/36;
    tSpeed0  = tSpeed - tSpeed(1);
    distTraveled = cumtrapz(tSpeed0, speed);

    if distTraveled(length(distTraveled)) < 250
        continue
    end

    % Compute segment stats
    totalEnergy_time(k) = trapz(tBat, Pbat-double(data.Gway_ACPwrCons(idx)));
    totalEnergy_dist(k) = trapz(distTraveled, Pbat-double(data.Gway_ACPwrCons(idx)));
    % meanPower(k) = totalEnergy(k)/distTraveled(length(distTraveled)) * 1000;
    meanPower(k) = totalEnergy_time(k)/distTraveled(length(distTraveled)) * 1000;
    meanPower(k) = totalEnergy_dist(k)/distTraveled(length(distTraveled)) * 1000;
    % meanPower(k) = mean(Pbat);
    meanGrade(k)   = mean(grade);
    meanSpeed(k)   = mean(speed);

    meanAC(k)   = mean(data.Gway_ACPwrCons(idx));

    if plot_chunks
        % Plot in 4x2 grid
        figure(100)
        set(gcf, 'Name', sprintf('%s – Mode %d', label, modeVal), ...
            'NumberTitle','off');
        subplot(3,2,1);
        plot(tBat - tBat(1), Pbat);
        xlabel('Time (s)'); ylabel('P_{bat} (W)');
        title('Battery Power'); hold on;
        subplot(3,2,2);
        plot(tAlt0, alt0);
        xlabel('Time (s)'); ylabel('Altitude (m)');
        title('PosAlt (zeroed)'); hold on;
        subplot(3,2,3);
        plot(tPitch - tPitch(1), grade);
        xlabel('Time (s)'); ylabel('Pitch (°)');
        title('Pitch Angle'); hold on;
        subplot(3,2,4);
        plot(tWhl - tWhl(1), speed);
        xlabel('Time (s)'); ylabel('Speed (m/s)');
        title('Vehicle Speed'); hold on;
        subplot(3,2,5);
        plot(tBat - tBat(1), cumEnergy);
        xlabel('Time (s)'); ylabel('Cum. Energy (J)');
        title('Cumulative Energy'); hold on;
        subplot(3,2,6);
        plot(tSpeed0, distTraveled);
        xlabel('Time (s)'); ylabel('Distance (m)');
        title('Traveled Distance'); hold on;

        leg_str{ileg+1} = sprintf('Chunk %d', k);
        ileg = ileg+1;
        % disp(k);
        % disp(meanGrade(k));
        % disp(meanSpeed(k));
        % disp(meanAC(k));
    end


end
if plot_chunks
    legend(leg_str)
end

for k = nSeg:-1:1
    if meanGrade(k)==0 && meanPower(k)==0
        meanGrade(k) = [];
        meanPower(k) = [];
    end

end

% 3D plot: Energy vs Grade vs Speed
% figure(fig_3d)
% set(fig_3d, 'Name', sprintf('%s – 3D Energy vs Grade vs Speed (Mode %d)', label, modeVal), ...
%        'NumberTitle','off');
% scatter3(meanGrade, meanSpeed, totalEnergy, 'filled');
% xlabel('Mean Pitch (°)'); ylabel('Mean Speed (m/s)'); zlabel('Total Energy (J)');
% title(sprintf('%s: Energy vs. Grade vs. Speed (Mode %d)', label, modeVal));
% hold on
end
