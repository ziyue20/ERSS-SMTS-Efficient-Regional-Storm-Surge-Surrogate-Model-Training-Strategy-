%% Minimal storm track & intensity visualization illustrative code
% This script demonstrates how to:
% 1) Plot storm tracks (lat/lon)
% 2) Compare intensity (Pc) distributions
% 3) Use different storm index sets
%
% This file is intended for public GitHub release.
% No raw data is included.



clear; clc; close all;

%% ------------------------------------------------------------------------
% User-defined data paths (PLACEHOLDERS)
% -------------------------------------------------------------------------
stormTrackMat = 'PATH/TO/storm_tracks.mat';   % <-- NEW
stormParamFile = 'PATH/TO/storm_parameters.xlsx';

%% ------------------------------------------------------------------------
% Load storm track time series from .mat file
% -------------------------------------------------------------------------
S = load(stormTrackMat);

% Expected structure:
% S.Track.StormID
% S.Track.Lat   (cell)
% S.Track.Lon   (cell)
Track = S.Track;

%% ------------------------------------------------------------------------
% Load storm intensity / metadata
% -------------------------------------------------------------------------
T = readtable(stormParamFile);
% Required columns: StormID | Pc

%% ------------------------------------------------------------------------
% Define storm index sets
% -------------------------------------------------------------------------
idx_all = Track.StormID;
idx_subset1 = idx_all(1:10);     % placeholder
idx_subset2 = idx_all(20:30);    % placeholder

%% ------------------------------------------------------------------------
% 1. Storm track visualization (from .mat time series)
% -------------------------------------------------------------------------
figure; hold on; box on;

plotTracksFromMat(Track, idx_all,     [0.8 0.8 0.8], 1.0);
plotTracksFromMat(Track, idx_subset1,[0.0 0.45 0.74], 2.0);
plotTracksFromMat(Track, idx_subset2,[0.85 0.33 0.10], 2.0);

xlabel('Longitude');
ylabel('Latitude');
title('Storm Tracks (from .mat)');
legend({'All storms','Subset 1','Subset 2'}, 'Location','best');
set(gca,'FontSize',14);

%% ------------------------------------------------------------------------
% 2. Intensity distribution (unchanged)
% -------------------------------------------------------------------------
figure; hold on; box on;

plotPDF(T.Pc(ismember(T.StormID,idx_all)),     'All storms');
plotPDF(T.Pc(ismember(T.StormID,idx_subset1)), 'Subset 1');
plotPDF(T.Pc(ismember(T.StormID,idx_subset2)), 'Subset 2');

xlabel('Central Pressure Pc (hPa)');
ylabel('PDF');
title('Storm Intensity Distribution');
legend('Location','best');
set(gca,'FontSize',14);

%% ------------------------------------------------------------------------
% Local functions
% -------------------------------------------------------------------------
function plotTracksFromMat(Track, stormIDs, color, lw)
% Plot storm tracks stored as time series in .mat file

for i = 1:length(stormIDs)
    id = stormIDs(i);
    k  = find(Track.StormID == id, 1);
    if isempty(k); continue; end

    lat = Track.Lat{k};
    lon = Track.Lon{k};

    plot(lon, lat, '-', 'Color', color, 'LineWidth', lw);
end
end

function plotPDF(x, labelStr)
x = x(~isnan(x));
if isempty(x); return; end
[f,xi] = ksdensity(x);
plot(xi, f, 'LineWidth', 2, 'DisplayName', labelStr);
end

