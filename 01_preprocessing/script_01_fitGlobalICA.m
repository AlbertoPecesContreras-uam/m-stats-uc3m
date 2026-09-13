addpath('pca_ica\');

%% CONFIGURATION
EEG_paths = {{'E:\...\DATOS\EEG\PRE\CONTROL',         'E:\...\DATOS\EEG\PRE\EXP'}, ...
             {'E:\...\DATOS\EEG\POST\CONTROL',        'E:\...\DATOS\EEG\POST\EXP'}, ...
             {'E:\...\DATOS\EEG\SEGUIMIENTO\CONTROL', 'E:\...\DATOS\EEG\SEGUIMIENTO\EXP'}};

%% READING
% Collect EDF paths for eyes-open (OA) and eyes-closed (OC) conditions
dir_oa = {};
dir_oc = {};
for instante = 1:3
    dir_oa = [dir_oa; readDirectory(EEG_paths{instante}, '.edf', '_OA')];
    dir_oc = [dir_oc; readDirectory(EEG_paths{instante}, '.edf', '_OC')];
end

%% CONCATENATION
% global_OA_EEG : (20 x K),  K = m·N  (m = subjects, N = samples per recording)
% global_OC_EEG : (20 x K)
global_OA_EEG = concatenateSignals(dir_oa);
global_OC_EEG = concatenateSignals(dir_oc);

%% FILTERING
% Bandpass filter [0.1, 30] Hz, fs = 500 Hz
band = {0.1, 30};
fs   = 500;
global_OA_EEG = applyPassBand(global_OA_EEG, fs, band);
global_OC_EEG = applyPassBand(global_OC_EEG, fs, band);

%% PREPROCESSING — Centre · Scale · Whiten
% Centre rows (subtract row-wise mean)
[Zc_OA, mu_OA] = centerRows(global_OA_EEG);  % mu_OA : (20 x 1)
[Zc_OC, mu_OC] = centerRows(global_OC_EEG);  % mu_OC : (20 x 1)

% Row-wise standard deviation
std_OA = std(global_OA_EEG, 1, 2);  % (20 x 1)
std_OC = std(global_OC_EEG, 1, 2);  % (20 x 1)

% Whiten: divide each row i by std(i), then apply PCA whitening transform
[Zw_OA, T_OA] = whitenRows(Zc_OA ./ std_OA);
[Zw_OC, T_OC] = whitenRows(Zc_OC ./ std_OC);

%% ICA — Fit global model (10 components, negentropy criterion)
[~, W_OA, ~, ~, ~] = fastICA(Zw_OA, 10, 'negentropy', 1);
[~, W_OC, ~, ~, ~] = fastICA(Zw_OC, 10, 'negentropy', 1);

%% SAVE PARAMETERS
GlobalParams.mu_OA  = mu_OA;
GlobalParams.std_OA = std_OA;
GlobalParams.T_OA   = T_OA;
GlobalParams.W_OA   = W_OA;

GlobalParams.mu_OC  = mu_OC;
GlobalParams.std_OC = std_OC;
GlobalParams.T_OC   = T_OC;
GlobalParams.W_OC   = W_OC;

save('global_ica_parameters.mat', 'GlobalParams');
fprintf('Global ICA parameters saved to global_ica_parameters.mat\n');

%% LOCAL FUNCTIONS

function files = readDirectory(paths, fileExt, searchTag)
% Collect all matching file paths across a list of folders.
    folderResults = cell(length(paths), 1);
    for i = 1:length(paths)
        folderResults{i} = getFilePaths(paths{i}, fileExt, searchTag);
    end
    files = vertcat(folderResults{:});
end

function fileList = getFilePaths(folderPath, fileExt, searchTag)
% Return full paths of all files matching extension and tag in a folder.
    pattern  = fullfile(folderPath, ['*' searchTag '*' fileExt]);
    theFiles = dir(pattern);
    numFiles = length(theFiles);
    fileList = cell(numFiles, 1);
    for k = 1:numFiles
        fileList{k} = fullfile(folderPath, theFiles(k).name);
    end
end

function out = concatenateSignals(fileList)
% Read each EDF, discard accelerometer channels (last 3) and edge samples,
% then concatenate all recordings column-wise -> (20 x K).
    out = [];
    for i = 1:length(fileList)
        fprintf('-) Reading file:\n\t%s\n', fileList{i});
        S          = readEDF(fileList{i});
        double_EEG = StructToDouble(S);
        double_EEG = double_EEG(1:end-3, 2000:74901);  % drop edges & accel channels
        out        = [out, double_EEG];
    end
end
