addpath('pca_ica\');

%% CONFIGURATION
data_path   = "E:\...\DATOS";
save_folder = "E:\...\PREPROCESADO";

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

%% BUILD OUTPUT DIRECTORY TREE
% Mirror the input folder structure under the output folder
EEG            = extractDirectory(data_path + '\EEG');
build_EEG_path = save_folder + "\EEG";
buildDirectory(save_folder, build_EEG_path, EEG);

%% PREPROCESSING
% Load global ICA parameters fitted in 01_fitGlobalICA.m
load('global_ica_parameters.mat');

W_oa_init = GlobalParams.W_OA;
W_oc_init = GlobalParams.W_OC;

info_ica_oa = preprocessRecordings(dir_oa, W_oa_init, build_EEG_path);
info_ica_oc = preprocessRecordings(dir_oc, W_oc_init, build_EEG_path);

%% SAVE CONVERGENCE LOG
current_dir = fileparts(mfilename('fullpath'));
log_path    = fullfile(current_dir, 'ica_convergence.csv');
writetable([info_ica_oa; info_ica_oc], log_path);
fprintf('ICA convergence log saved to %s\n', log_path);

%% LOCAL FUNCTIONS

function buildDirectory(folder, eeg_path, s)
% Create the output directory tree mirroring the input structure.
    if ~exist(folder,   'dir'), mkdir(folder);   end
    if ~exist(eeg_path, 'dir'), mkdir(eeg_path); end
    for i = 1:length(s.instante)
        inst_path = eeg_path + "\" + s.instante{i};
        if ~exist(inst_path, 'dir'), mkdir(inst_path); end
        for j = 1:length(s.grupo{i})
            grp_path = inst_path + "\" + s.grupo{i}{j};
            if ~exist(grp_path, 'dir'), mkdir(grp_path); end
        end
    end
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

function files = readDirectory(paths, fileExt, searchTag)
% Collect all matching file paths across a list of folders.
    folderResults = cell(length(paths), 1);
    for i = 1:length(paths)
        folderResults{i} = getFilePaths(paths{i}, fileExt, searchTag);
    end
    files = vertcat(folderResults{:});
end

function X_filt = removeBlinkArtifacts(Zw, Zica, T, W, sigma, mu)
% Identify and remove eye-blink ICA components by correlation with Fp1/Fp2.
% The top 2 components most correlated with the frontal average are zeroed
% out before the signal is reconstructed in channel space.

    n_components = size(Zica, 1);

    % Reference: average of Fp1 (ch 7) and Fp2 (ch 8), z-scored
    avg_Fp      = mean(Zw([7, 8], :), 1);
    avg_Fp_norm = (avg_Fp - mean(avg_Fp)) / std(avg_Fp);

    % Absolute correlation of each IC with the frontal reference
    % (absolute value because ICA does not fix the sign of components)
    correlations = zeros(n_components, 1);
    for i = 1:n_components
        comp_norm       = (Zica(i,:) - mean(Zica(i,:))) / std(Zica(i,:));
        corr_mat        = corrcoef(comp_norm, avg_Fp_norm);
        correlations(i) = abs(corr_mat(1, 2));
    end

    % --- 1. Identify artifact components (top 2 by correlation) ---
    [~, idx_ranking] = sort(correlations, 'descend');
    comps_to_remove  = idx_ranking(1:2);

    % --- 2. Zero out artifact components ---
    Zica_clean                    = Zica;
    Zica_clean(comps_to_remove,:) = 0;

    % --- 3. Reconstruct: project back to channel space and de-whiten ---
    Z_back = pinv(W) * Zica_clean;
    Zc_rec = pinv(T) * Z_back;

    % --- 4. De-scale and de-centre ---
    X_filt = (Zc_rec .* sigma) + mu;
end

function ica_table = preprocessRecordings(fileList, W_init, build_path)
% Apply the ICA-based eye-blink removal pipeline to a list of EDF recordings.
% Returns a table with per-file ICA convergence information.

    n_records = length(fileList);
    info_ica  = cell(n_records, 2);  % columns: path | delta

    band = {0.1, 30};
    fs   = 500;

    for i = 1:n_records

        search_path = fileList{i};
        fprintf('-) Reading file:\n\t%s\n', search_path);

        % --- Parse filename metadata ---
        [~, stem, ~] = fileparts(search_path);
        nameParts = strsplit(stem, '_');
        inst  = nameParts{2};  % PRE | POST | SEG
        group = nameParts{3};  % Control | EXP
        iden  = nameParts{4};  % subject ID
        task  = nameParts{5};  % GS | MAP | ...
        cond  = nameParts{6};  % OA | OC

        % --- Load and crop EEG ---
        S          = readEDF(search_path);
        channels   = fieldnames(S);
        double_EEG = StructToDouble(S);
        double_EEG = double_EEG(1:end-3, 2000:74901);  % drop edges & accel channels

        % --- Filter ---
        X = applyPassBand(double_EEG, fs, band);

        % --- Centre · Scale · Whiten ---
        [Zc, mu] = centerRows(X);        % mu    : (20 x 1)
        sigma    = std(X, 1, 2);         % sigma : (20 x 1)
        [Zw, T]  = whitenRows(Zc./sigma);

        % --- ICA (warm-started from global W) ---
        [~, W_rec, ~, ~, delta] = fastICA(Zw, 10, 'negentropy', 0, W_init);
        info_ica{i, 1} = search_path;
        info_ica{i, 2} = delta;

        % --- Project to IC space and remove blink artifacts ---
        Zica   = W_rec * Zw;                                               % (10 x N)
        X_filt = removeBlinkArtifacts(Zw, Zica, T, W_rec, sigma, mu)';    % (N x 20)

        % --- Build output table: [metadata | EEG channels] ---
        N        = size(X_filt, 1);
        info_mat = repmat({cond, inst, group, iden}, N, 1);
        info_tbl = cell2table(info_mat, 'VariableNames', {'cond','instant','group','id'});
        eeg_tbl  = array2table(X_filt, 'VariableNames', channels(1:end-3));
        out_table = [info_tbl, eeg_tbl];

        % --- Save CSV ---
        if inst == "SEG"
            inst = "SEGUIMIENTO";
        end
        save_path = build_path + "\" + inst + "\" + upper(group);
        filename  = inst + "_" + group + "_" + iden + "_" + task + "_" + cond + ".csv";
        writetable(out_table, fullfile(save_path, filename));

    end

    ica_table = cell2table(info_ica, 'VariableNames', {'path', 'delta'});
end
