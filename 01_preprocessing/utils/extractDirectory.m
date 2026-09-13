function struct_data = extractDirectory(root_path)
% Build a directory structure comprising time-points, groups, and filenames.
%
%   Syntax:
%       S = extractDirectory(root_path)
%
%   Input:
%       root_path  : Root folder to scan (e.g. "E:\...\DATOS\EEG").
%
%   Output:
%       struct_data : Struct with fields:
%                       .instante  — cell array of time-point folder names
%                       .grupo     — cell array of group subfolders per time-point
%                       .datos     — nested cell array of filenames per group

    struct_data          = struct();
    struct_data.instante = getFolders(root_path);
    struct_data.grupo    = getSubFolders(root_path);
    struct_data.datos    = getFiles(root_path, struct_data);
end

% -------------------------------------------------------------------------
function folders = getFolders(root_path)
% Return the names of all non-system folders in root_path.
    entries = dir(root_path);
    folders = {};
    for i = 1:length(entries)
        name = entries(i).name;
        if ~strcmp(name, '.') && ~strcmp(name, '..')
            folders{end+1} = name;
        end
    end
end

% -------------------------------------------------------------------------
function subfolders = getSubFolders(root_path)
% Return, for each folder in root_path, the list of its sub-folder names.
    entries    = dir(root_path);
    subfolders = {};
    for j = 1:length(entries)
        name = entries(j).name;
        if ~strcmp(name, '.') && ~strcmp(name, '..')
            subfolders{end+1} = getFolders(fullfile(root_path, name));
        end
    end
end

% -------------------------------------------------------------------------
function all_files = getFiles(root_path, s)
% Return all subject filenames organised by time-point and group.
%
%   Output layout:
%       all_files{i}    -> time-point i  (e.g. PRE, POST, SEGUIMIENTO)
%       all_files{i}{j} -> group j within time-point i (e.g. CONTROL, EXP)

    n_timepoints = length(s.instante);
    all_files    = {};

    for i = 1:n_timepoints
        groups = s.grupo{i};
        files  = {};

        for j = 1:length(groups)
            group_files = {};
            folder_path = fullfile(root_path, s.instante{i}, groups{j});
            entries     = dir(folder_path);

            for k = 1:length(entries)
                name = entries(k).name;
                % Skip system entries and known non-data files
                if ~strcmp(name, '.') && ~strcmp(name, '..') && ...
                   ~strcmp(name, 'INCIDENCIAS') && ~strcmp(name, 'desktop.ini')
                    group_files{end+1, 1} = name;
                end
            end

            files{1, j} = group_files;
            fprintf('%s\\%s -> DONE\n', s.instante{i}, groups{j});
        end

        all_files{i} = files;
    end
end
