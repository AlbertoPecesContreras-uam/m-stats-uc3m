function struct_channels = readEDF(filepath)
% Read an EDF file and return all channel data as a struct.
%
%   Syntax:
%       S = readEDF(filepath)
%
%   Input:
%       filepath : Full path to the .edf file.
%
%   Output:
%       struct_channels : Struct with channel names as fields and
%                         (1 x n_samples) double arrays as values.

    data           = edfread(filepath);
    channel_labels = data.Properties.VariableNames;
    n_channels     = length(channel_labels);
    n_seconds      = height(data);

    struct_channels = struct();

    for j = 1:n_channels
        channel_j = [];
        for i = 1:n_seconds
            % Each cell contains a 500-sample window (one second at 500 Hz)
            window    = data(i, j).(1);
            samples   = window{1, 1};
            channel_j = [channel_j; samples];
        end
        struct_channels.(channel_labels{j}) = channel_j';
    end
end
