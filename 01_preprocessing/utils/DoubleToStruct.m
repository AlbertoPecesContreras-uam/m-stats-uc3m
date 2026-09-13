function S_out = DoubleToStruct(double_in, channel_names)
% Convert a double matrix back to an EEG struct.
%
%   Syntax:
%       S = DoubleToStruct(X, channel_names)
%
%   Input:
%       double_in      : Matrix [n_channels x n_samples] where each
%                        row is one channel.
%       channel_names  : Cell array of channel name strings.
%
%   Output:
%       S_out : Struct with channel names as fields.

    S_out = struct();
    for i = 1:length(channel_names)
        S_out.(channel_names{i}) = double_in(i, :);
    end
end
