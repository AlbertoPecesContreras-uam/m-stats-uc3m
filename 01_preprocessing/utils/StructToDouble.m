function out_double = StructToDouble(S)
% Convert an EEG struct to a double matrix.
%
%   Syntax:
%       X = StructToDouble(S)
%
%   Input:
%       S          : Struct with channel names as fields and
%                    (1 x n_samples) signals as values.
%
%   Output:
%       out_double : Matrix [n_channels x n_samples] where each row
%                   is one channel.

    keys       = fieldnames(S);
    out_double = zeros(length(keys), length(S.(keys{1})));

    for i = 1:length(keys)
        out_double(i, :) = S.(keys{i});
    end
end
