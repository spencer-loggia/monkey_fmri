function monk_nordic(inFile,noiseOption,outFile)
% Wrapper for NORDIC denoising functions.

% TODO: Make sure everything is formatted correctly.

if isstring(inFile)
    inFile = {inFile};
elseif ischar(inFile)
    inFile = {inFile};
end

% Temporarily holds options for NORDIC.
ARG_.noise_volume_last = noiseOption;
ARG_.kernel_size_PCA = [9 9 9];
ARG_.MP = 1;
ARG_.factor_error = 1.5;
ARG_.temporal_phase = 1;
ARG_.phase_filter_width = 10;
ARG_.save_add_info = 1;
ARG_.magnitude_only = 1;

inPhase = inFile;

ARG = repmat({ARG_},length(inFile));

for i = 1:length(inFile)
    % Using `fileparts` in this manner only returns the directory.
    ARG{i}.DIROUT = [fileparts(inFile{i}) '/'];
end

if ARG_.magnitude_only
    % Use a parallel for loop.
    parfor i = 1:length(inFile)
        NIFTI_NORDIC(inFile{i},inPhase{i},outFile,ARG{i})
    end
end
