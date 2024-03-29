function monk_nordic(func_in, noise_option,new_filename)
%%%
% Stuart J Duffield 06/27/2022
% A brief script to initialize parameters for NIFTI NORDIC
% Make sure you add your NIFTI NORDIC scripts to the matlab path
% Go to startup.m (in your default matlab folder) and add
% addpath(genpath('path/to/NIFTI_NORDIC/Folder'))
% funcs_in can either be a path or a cell struct of paths
% phase_in either a path to the noise or a  string 'None' / "None"
%%%
    if isstring(func_in)
        func_in = {func_in}
    end
    if ischar(func_in)
        func_in = {func_in}
    end
    
    ARG_temp.noise_volume_last = noise_option; % 0 if no noise volumes, > 0 if using noise

    ARG_temp.kernel_size_PCA = [9 9 9];
    ARG_temp.MP = 1;    
    ARG_temp.factor_error = 1.5 %  estimate for gfactor, high noise floor
    ARG_temp.temporal_phase = 1;
    ARG_temp.phase_filter_width=10;
    ARG_temp.save_add_info = 1;
    ARG_temp.magnitude_only = 1; % 1: do not use phase data; 
    phase_in = func_in; % This input will be ignored

    ARG = repmat({ARG_temp},length(func_in));
    a = {};
    if ARG_temp.magnitude_only == 1
        for i = 1:length(func_in)
            [a{i},~,~] = fileparts(func_in{i});
            ARG{i}.DIROUT = [a{i} '/'];
            NIFTI_NORDIC(func_in{i},phase_in{i},new_filename,ARG{i})
        end


    end
end