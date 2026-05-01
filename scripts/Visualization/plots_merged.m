%% Dunkle_ML Visualization Master Script
% This merged MATLAB script collects all plot workflows used across the
% repository and points them to the new folder structure.
%
% Inputs are read from:
%   - data/processed/
%   - scripts/Phase I - hc and he discovery/<technique>/
%   - scripts/Phase II - ODE discovery/<technique>/
%
% Outputs are the figures created by this script, while the training scripts
% already write their own CSV/XLSX/PNG artifacts into their local folders.

clear; clc;
setup_plot_style();
paths = repo_paths();

%% Phase 0 - Synthetic Data Overview
% Input:
%   - data/processed/dunkle_clean.csv
% Output:
%   - two-panel overview figure for I/Ta and Tw/Tgi
plot_dunkle_overview(fullfile(paths.processed, 'dunkle_clean.csv'));

%% Phase 0 - Experimental Data Overview
% Input:
%   - data/processed/Experimental_Data_Cleaned_Final.xlsx
%   - Experimental workbook is restricted; request access before using it.
% Output:
%   - one figure per day block, matching the original MATLAB style
plot_experimental_overview(fullfile(paths.processed, 'Experimental_Data_Cleaned_Final.xlsx'));

%% Phase II - PINN
% Inputs produced by the training script:
%   - scripts/Phase II - ODE discovery/PINN/phase2_pinn_rollout_Tw.csv
%   - scripts/Phase II - ODE discovery/PINN/phase2_pinn_loss_curves.png
% Output:
%   - PINN rollout figure plus the saved loss-curve image
plot_phase2_pinn(
    fullfile(paths.phase2Pinn, 'phase2_pinn_rollout_Tw.csv'), ...
    fullfile(paths.phase2Pinn, 'phase2_pinn_loss_curves.png') ...
);

%% Phase II - pySR
% Inputs produced by the training script:
%   - scripts/Phase II - ODE discovery/pySR/phase2_pysr_predictions_dTw_dt.xlsx
%   - scripts/Phase II - ODE discovery/pySR/phase2_pysr_training_report_dTw_dt.xlsx
% Output:
%   - PySR rollout figure for Tw and the saved report workbook
plot_phase2_pysr(
    fullfile(paths.phase2PySR, 'phase2_pysr_predictions_dTw_dt.xlsx'), ...
    fullfile(paths.phase2PySR, 'phase2_pysr_training_report_dTw_dt.xlsx') ...
);

%% Phase II - SINDy Synthetic
% Inputs produced by the training script:
%   - scripts/Phase II - ODE discovery/SINDy - Synthetic/phase2_sindy_rollout_Tw.csv
% Output:
%   - SINDy rollout figure for Tw
plot_phase2_sindy_synthetic(fullfile(paths.phase2SindySynthetic, 'phase2_sindy_rollout_Tw.csv'));

%% Phase II - SINDy Experimental
% Inputs produced by the training script:
%   - scripts/Phase II - ODE discovery/SINDy - Experimental/Phase_II_SINDy_exp_results.xlsx
%   - scripts/Phase II - ODE discovery/SINDy - Experimental/Phase_II_SINDy_exp_plots/
% Output:
%   - The saved per-day rollout plots and iteration-error plot are displayed
show_png_folder(paths.phase2SindyExpPlots, 'Phase II - SINDy Experimental outputs');

%% Phase I - NN
% Inputs produced by the training script:
%   - scripts/Phase I - hc and he discovery/Baseline NN/phase1_nn_forward_predictions.csv
%   - scripts/Phase I - hc and he discovery/Baseline NN/phase1_nn_metrics.csv
% Output:
%   - overlay plots for hc and he
plot_phase1_nn(
    fullfile(paths.phase1NN, 'phase1_nn_forward_predictions.csv'), ...
    fullfile(paths.phase1NN, 'phase1_nn_metrics.csv'), ...
    fullfile(paths.processed, 'dunkle_clean.csv') ...
);

%% Phase I - pySR
% Inputs produced by the training script:
%   - scripts/Phase I - hc and he discovery/pySR/overlay_discovered_vs_actual_*.png
%   - scripts/Phase I - hc and he discovery/pySR/mse_progress_*.png
% Output:
%   - Saved PySR output plots are displayed directly
show_png_folder(paths.phase1PySR, 'Phase I - pySR output plots');

%% Phase I - SINDy Trial 2
% Inputs produced by the training script:
%   - scripts/Phase I - hc and he discovery/SINDy - Trial 2/*.png
%   - scripts/Phase I - hc and he discovery/SINDy - Trial 2/*.csv
% Output:
%   - Saved Trial 2 figures are displayed directly
show_png_folder(paths.phase1SindyTrial2, 'Phase I - SINDy Trial 2 output plots');

%% Phase I - SINDy Trial 3
% Inputs produced by the training script:
%   - scripts/Phase I - hc and he discovery/SINDy - Trial 3/*.png
%   - scripts/Phase I - hc and he discovery/SINDy - Trial 3/*.csv
% Output:
%   - Saved Trial 3 figures are displayed directly
show_png_folder(paths.phase1SindyTrial3, 'Phase I - SINDy Trial 3 output plots');

%% Local Functions
function setup_plot_style()
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
end

function paths = repo_paths()
    scriptDir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(fileparts(scriptDir));

    paths.repo = repoRoot;
    paths.processed = fullfile(repoRoot, 'data', 'processed');

    paths.phase1NN = fullfile(repoRoot, 'scripts', 'Phase I - hc and he discovery', 'Baseline NN');
    paths.phase1PySR = fullfile(repoRoot, 'scripts', 'Phase I - hc and he discovery', 'pySR');
    paths.phase1SindyTrial2 = fullfile(repoRoot, 'scripts', 'Phase I - hc and he discovery', 'SINDy - Trial 2');
    paths.phase1SindyTrial3 = fullfile(repoRoot, 'scripts', 'Phase I - hc and he discovery', 'SINDy - Trial 3');

    paths.phase2Pinn = fullfile(repoRoot, 'scripts', 'Phase II - ODE discovery', 'PINN');
    paths.phase2PySR = fullfile(repoRoot, 'scripts', 'Phase II - ODE discovery', 'pySR');
    paths.phase2SindySynthetic = fullfile(repoRoot, 'scripts', 'Phase II - ODE discovery', 'SINDy - Synthetic');
    paths.phase2SindyExperimental = fullfile(repoRoot, 'scripts', 'Phase II - ODE discovery', 'SINDy - Experimental');
    paths.phase2SindyExpPlots = fullfile(paths.phase2SindyExperimental, 'Phase_II_SINDy_exp_plots');
end

function plot_dunkle_overview(csvFile)
    require_file(csvFile, 'Synthetic processed data file is missing.');
    T = readtable(csvFile);
    t = T.time_hr;

    figure('Color','w','Position',[100 100 800 500]);

    subplot(2,1,1);
    yyaxis left;
    p1 = plot(t, T.I, 'k-', 'LineWidth', 1);
    ylabel('$I$ (W/m$^2$)', 'Interpreter','latex');
    ylim([-20 900]);

    yyaxis right;
    p2 = plot(t, T.T_a, 'k--', 'LineWidth', 1);
    ylabel('$T$ ($^\circ$C)', 'Interpreter','latex');
    ylim([20 32]);

    grid on;
    xlim([0 24]);
    xticks([0 6 12 18 24]);
    title('$I$ and $T_a$', 'Interpreter','latex');
    legend([p1 p2], {'$I$','$T_a$'}, 'Location','best');

    ax = gca;
    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = 'k';
    ax.XColor = 'k';
    ax.FontSize = 12;

    subplot(2,1,2);
    plot(t, T.Tw, 'k-', 'LineWidth', 1); hold on;
    plot(t, T.Tgi, 'k--', 'LineWidth', 1);
    grid on;
    xlim([0 24]);
    xticks([0 6 12 18 24]);
    ylabel('$T$ ($^\circ$C)', 'Interpreter','latex');
    xlabel('Time (h)', 'Interpreter','latex');
    title('$T_w$ and $T_{g}$', 'Interpreter','latex');
    legend({'$T_w$','$T_{g}$'}, 'Location','best');
    set(gca, 'FontSize', 12, 'XColor','k', 'YColor','k');
end

function plot_experimental_overview(xlsxFile)
    require_file(xlsxFile, ['Experimental workbook not found: ' xlsxFile newline ...
        'Request access to the experimental data before running this section.']);

    C = readcell(xlsxFile);

    timeCols = find(strcmpi(string(C(2,:)), 'Time'));
    if isempty(timeCols)
        error('Could not find any "Time" headers in row 2.');
    end

    for k = 1:numel(timeCols)
        c0 = timeCols(k);
        dayLabel = string(C{1,c0});
        if strlength(dayLabel) == 0 || ismissing(dayLabel)
            dayLabel = "Day " + k;
        end

        headers = string(C(2, c0:c0+4));
        if ~all(ismember(["Time","Ta","Tg","Tw","I"], headers))
            error('Block %d does not contain the expected headers.', k);
        end

        data = C(3:end, c0:c0+4);
        isEmpty = cellfun(@(x) isempty(x) || (isstring(x) && strlength(x)==0), data(:,1));
        data = data(~isEmpty, :);
        if isempty(data)
            continue;
        end

        timeStr = string(data(:, strcmp(headers, 'Time')));
        Ta = cell2mat(data(:, strcmp(headers, 'Ta')));
        Tg = cell2mat(data(:, strcmp(headers, 'Tg')));
        Tw = cell2mat(data(:, strcmp(headers, 'Tw')));
        I  = cell2mat(data(:, strcmp(headers, 'I')));

        tClock = datetime(timeStr, 'InputFormat', 'HH:mm:ss');
        t = hours(tClock - tClock(1));
        for j = 2:numel(t)
            if t(j) < t(j-1)
                t(j:end) = t(j:end) + 24;
            end
        end

        figure('Color','w','Position',[100 100 800 500]);

        subplot(2,1,1);
        plot(t, Tw, 'k-', 'LineWidth', 1); hold on;
        plot(t, Tg, 'k:', 'LineWidth', 1);
        plot(t, Ta, 'k-.', 'LineWidth', 1);
        grid on;
        xlim([0 24]);
        xticks([0 6 12 18 24]);
        xticklabels({'8 am','2 pm','8 pm','2 am','8 am'});
        ylabel('$T$ ($^\circ$C)', 'Interpreter','latex');
        title(dayLabel + ": $T_w$, $T_g$, $T_a$", 'Interpreter','latex');
        legend({'$T_w$','$T_g$','$T_a$'}, 'Location','best');
        ax = gca;
        ax.XColor = 'k';
        ax.YColor = 'k';
        ax.FontSize = 12;

        subplot(2,1,2);
        plot(t, I, 'k-', 'LineWidth', 1);
        grid on;
        xlim([0 24]);
        ylim([0 1100]);
        xticks([0 6 12 18 24]);
        xticklabels({'8 am','2 pm','8 pm','2 am','8 am'});
        ylabel('$I$ (W/m$^2$)', 'Interpreter','latex');
        xlabel('Time', 'Interpreter','latex');
        title(dayLabel + ": $I$", 'Interpreter','latex');
        legend({'$I$'}, 'Location','best');
        set(gca, 'FontSize', 12, 'XColor','k', 'YColor','k');
    end
end

function plot_phase2_pinn(rolloutCsv, lossPng)
    require_file(rolloutCsv, 'Phase II PINN rollout CSV not found.');
    require_file(lossPng, 'Phase II PINN loss plot not found.');

    R = readtable(rolloutCsv);
    t = R.time_s / 3600;

    figure('Color','w','Position',[100 100 800 500]);
    plot(t, R.Tw_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(t, R.Tw_pred_rollout, 'r-', 'LineWidth', 1.5);
    grid on;
    xlim([min(t) max(t)]);
    xlabel('Time (h)', 'Interpreter','latex');
    ylabel('$T_w$ ($^\circ$C)', 'Interpreter','latex');
    title('Phase II PINN: $T_w$ rollout', 'Interpreter','latex');
    legend({'Ground truth', 'PINN rollout'}, 'Location','best');

    if ismember('split', R.Properties.VariableNames)
        splitVals = string(R.split);
        trainIdx = find(splitVals == "train", 1, 'last');
        valIdx = find(splitVals == "val", 1, 'last');
        if ~isempty(trainIdx) && trainIdx < numel(t)
            xline(t(trainIdx), '--k', 'Train/Val', 'HandleVisibility','off');
        end
        if ~isempty(valIdx) && valIdx < numel(t)
            xline(t(valIdx), ':k', 'Val/Test', 'HandleVisibility','off');
        end
    end

    show_png(lossPng, 'Phase II PINN loss curves');
end

function plot_phase2_pysr(predictionsXlsx, reportXlsx)
    require_file(predictionsXlsx, 'Phase II PySR predictions workbook not found.');
    require_file(reportXlsx, 'Phase II PySR training report workbook not found.');

    rollout = readtable(predictionsXlsx, 'Sheet', 'rollout_tw');
    t = rollout.time_s / 3600;

    figure('Color','w','Position',[100 100 800 500]);
    plot(t, rollout.Tw_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(t, rollout.Tw_pred_rollout, 'r-', 'LineWidth', 1.5);
    grid on;
    xlim([min(t) max(t)]);
    xlabel('Time (h)', 'Interpreter','latex');
    ylabel('$T_w$ ($^\circ$C)', 'Interpreter','latex');
    title('Phase II PySR: $T_w$ rollout', 'Interpreter','latex');
    legend({'Ground truth', 'PySR rollout'}, 'Location','best');

    if ismember('split', rollout.Properties.VariableNames)
        splitVals = string(rollout.split);
        trainIdx = find(splitVals == "train", 1, 'last');
        valIdx = find(splitVals == "val", 1, 'last');
        if ~isempty(trainIdx) && trainIdx < numel(t)
            xline(t(trainIdx), '--k', 'Train/Val', 'HandleVisibility','off');
        end
        if ~isempty(valIdx) && valIdx < numel(t)
            xline(t(valIdx), ':k', 'Val/Test', 'HandleVisibility','off');
        end
    end

    try
        bestModel = readtable(reportXlsx, 'Sheet', 'best_model');
        if ~isempty(bestModel)
            fprintf('Phase II PySR best model validation MSE: %g\n', bestModel.val_mse(1));
        end
    catch
        % If the workbook sheet name changes, keep the plot workflow intact.
    end
end

function plot_phase2_sindy_synthetic(rolloutCsv)
    require_file(rolloutCsv, 'Phase II synthetic SINDy rollout CSV not found.');

    R = readtable(rolloutCsv);
    t = R.time_s / 3600;

    figure('Color','w','Position',[100 100 800 500]);
    plot(t, R.Tw_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(t, R.Tw_pred_rollout, 'r-', 'LineWidth', 1.5);
    grid on;
    xlim([min(t) max(t)]);
    xlabel('Time (h)', 'Interpreter','latex');
    ylabel('$T_w$ ($^\circ$C)', 'Interpreter','latex');
    title('Phase II SINDy (Synthetic): $T_w$ rollout', 'Interpreter','latex');
    legend({'Ground truth', 'SINDy rollout'}, 'Location','best');

    if ismember('split', R.Properties.VariableNames)
        splitVals = string(R.split);
        trainIdx = find(splitVals == "train", 1, 'last');
        valIdx = find(splitVals == "val", 1, 'last');
        if ~isempty(trainIdx) && trainIdx < numel(t)
            xline(t(trainIdx), '--k', 'Train/Val', 'HandleVisibility','off');
        end
        if ~isempty(valIdx) && valIdx < numel(t)
            xline(t(valIdx), ':k', 'Val/Test', 'HandleVisibility','off');
        end
    end
end

function plot_phase1_nn(predictionsCsv, metricsCsv, actualCsv)
    require_file(predictionsCsv, 'Phase I NN forward-prediction CSV not found.');
    require_file(metricsCsv, 'Phase I NN metrics CSV not found.');
    require_file(actualCsv, 'Phase I NN actual data CSV not found.');

    actual = readtable(actualCsv);
    predTbl = readtable(predictionsCsv);
    metrics = readtable(metricsCsv);

    if any(strcmpi(actual.Properties.VariableNames, 'time_s')) && any(strcmpi(predTbl.Properties.VariableNames, 'time_s'))
        key = 'time_s';
    elseif any(strcmpi(actual.Properties.VariableNames, 'index')) && any(strcmpi(predTbl.Properties.VariableNames, 'index'))
        key = 'index';
    else
        error('No overlapping time column found. Expected time_s or index in both actual and prediction files.');
    end

    hcJoin = innerjoin(actual(:, {key, 'hc'}), predTbl(:, {key, 'hc_pred'}), 'Keys', key);
    heJoin = innerjoin(actual(:, {key, 'he'}), predTbl(:, {key, 'he_pred'}), 'Keys', key);

    if isempty(hcJoin) || isempty(heJoin)
        error('No overlapping time values found between actual data and prediction file.');
    end

    if strcmpi(key, 'time_s')
        x_hc = hcJoin.(key) / 3600;
        x_he = heJoin.(key) / 3600;
        xLabel = 'Time (h)';
    else
        x_hc = hcJoin.(key);
        x_he = heJoin.(key);
        xLabel = 'Index';
    end

    if height(metrics) ~= 1
        error('phase1_nn_metrics.csv is expected to contain exactly one row.');
    end

    figure('Color','w','Position',[100 100 800 500]);
    subplot(2,1,1);
    plot(x_hc, hcJoin.hc, 'k-', 'LineWidth', 1); hold on;
    plot(x_hc, hcJoin.hc_pred, 'r-', 'LineWidth', 1);
    grid on;
    xlabel(xLabel, 'Interpreter','latex');
    ylabel('$h_c$ (W/m$^2$K)', 'Interpreter','latex');
    title('Phase I NN: $h_c$ forward overlay', 'Interpreter','latex');
    legend({'Actual', 'Predicted'}, 'Location','best');
    annotation('textbox', [0.13 0.74 0.24 0.14], 'String', { ...
        sprintf('$\\mathrm{Train\\ MSE}=%.6g$', metrics.hc_train_mse(1)), ...
        sprintf('$\\mathrm{Val\\ MSE}=%.6g$', metrics.hc_val_mse(1)), ...
        sprintf('$\\mathrm{Test\\ MSE}=%.6g$', metrics.hc_test_mse(1)) ...
        }, 'Interpreter', 'latex', 'FitBoxToText', 'on', 'BackgroundColor', 'w');

    subplot(2,1,2);
    plot(x_he, heJoin.he, 'k-', 'LineWidth', 1); hold on;
    plot(x_he, heJoin.he_pred, 'r-', 'LineWidth', 1);
    grid on;
    xlabel(xLabel, 'Interpreter','latex');
    ylabel('$h_e$ (W/m$^2$K)', 'Interpreter','latex');
    title('Phase I NN: $h_e$ forward overlay', 'Interpreter','latex');
    legend({'Actual', 'Predicted'}, 'Location','best');
    annotation('textbox', [0.13 0.30 0.24 0.14], 'String', { ...
        sprintf('$\\mathrm{Train\\ MSE}=%.6g$', metrics.he_train_mse(1)), ...
        sprintf('$\\mathrm{Val\\ MSE}=%.6g$', metrics.he_val_mse(1)), ...
        sprintf('$\\mathrm{Test\\ MSE}=%.6g$', metrics.he_test_mse(1)) ...
        }, 'Interpreter', 'latex', 'FitBoxToText', 'on', 'BackgroundColor', 'w');
end

function show_png_folder(folderPath, figTitle)
    if ~isfolder(folderPath)
        warning('%s folder not found: %s', figTitle, folderPath);
        return;
    end

    pngFiles = dir(fullfile(folderPath, '*.png'));
    if isempty(pngFiles)
        warning('No PNG files found in %s', folderPath);
        return;
    end

    [~, order] = sort({pngFiles.name});
    pngFiles = pngFiles(order);

    for k = 1:numel(pngFiles)
        filePath = fullfile(folderPath, pngFiles(k).name);
        show_png(filePath, sprintf('%s - %s', figTitle, pngFiles(k).name));
    end
end

function show_png(filePath, figTitle)
    require_file(filePath, sprintf('Image not found: %s', filePath));
    img = imread(filePath);
    figure('Color','w','Name',figTitle);
    image(img);
    axis image off;
    title(strrep(figTitle, '_', '\_'), 'Interpreter','none');
end

function require_file(filePath, message)
    if ~isfile(filePath)
        error('%s\nMissing file: %s', message, filePath);
    end
end