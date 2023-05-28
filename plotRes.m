global styles alg_list ds alg_dispNames ds_dispNames lineWidth font_size base_line

alg_list = {'att_soft_triple', 'att_triplet_hinge', 'soft_triple', 'triplet_hinge'};
ds_list = {'plant_village', 'cub', 'cars'};
ds = ds_list{1}; gsplit = 0.01;
metric = 'Recall@1'; %'NMI';

alg_dispNames = containers.Map({'att_soft_triple', 'att_triplet_hinge', 'soft_triple', 'triplet_hinge'}, ...
     {'Soft Triple + GDFL', 'Triplet + GDFL', 'Soft Triple', 'Triplet Loss'});
n_alg = length(alg_list);

ds_dispNames = containers.Map({'plant_village', 'flowers102', 'cub', 'cars'}, ...
    {'Plant Village', 'Oxford 102 Flowers ','CUB-200-2011', 'CARS 196'});

lineWidth = 1.5; font_size = 14;
styles = {'o','*','s','d', 'h', 'p', '.', '^', 'v', 'x'};
fig_name = sprintf('%s_%s_%s', metric, base_line, ds);


hist_list = {};
nmi_list = cell(n_alg,1); recall1_list = cell(n_alg,1);
num_epoch = 1000;
for i=1:n_alg
     [best_score, best_params, best_res, best_hist] = ...
          get_best_res(ds,alg_list{i}, gsplit);
     hist_list{i} = best_hist;
     nmi_list{i} = best_hist.nmi_list;
     recall1_list{i} = best_hist.recallK_list(:,1)';
     num_epoch = min(num_epoch, length(nmi_list{i}));
end
if (strcmpi(metric, 'recall@1'))
     vals = recall1_list;
else 
     vals = nmi_list;
end
plot_metric(metric, ds_dispNames(ds), vals,num_epoch, fig_name);


function  [best, best_params, best_res, best_hist] = get_best_res(ds,alg, gsplit)
    dir_name = sprintf('.\\Res_%s\\',ds);
    files = dir(dir_name);
    file_names = string({files.name});
    best = 0; best_params = []; best_res = []; best_hist = [];
    for f=file_names
         fname = fullfile(dir_name, f);
         m = split(f, '__'); m = m{1};
         if (~endsWith(f, '.mat') || ~strcmp(m, alg))
              continue;
         end
         load(fname, 'best_score', 'params', 'res', 'hist');
         if (params.gallery_split ~= gsplit)
              continue;
         end

         if (best_score > best)
              best = best_score;
              best_params = params;
              best_res = res;
              best_hist = hist;
        end
    end
end


function plot_metric(metric, ds, vals, num_epoch, fig_name)
     global styles alg_dispNames alg_list lineWidth font_size base_line

    figure;
    epochs = 1:num_epoch;
    n_alg = length(vals);
    alg_names = cell(n_alg,1);
    for i=1:n_alg
         v = vals{i}; v = v(1:num_epoch);
         plot(epochs, v, ['-',styles{i}],'LineWidth',lineWidth, 'MarkerSize', 6);
         alg_names{i} = alg_dispNames(alg_list{i});
         hold on;
    end
    ax = gca;
    ax.FontWeight = 'bold'; 
    title(sprintf('Test %s on the %s dataset', metric,upper(ds)),'FontSize',font_size+2);
    ylabel(metric,'FontSize',font_size, 'FontWeight','bold');
    xlabel('Epochs','FontSize',font_size, 'FontWeight','bold');
    % ylim([28 75])
    legend(alg_names,'Location','southwest','FontSize',font_size); 
    % dim = [.4 .2 .3 .3];
    % a = annotation('textbox',dim,'String',alg_dispNames(base_line), 'FitBoxToText','on','FontSize',16, 'Interpreter','latex');
    % delete(findall(gcf,'type','annotation'))
    if(~isempty(fig_name))
         fname = fullfile('.\mat_figs',fig_name);
%          saveas(gcf, [fname,'.eps']);
         saveas(gcf, [fname,'.fig']);
    end
end


% function plot_vs(metric, ds, vals, vs_vals, vs_param, base_line, base_line_val, fig_name)
%      global styles alg_dispNames alg_list lineWidth font_size base_line
% 
%     figure;
%     epochs = 1:num_epoch;
%     n_alg = length(vals);
%     alg_names = cell(n_alg,1);
%     for i=1:n_alg
%          v = vals{i}; v = v(1:num_epoch);
%          plot(epochs, v, ['-',styles{i}],'LineWidth',lineWidth, 'MarkerSize', 6);
%          alg_names{i} = alg_dispNames(alg_list{i});
%          hold on;
%     end
%     ax = gca;
%     ax.FontWeight = 'bold'; 
%     title(sprintf('Test %s on the %s dataset', metric,upper(ds)),'FontSize',font_size+2);
%     ylabel(metric,'FontSize',font_size, 'FontWeight','bold');
%     xlabel('Epochs','FontSize',font_size, 'FontWeight','bold');
%     % ylim([28 75])
%     legend(alg_names,'Location','southwest','FontSize',font_size); 
%     % dim = [.4 .2 .3 .3];
%     % a = annotation('textbox',dim,'String',alg_dispNames(base_line), 'FitBoxToText','on','FontSize',16, 'Interpreter','latex');
%     % delete(findall(gcf,'type','annotation'))
%     if(~isempty(fig_name))
%          fname = fullfile('.\mat_figs',fig_name);
% %          saveas(gcf, [fname,'.eps']);
%          saveas(gcf, [fname,'.fig']);
%     end
% end
























    
    
    
