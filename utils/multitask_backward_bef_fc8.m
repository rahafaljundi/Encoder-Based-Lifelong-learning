function res_in=multitask_backward_bef_fc8(layer, res_in, res_out)
%MULTITASK_BACKWARD_BEF_FC8 is a custom backward function for the last
%fork in the model 
%This fork contains the specific task layers for each of  the tasks of the
%sequence
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

cudnn = {'NoCuDNN'} ;
leak = {} ;
clear res_in
res_in.dzdx=0;
res_out.dzdx=1;
for t=1: numel(layer.tasks)
    if (t<numel(layer.tasks))
          layer.tasks{t}.class=layer.tasks_targets{t};
    end
    max_number_of_layers=numel(res_out.aux{t}.layers);
    res_in.aux{t}.layers{max_number_of_layers}.dzdx=res_out.dzdx;
    for t_layer=numel(layer.tasks{t}.layers):-1:1
        t_l=layer.tasks{t}.layers{t_layer};
        switch t_l.type
            case 'conv'
                [res_in.aux{t}.layers{t_layer}.dzdx, res_in.aux{t}.layers{t_layer}.dzdw{1}, res_in.aux{t}.layers{t_layer}.dzdw{2}] = ...
                    vl_nnconv( res_out.aux{t}.layers{t_layer}.x , t_l.weights{1}, t_l.weights{2}, ...
                    res_in.aux{t}.layers{t_layer+1}.dzdx, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    cudnn{:}) ;
            case 'relu'
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnrelu(res_out.aux{t}.layers{t_layer}.x,res_in.aux{t}.layers{t_layer+1}.dzdx, leak{:}) ;
            case 'dropout'
                if layer.testMode
                    res_in.aux{t}.layers{t_layer}.dzdx = res_in.aux{t}.layers{t_layer+1}.dzdx ;
                else
                    res_in.aux{t}.layers{t_layer}.dzdx = vl_nndropout(res_out.aux{t}.layers{t_layer}.x, res_in.aux{t}.layers{t_layer+1}.dzdx, ...
                        'mask',res_out.aux{t}.layers{t_layer+1}.mask) ;
                end
            case 'softmaxloss'
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnsoftmaxloss(res_out.aux{t}.layers{t_layer}.x, layer.class, res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'softmax'
                
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnsoftmax(res_out.aux{t}.layers{t_layer}.x,res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'softmaxlossdiff'
                opts.mode='MI';
                opts.temperature=2;
                res_in.aux{t}.layers{t_layer}.dzdx =vl_nnsoftmaxdiff(res_out.aux{t}.layers{t_layer}.x, layer.tasks{t}.class, res_in.aux{t}.layers{t_layer+1}.dzdx,opts) ;
        end
    end
    %sumup the gradients from all the tasks
    if(~isfield(layer.tasks{t}.layers{1},'gr_weight'))
        layer.tasks{t}.layers{1}.gr_weight=1;
    end  
    res_in.dzdx=res_in.dzdx+ layer.tasks{t}.layers{1}.gr_weight*res_in.aux{t}.layers{1}.dzdx;
    
end

