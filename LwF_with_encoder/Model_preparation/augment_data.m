function [aug_imdb] = augment_data(imdb,net,output_path,varargin)
%AUGMENT_DATA computes the data augmentation
% Author: Rahaf Aljundi
% 
% See the COPYING file.

opts.normalize_encoder_input=true;
opts.aug_opts='augmentation_opts';
opts.output_layer_id= 15;
[opts, ~] = vl_argparse(opts, varargin) ;

net.layers{end}.type='softmax';

%preprocess the images
if(isfield(imdb,'imageDir'))
    im_links = strcat([imdb.imageDir filesep], imdb.images.name) ;
else
    im_links = imdb.images.data ;
end
aug_imdb=imdb;
aug_imdb.images.labels =[];
aug_imdb.images.set =[];
aug_imdb.images.data={};
aug_opts=load(opts.aug_opts);
if isfield(aug_opts, 'opts')
    aug_opts=aug_opts.opts;
end;


%getAverageImage
if(isfield(net.meta,'normalization'))
    mu = imresize(single(net.meta.normalization.averageImage), ...
        net.meta.normalization.imageSize(1:2)) ;
    aug_opts.train.subtractAverage=mu;
    aug_opts.test.subtractAverage=mu;
end
for i=1:numel(im_links)
    fprintf('Augmenting image %d out of %d\n', i, numel(im_links));
     if(imdb.images.set(i)==1)
        numAugments=aug_opts.numAugments;
        phase='train';
        
    else
        numAugments=1;
        phase='test';
        
     end
     
     %get the images and their augmentation
     for j=1:numAugments
         
         if(j==numAugments)
             phase='test';
         end
         data=getImageBatch(im_links(i), aug_opts.(phase));
         
         
         this_link=fullfile(output_path,strcat(num2str(i),'_',num2str(j),'.mat'));
         im=data;
         save(this_link,'im');
         aug_imdb.images.data{end+1}=this_link;
         aug_imdb.images.labels=[ aug_imdb.images.labels imdb.images.labels(i)];
         aug_imdb.images.set=[ aug_imdb.images.set imdb.images.set(i)];
     end
    
end








