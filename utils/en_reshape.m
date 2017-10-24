function out = en_reshape(x,dzdy)
%Reshapes x to feed to an autoencoder in forward pass, and reshapes it back
%to its original shape in backward pass.
%If size(x) = [H h W b], then its reshaped to [1 1 HxhxW b] to feed to the
%autoencoder
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

initial_dimensions=size(x);
if nargin <= 1|| isempty(dzdy)
     out=reshape(x,1,1,size(x,1)*size(x,1)*size(x,3),[]);
else
    out=reshape(dzdy,initial_dimensions);
end
