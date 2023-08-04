function [points] = collimate_points_in_dir(im_dir, varargin)
%Wrapper for the function for collimating n points on all the images within
%a directory
%  
%  points = collimate_points_in_dir(im_dir_path, npts=1, axLims=axLims, printPoints=false);
% 
%  eg. points = collimate_points_in_dir('../data/img/p2', im_ext='.jpg', npts=1, search_win = 15, out_path = 'targets_p2.txt');
% 

%% Prase input
def_npts = Inf;
def_im_ext = '.tif';
def_out_path = 'collimated_points.txt';
def_ax_lim = {};
def_search_win = 20;
def_printsOut = false;

p = inputParser;
addRequired(p,'im', @ischar);
addParameter(p,'im_ext', def_im_ext, @ischar)
addParameter(p,'npts', def_npts, @isnumeric)
addParameter(p,'out_path', def_out_path, @ischar)
addParameter(p,'axLims', def_ax_lim)
addParameter(p,'search_win', def_search_win, @isnumeric)
addParameter(p,'printPoints', def_printsOut, @islogical)

parse(p,im_dir,varargin{:})
im_ext = p.Results.im_ext;
npts  = p.Results.npts;
out_path  = p.Results.out_path;
axLims = p.Results.axLims;
search_win = p.Results.search_win;
printPoints  = p.Results.printPoints;

%% 

%- initialize image datastore 
imds = imageDatastore(im_dir, ...
    'FileExtensions',{im_ext}, 'LabelSource','foldernames'); 

%- initialize other variables
num_imgs = length(imds.Files);
axLims = {};
points = {};

%- Call collimate_points function
fprintf('Collimation Started. collimating %i points in %i images...\n', npts, num_imgs)
for i=1:num_imgs
    im = readimage(imds, i);
    if i>1
        axLims = {[points{i-1}(1)-search_win, points{i-1}(1)+search_win], ...
                    [points{i-1}(2)-search_win, points{i-1}(2)+search_win] };
    end
    points{i} = collimate_points(im, npts=1, axLims=axLims);
end
fprintf('Collimation completed. Save points to save...\n')

% Save points to file
f = fopen(out_path, 'w');
fprintf(f, '#x,y\n');
for i = 1:length(points)
    fprintf(f, '%.4f,%.4f\n', points{i});
end
fclose(f);

fprintf('Done.\n')

end