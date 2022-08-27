%% Belvedere Stereo - Relative Ori - CNN
clear; clc; close all;

%%

%-
rootDirPath = ('.\'); % 'D:\francesco\belvedereStereoCNN'); % 'Y:\francesco\belvedereStereoCNN'); %

%-
imFld           = 'IMG'; % 'IMG_subsampled'; %
imExt           = '.tif'; %'.JPG';  % 
calibFld        = 'calib';
% targetsWorldCoordFlmn = 'GCPs_daModelloDrone.txt';

numCams         = 2;
 
%-
undistimgs      = true;
enhanceimgs     = true;
printFigs       = false;
useTiles        = false;

%-
warpImages      = false;
% matchingMethod  = 'SG';

%% Add path to toolboxes
addpath(genpath(fullfile(rootDirPath,'\mat')));

% rmpath(fullfile(rootDirPath,'\mat\thirdParts\vlfeat\toolbox\kmeans\'));
% rmpath(fullfile('C:\vlfeat\toolbox\kmeans\'));

%% Initialize structures for storing data
cameras         = struct();                                                 % struct for storing cameras information
images          = struct('imds', [], 'exif', []);                           % struct for storing image datastore strctures
im              = cell(1, numCams);                                         % Cell array for storing image pairs

matchedPts      = struct();                                                 % struct for storing double points found at each epoch by DFM or SuperGlue
trackedPoints   = struct();                                                 % struct for storing tracked points for each camera in time
features        = struct();                                                 % struct for storing all the valid matched features at all epochs
sparsePts       = struct('labels', [], 'pointsXYZ', []);                    % struct for storing tra at all epochs


% fMats = struct('F', []);

maskBB          = [];
maskGlacier     = [];

 
%% Load data
fprintf('Loading data:...')

%- Get lists of images and calibration data
d = dir(fullfile(rootDirPath, imFld)) ;
% d = d([d(:).isdir] & ~ismember({d(:).name},{'.','..'}));
% d = d(ismember({d(:).name},{'p1','p2'}));
d = d(ismember({d(:).name},{'p2','p3'}));
c = dir(fullfile(rootDirPath, calibFld,'*.mat'));                           % Rivedere assegnazione file calibrazione a camera.

%- inizialize camera structures
for jj = 1:numCams
    a = load(fullfile(rootDirPath, calibFld, c(jj).name));      
    cameras(jj).cameraParams   = a.cameraParams;  
    cameras(jj).K       = cameras(jj).cameraParams.IntrinsicMatrix';
    cameras(jj).t{1}    = zeros(3,1);     
    cameras(jj).R{1}    = eye(3,3);
    cameras(jj).P{1}    = cameras(jj).K * [cameras(jj).R{1} cameras(jj).t{1}];
    cameras(jj).X0{1}   = - cameras(jj).P{1}(:,1:3)\cameras(jj).P{1}(:,4);
end

%- inizialize image datastore structures
warning off
for jj = 1:numCams
    images(jj).imds    = imageDatastore(fullfile(rootDirPath, imFld, d(jj).name),'FileExtensions',{imExt}, 'LabelSource','foldernames'); 
    for imgId = 1:length(images(jj).imds.Files)
        images(jj).exif    = imfinfo(images(jj).imds.Files{imgId});
    end
end
clearvars a c d jj
warning on      

%- Load mask for cropping images  (To do: allow also polygon mask)
if exist(fullfile(rootDirPath, imFld,'mask.mat'), 'file')
    load(fullfile(rootDirPath, imFld,'mask.mat'))
else
    maskBB = struct('roi', []);
    for jj = 1:numCams
        mskFig = figure('Name', 'Mask');     
        im{jj} = readimage(images(jj).imds,1); 
        ax = mskFig.CurrentAxes; 
        imshow(im{jj});  
        roi = drawrectangle(ax);
        maskBB(jj).roi = roi;
        maskBB(jj).pos = round(maskBB(jj).roi.Position/100)*100;
    end
    save(fullfile(rootDirPath, imFld,'mask.mat'), 'maskBB')    
    close(mskFig), clearvars ax fig roi
end

%- load mask on glacier
if exist(fullfile(rootDirPath, imFld,'maskGlacier.mat'), 'file')
    load(fullfile(rootDirPath, imFld,'maskGlacier.mat'))
else
    maskGlacier = struct('roi', []);
    for jj = 1:numCams
        mskFig = figure('Name', 'Mask');     
        im{jj} = readimage(images(jj).imds,1); 
        ax = mskFig.CurrentAxes; 
        imshow(im{jj});  
        roi = drawpolygon(ax);
        maskGlacier(jj).roi = roi;
    end
    save(fullfile(rootDirPath, imFld,'maskGlacier.mat'), 'maskGlacier')    
    close(mskFig), clearvars ax fig roi I1
end

%- load DFM network
% dfmModel = load(fullfile(rootDirPath, 'mat', 'thirdParts', 'DFM\models\imagenet-vgg-verydeep-19.mat'));

fprintf('Done.\n')

%% Warping following epochs to first one (master camera) ---> TO PUT IN A SEPARATE FUNCTION!
% epoches2process = 0:9; 
% homographyMats  = struct('Htform', [], 'H', []);
% 
% for epoch = epoches2process+1    %- Warp images to reference camera (first epoch) by homography
%     for jj = 1:numCams
%         fprintf('Warping image of camera %i to reference camera:\n', jj)
%         
%         fprintf('Detecting features... ')
%         im{1} = adapthisteq(rgb2gray(readimage(images(jj).imds, 1))); 
%         im{2} = adapthisteq(rgb2gray(readimage(images(jj).imds, epoch))); 
% 
%         dpts = struct();
%         for i = 1:2
% 
%             %- Detect ORB features
%             dpts(i).pts = detectORBFeatures(im{i},'ScaleFactor',1.001,'NumLevels',8);
% %             dpts(i).pts = detectKAZEFeatures(im{i});
% %             dpts(i).pts = detectSURFFeatures(im{i}, 'MetricThreshold', 1000);
% 
%             %- Reject features on moving glacier
%             inroi = inROI(maskGlacier(jj).roi, double(dpts(i).pts.Location(:,1)),  double(dpts(i).pts.Location(:,2)));
%             dpts(i).pts = dpts(i).pts(~inroi);
% 
%             %- Select n strongest points and extract features
%             dpts(i).strongestPts = selectStrongest(dpts(i).pts, 2e4);
%             [dpts(i).features, dpts(i).vpts] = extractFeatures(im{i}, dpts(i).strongestPts);
%         end
% 
%         fprintf('Matching features... ')
% %         indexPairs = matchFeatures(dpts(1).features, dpts(2).features, 'Unique', true);
%         indexPairs = matchFeaturesInRadius(dpts(1).features, dpts(2).features,...
%             dpts(2).vpts.Location,dpts(1).vpts.Location, 50, Unique=true, ...
%             MaxRatio=0.8, MatchThreshold=15);
%         dpts(1).matches     = dpts(1).vpts(indexPairs(:,1),:);
%         dpts(2).matches     = dpts(2).vpts(indexPairs(:,2),:);
% 
%         %- save matching to file
%         matchFig = figure("Units","normalized", "Position",[.1 .1 .8 .8],'visible','off'); 
%         tiledlayout('flow', 'TileSpacing', 'tight', 'Padding','tight', 'TileIndexing','columnmajor');
%         nexttile; imshow(im{1}); hold on; dpts(1).matches.plot
%         nexttile; imshow(im{2}); hold on; dpts(2).matches.plot
%         nexttile([2 3]); showMatchedFeatures(im{1}, im{1}, dpts(1).matches, dpts(2).matches)
%         pathparts = strsplit(images(jj).imds.Files{epoch},filesep); 
%         exportgraphics(matchFig, fullfile(pathparts{1:end-2}, 'warpingResults', pathparts{end-1:end}))
% 
%         fprintf('Estimating homography transformation... ')
%         [homographyMats(epoch).Htform, inliers1,inliers2] = estimateGeometricTransform(dpts(1).matches.Location, dpts(2).matches.Location,...
%         'projective','Confidence',99.99,'MaxNumTrials',5000,'MaxDistance',1.5);
%         fprintf('Percentage Inliers: %.1f%%\n', length(inliers1)/length(dpts(1).matches.Location)*100)
%         homographyMats(epoch).H = homographyMats(epoch).Htform.T';
%         
%         fprintf('Warping image... ')
%         im{1} = readimage(images(jj).imds, 1); 
%         im{2} = readimage(images(jj).imds, epoch); 
%         imwarped = uint8(HomographyWarping(im{2}, size(im{1}), homographyMats(epoch).H));
% 
%         %- visualize result
%         figure("Name", sprintf('Camera %i: image at epoch t0 and warped image from epoch t%i', jj, epoch-1));
%         imshowpair(im{1}, imwarped, 'falsecolor')
%         
%         %- Write image and homography matrices to disk
% %             if ~exist(fullfile(images(jj).imds.Folders{1},' warped'), "dir")
% %                 mkdir(fullfile(images(jj).imds.Folders{1},' warped'))
% %             end
%         imwrite(imwarped, fullfile(pathparts{1:end-1}, 'warped', pathparts{end}) )
% 
%         fprintf('Done.\n')
%     end
% end
% 
% save(fullfile(pathparts{1:end-1}, 'warpingResults', 'homographyMats'), 'homographyMats')
% clearvars dpts imwarped pathparts

if warpImages == true
%     epoches2process = 1; 
    epoches2process = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];     
    homography2masterCams(images, epoches2process, maskGlacier)
end 

%% Big-Loop

epoches2process = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];     
% epoches2process = [0:15];     
numEpoch2track  = 2;
for epoch = epoches2process+1
    fprintf('Processing epoch %i...\n', epoch-1)

    %------- From second epoch on, track previous features -----% 
%     if epoch > 1       
%         
%         SGdir = 'SGmatching';
%         im = cell(1, numCams);         
%         trackedPoints = struct();  % tmp var --> to be removed!!
% 
%         trackedFeatures = struct();
%         trackedFeatures(1).p{epoch} = [];
%         trackedFeatures(2).p{epoch} = [];
%         
%         epoch = 2
% %         prevEpoch = 2
%         prevEpoch = epoch-1;
%         while (epoch - prevEpoch) < numEpoch2track+1 && prevEpoch > 0
%             for jj = 1:numCams
%                 epochDir = fullfile(SGdir, ['t', num2str(epoch-1)], ['from_t', num2str(prevEpoch-1), '_cam', num2str(jj)]);
%                 if ~exist(epochDir, "dir"); mkdir(epochDir); end            
%                 prevEpochDir = fullfile(SGdir, ['t', num2str(prevEpoch-1)], 'res');           
%     %             for ii = epoch-1:epoch
%                 for ii = [prevEpoch, epoch]
%                     im{ii} = readimage(images(jj).imds, ii);
%                     if exist('maskBB', 'var')
%                         im{ii} = imcrop(im{ii}, maskBB(jj).pos); 
%                     end
%                 end
%                             
%                 %- Run SG on the whole image
%                 if ~exist(fullfile(epochDir, 'img'), "dir"); mkdir(fullfile(epochDir, 'img')); end        
%                 f = fopen(fullfile(epochDir, 'impairs.txt'), 'w');                
%                 for ii = [prevEpoch, epoch]
%                     impth = strsplit(images(jj).imds.Files{ii}, filesep); 
%                     imwrite(im{ii}, fullfile(epochDir, 'img', impth{end}));
%                     fprintf(f, '%s ', impth{end});
%                 end
%                 fclose(f);
%                 runSuperGlue_tracking(epochDir, prevEpochDir, jj)
%                 SGmatches = readSupeGlueTrackingRes(fullfile(epochDir, 'res'));  %readSupeGlueTrackingRes
%                
%                 pointsA = cat(2, SGmatches.kpts0{1}, SGmatches.kpts0{2});
%                 pointsB = cat(2, SGmatches.kpts1{1}, SGmatches.kpts1{2});            
%                 pointsAm = cat(2, SGmatches.mkpts0{1}, SGmatches.mkpts0{2});
%                 pointsBm = cat(2, SGmatches.mkpts1{1}, SGmatches.mkpts1{2});
%                 trackedPoints(jj).kpts0          = pointsA;
%                 trackedPoints(jj).kpts1          = pointsB;
%                 trackedPoints(jj).matches0       = SGmatches.matches0{1};
% %                 trackedPoints(jj).matches1       = SGmatches.matches1{1};            
%                 trackedPoints(jj).match_confidence = SGmatches.match_confidence{1}; 
%                 if printFigs == 1        
%                     figure('Name', sprintf('Tracked pointd on cam %i, from epoch t%i to epoch t%i', jj, prevEpoch-1, epoch-1));
%                     showMatchedFeatures(im{prevEpoch}, im{epoch}, pointsAm, pointsBm, 'montage','PlotOptions',{'ro','go','y:'});
%                 end
%             end
%     
%             mkpts0 = zeros(size(trackedPoints(1).kpts0));
%             mkpts1 = zeros(size(trackedPoints(2).kpts0));    
%             for i = 1:length(trackedPoints(1).kpts0) 
%                 res0 = trackedPoints(1).matches0(i);
%                 res1 = trackedPoints(2).matches0(i);
%                 if res0 > -1 && res1 > -1
% %                     mkpts0(i,:) = trackedPoints(1).kpts1(res0+1,:);
% %                     mkpts1(i,:) = trackedPoints(2).kpts1(res1+1,:);
%                     mkpts0(i,:) = trackedPoints(1).kpts1(i,:);
%                     mkpts1(i,:) = trackedPoints(2).kpts1(i,:);
%                 end
%             end
%             mkpts0 = mkpts0(mkpts0(:,1)>0,:);
%             mkpts1 = mkpts1(mkpts1(:,1)>0,:);      
%     
%             %- Restore original image points coordinate on full images (not cropped)
%             trackedFeatures(1).p{epoch}   = cat(1, trackedFeatures(1).p{epoch}, (mkpts0 +  maskBB(1).pos(1:2) - 1));
%             trackedFeatures(2).p{epoch}   = cat(1, trackedFeatures(2).p{epoch}, (mkpts1 +  maskBB(2).pos(1:2) - 1));
%             prevEpoch = prevEpoch - 1;
% 
%         end
% 
%         if printFigs == 1        
%             im = cell(1, numCams);
%             for jj=1:numCams 
%                 im{jj} = readimage(images(jj).imds, epoch);     
%             end
%                 figure('Name', sprintf('Tracked pointd from epoch t%i to epoch t%i', 0, epoch-1));
%                 showMatchedFeatures(im{1}, im{2}, trackedFeatures(1).p{epoch}, trackedFeatures(2).p{epoch}, 'montage','PlotOptions',{'ro','go','y:'});
%         end
%     end

    if epoch > 1 % Do both cameras together
        
        SGdir = 'SGmatching';
        im = cell(1, numCams);         

        trackedFeatures = struct();
        trackedFeatures(1).p{epoch} = [];
        trackedFeatures(2).p{epoch} = [];
        
%         epoch = 2
%         prevEpoch = 2
        prevEpoch = epoch-1;
        while (epoch - prevEpoch) < numEpoch2track+1 && prevEpoch > 0
            % prepare images and impairs.txt file
            epochDir = fullfile(SGdir, ['t', num2str(epoch-1)], ['from_t', num2str(prevEpoch-1)]);
            if ~exist(epochDir, "dir"); mkdir(epochDir); end            
            if ~exist(fullfile(epochDir, 'img'), "dir"); mkdir(fullfile(epochDir, 'img')); end       
            prevEpochDir = fullfile(SGdir, ['t', num2str(prevEpoch-1)], 'res');    
            f = fopen(fullfile(epochDir, 'impairs.txt'), 'w');                
            for jj = 1:numCams
                for ii = [prevEpoch, epoch]
                    im{ii} = readimage(images(jj).imds, ii);
                    if exist('maskBB', 'var')
                        im{ii} = imcrop(im{ii}, maskBB(jj).pos); 
                    end
                    impth = strsplit(images(jj).imds.Files{ii}, filesep); 
                    imwrite(im{ii}, fullfile(epochDir, 'img', impth{end}));
                    fprintf(f, '%s ', impth{end});
                end
                fprintf(f, '\n');

            end
            fclose(f);

            %- Run SG on the whole image       
            runSuperGlue_tracking(epochDir, prevEpochDir, jj)
            SGmatches = readSupeGlueRes(fullfile(epochDir, 'res'));  %readSupeGlueTrackingRes
           
            mkpts0 = cat(2, SGmatches.mkpts0{1}, SGmatches.mkpts0{2});
            mkpts1 = cat(2, SGmatches.mkpts1{1}, SGmatches.mkpts1{2});
            
            trackedFeatures(1).p{epoch} = cat(1, trackedFeatures(1).p{epoch}, mkpts0);
            trackedFeatures(2).p{epoch} = cat(1, trackedFeatures(2).p{epoch}, mkpts1); 
            
%             trackedFeatures(1).epoch{epoch}    = repmat(epoch, length(trackedFeatures(1).p{epoch}),1) ;

%             %- Restore original image points coordinate on full images (not cropped)
%             trackedFeatures(1).p{epoch} = cat(1, trackedFeatures(1).p{epoch}, mkpts0 +  maskBB(1).pos(1:2) - 1);
%             trackedFeatures(2).p{epoch} = cat(1, trackedFeatures(2).p{epoch}, mkpts1 +  maskBB(2).pos(1:2) - 1);

            prevEpoch = prevEpoch - 1;
        end

        if printFigs == 1        
            im = cell(1, numCams);
            for jj=1:numCams 
                im{jj} = readimage(images(jj).imds, epoch);     
            end
                figure('Name', sprintf('Tracked pointd from epoch t%i to epoch t%i', 0, epoch-1));
                showMatchedFeatures(im{1}, im{2}, trackedFeatures(1).p{epoch}, trackedFeatures(2).p{epoch}, 'montage','PlotOptions',{'ro','go','y:'});
        end
    end

    %------- Find matches at current epoch -----% 
    SGdir = 'SGmatching';
    epochDir = fullfile(SGdir, ['t', num2str(epoch-1)]);
    if ~exist(epochDir, "dir"); mkdir(epochDir); end
    im = cell(1, numCams);
    for jj=1:numCams 
        im{jj} = readimage(images(jj).imds, epoch);
        if exist('maskBB', 'var')
            im{jj} = imcrop(im{jj}, maskBB(jj).pos); 
        end
    end

    if useTiles == false
        %- Run SG on the whole image 
        if ~exist(fullfile(epochDir, 'img'), "dir"); mkdir(fullfile(epochDir, 'img')); end
        f = fopen(fullfile(epochDir, 'impairs.txt'), 'w');                
        for jj = 1:numCams 
            impth = strsplit(images(jj).imds.Files{epoch}, filesep); 
            imwrite(im{jj}, fullfile(epochDir, 'img', impth{end}));
            fprintf(f, '%s ', impth{end});
        end
        fclose(f);
        runSuperGlue(epochDir)
        SGmatches = readSupeGlueRes(fullfile(epochDir, 'res'));   
        pointsA = cat(2, SGmatches.mkpts0{1}, SGmatches.mkpts0{2});
        pointsB = cat(2, SGmatches.mkpts1{1}, SGmatches.mkpts1{2});
        if printFigs == 1        
            figure('Name', sprintf('Matches  at epoch t%i', epoch-1));
            showMatchedFeatures(im{1}, im{2}, pointsA, pointsB, 'montage','PlotOptions',{'ro','go','y:'});          
        end
    else          
        %-  Subdivide image in tiles and run SG on each tile   
%         nRowTiles   = 2;
%         nColTiles   = 3;
%         tiles       = {};                                               % Tiles are arranges column-wise!  
%         tileLimits  = {}; 
%         for jj = 1:numCams
%             [tiles(jj, :), tileLimits(jj, :)] = genImageTiles(im{jj}, rows=nRowTiles, cols=nColTiles, overlap=400); %  maxLength=1600
%             printImageTiles(tiles(jj,:))           
%         end
% 
%         pointsA = []; pointsB = [];
%         for i = 1:length(tiles)
%             fprintf('\tImage A: tile %i - Image B: tile %i...', i, i)
% 
%             tileDir = fullfile(epochDir, ['tile_', num2str(i)]);
%             if ~exist(tileDir, "dir"); mkdir(tileDir); end
%             if ~exist(fullfile(tileDir, 'img'), "dir"); mkdir(fullfile(tileDir, 'img')); end
% 
%             f = fopen(fullfile(tileDir, 'impairs.txt'), 'w');          
%             for jj = 1:numCams 
%                 impth = strsplit(images(jj).imds.Files{epoch}, filesep); 
%                 imwrite(tiles{jj,i}, fullfile(tileDir, 'img', impth{end}));  
%                 fprintf(f, '%s ', impth{end});
%             end
%             fclose(f);
%             runSuperGlue(tileDir)
% 
%             SGmatches = readSupeGlueRes(fullfile(tileDir, 'res'));
%             pointsTileA = cat(2, SGmatches.mkpts0{1}, SGmatches.mkpts0{2});
%             pointsTileB = cat(2, SGmatches.mkpts1{1}, SGmatches.mkpts1{2});
% 
%             pointsA = cat(1, pointsA, (pointsTileA + tileLimits{1,i}(1:2) -1 ) );
%             pointsB = cat(1, pointsB, (pointsTileB + tileLimits{1,i}(1:2) -1 ) );
%         end
%                     
%         figure('Name', sprintf('Matches at epoch t%i', epoch-1));
%         showMatchedFeatures(im{1}, im{2}, pointsA, pointsB, 'montage','PlotOptions',{'ro','go','y:'});       
    end

    %- Restore original image points coordinate on full images (not cropped)     
%     if epoch > 1 
%         features(1).p{epoch}   = ( cat(1, trackedFeatures(1).p{epoch}, (pointsA + maskBB(1).pos(1:2) - 1) ))';  
%         features(2).p{epoch}   = ( cat(1, trackedFeatures(2).p{epoch}, (pointsB + maskBB(2).pos(1:2) - 1) ))';  
%     else 
%         features(1).p{epoch}   = (pointsA + maskBB(1).pos(1:2) - 1)';  
%         features(2).p{epoch}   = (pointsB + maskBB(2).pos(1:2) - 1)';         
%     end
    if epoch > 1 
        features(1).p{epoch}   = ( cat(1, trackedFeatures(1).p{epoch}, pointsA ))';  
        features(2).p{epoch}   = ( cat(1, trackedFeatures(2).p{epoch}, pointsB ))';  
    else 
        features(1).p{epoch}   = pointsA';  
        features(2).p{epoch}   = pointsB';         
    end

    
    %- Store labels and compute undistorted points
    for jj=1:numCams 
%         features(jj).label{epoch}    = string(1:length(features(jj).p{epoch}));
%         features(jj).epoch{epoch}    = zeros(1,length(features(jj).p{epoch}));
        features(jj).pUnd{epoch}     = undistortPoints(features(jj).p{epoch}', cameras(jj).cameraParams)';       
    end
    
    %- Plot all features
    im = cell(1, numCams);    
    for jj=1:numCams 
        im{jj} = readimage(images(jj).imds, epoch);
    end   
    matchesEpochFig = figure('Name', sprintf('All features at epoch t%i', epoch-1), 'visible','off');
    showMatchedFeatures(im{1}, im{2}, features(1).p{epoch}', features(2).p{epoch}', 'montage','PlotOptions',{'ro','go','y:'});
    exportgraphics(matchesEpochFig, fullfile('matches',['matches_epoch_t', num2str(epoch-1), '.jpg']))

%     epoch = 2;
%     im = cell(1, numCams);    
%     for jj=1:numCams 
%         im{jj} = readimage(images(jj).imds, epoch);
%     end   
%     matchesEpochFig = figure('Name', sprintf('All features at epoch t%i', epoch-1));
%     showMatchedFeatures(im{1}, im{2}, features(1).p{epoch}(:,1:691)', features(2).p{epoch}(:,1:691)', 'montage','PlotOptions',{'ro','go','y:'});


    %----- 
    if epoch <= epoches2process(end)
        fprintf('Done. Moving to epoch %i.\n', epoch)
    else 
        fprintf('Processing complete successfully.\n')
    end


end

%%  Save workspace
fprintf('Saving workspace to file...')
save('Results')
fprintf('Done.\n')



%% Camera orientation and sparse reconstruction
for epoch = epoches2process+1
    %- Estimate camera 2 EO and triangulate sparse point cloud    
    if epoch > 1 % ipotesi che camera 1 sia fissa
        cameras(1).R{epoch} = cameras(1).R{1}; 
        cameras(1).t{epoch} = cameras(1).t{1}; 
        cameras(1).P{epoch} = cameras(1).P{1};
        cameras(1).X0{epoch} = cameras(1).X0{1};        
    end

%     if epoch > 1 % ipotesi che camera 1 sia fissa
%         cameras(1).R{epoch} = cameras(1).R{epoch-1}; 
%         cameras(1).t{epoch} = cameras(1).t{epoch-1}; 
%         cameras(1).P{epoch} = cameras(1).P{epoch-1};
%         cameras(1).X0{epoch} = cameras(1).X0{epoch-1};        
%     end

    % Fix also the second camera to the first epoch
    if epoch == 1 
        pts1 = features(1).pUnd{epoch};
        pts2 = features(2).pUnd{epoch};
        [cameras(2).R{epoch}, cameras(2).t{epoch}] = relative_lin(pts2, pts1, ...
                cameras(2).K, cameras(1).K);
        [cameras(2).R{epoch}, cameras(2).t{epoch}] = relative_nonlin(cameras(2).R{epoch}, cameras(2).t{epoch}, ...
            pts2, pts1, cameras(2).K, cameras(1).K);
        cameras(2).P{epoch}      = cameras(2).K * [cameras(2).R{epoch} cameras(2).t{epoch}];
        cameras(2).X0{epoch}     = - cameras(2).P{epoch}(:,1:3)\cameras(2).P{epoch}(:,4);
        
        %- Scale model by using camera baseline
        X01_meta = [416651.52489669225	5091109.91215075	1858.908434299682]';    % IMG_2092
        X02_meta = [416622.27552777925	5091364.507128085	1902.4053286545502]';   % IMG_0481
        camWorldBaseline    = norm(X01_meta - X02_meta);                            % [m] From Metashape model at epoch t0
        camRelOriBaseline   = norm(cameras(1).X0{epoch} - cameras(2).X0{epoch});
        scaleFct        = camWorldBaseline ./ camRelOriBaseline;
        cameras(2).X0{epoch} = cameras(2).X0{epoch} * scaleFct;
        cameras(2).t{epoch} = - cameras(2).R{epoch} * cameras(2).X0{epoch};
        cameras(2).P{epoch} = cameras(2).K * [cameras(2).R{epoch} cameras(2).t{epoch}];        
    else   % ipotesi che anche camera 2 sia fissa
        cameras(2).R{epoch} = cameras(2).R{1}; 
        cameras(2).t{epoch} = cameras(2).t{1}; 
        cameras(2).P{epoch} = cameras(2).P{1};
        cameras(2).X0{epoch} = cameras(2).X0{1};        
    end
    
    sparsePts(epoch).pointsXYZ  = triang_lin_batch( {cameras(1).P{epoch}, cameras(2).P{epoch}}, {features(1).pUnd{epoch}, features(2).pUnd{epoch}} );
    for i = 1:size(sparsePts(epoch).pointsXYZ,2)
        sparsePts(epoch).pointsXYZ(:, i) = triang_nonlin(sparsePts(epoch).pointsXYZ(:,i), ...
             {cameras(1).P{epoch}, cameras(2).P{epoch}}, {features(1).pUnd{epoch}(:,i), features(2).pUnd{epoch}(:,i)});
    end
    
    %-  Create and plot Sparse Point Cloud with PointCloudTools
    im{2} = readimage(images(2).imds, epoch); 
    col = interpPointCol(sparsePts(epoch).pointsXYZ, im{2}, cameras(2).P{epoch}, cameras(2).cameraParams);
    sparsePts(epoch).ptCloud    = pointCloud(sparsePts(epoch).pointsXYZ', "Color", col');
    pcwrite(sparsePts(epoch).ptCloud, ['ptcloud/sparsepts_t', num2str(epoch-1), '.ply'])
    if printFigs == 1        
        fig = figure('Name', sprintf('Point cloud at epoch t%i', epoch-1)); hold on;
        pcshow(sparsePts(epoch).ptCloud, 'VerticalAxis','Y', 'VerticalAxisDir','Down', 'MarkerSize',45)
        xlabel('x'), ylabel('y'), zlabel('z')
        plotCamera('Location', cameras(1).X0{epoch}, 'Orientation', cameras(1).R{epoch}, 'Size', 10, ...
            'Color', [0.9294 0.6941 0.1255], 'Opacity', 0);
        plotCamera('Location', cameras(2).X0{epoch}, 'Orientation', cameras(2).R{epoch}, 'Size', 10, ...
            'Color', [0.3020 0.7451 0.9333], 'Opacity', 0);
        line([0 30], [0 0], [0 0], 'Color', 'r', 'LineWidth',3)
        line([0 0], [0 30], [0 0], 'Color', 'g', 'LineWidth',3)
        line([0 0], [0 0], [0 40], 'Color', 'b', 'LineWidth',3)
%         exportgraphics(fig, 'camerasEO.png', 'Resolution',300, 'BackgroundColor', [0,0,0])
    end

end

rms_cam_mov = zeros(epoch,1);
diff_cam_rot = zeros(epoch,3);
fprintf('RMSE camera 2 movement:\n')
for i = epoches2process+1
    rms_cam_mov(i) = rms(cameras(2).X0{i} - cameras(2).X0{1});
    diff_cam_rot(i,:) =  rad2deg(ieul(cameras(2).R{i})) -  rad2deg(ieul(cameras(2).R{1}));
    fprintf('Epoch %i: \t%.3f \t\t%+.4f \t\t%+.4f \t\t%+.4f  \n', ...
        i-1, rms_cam_mov(i), diff_cam_rot(i,1), diff_cam_rot(i,2), diff_cam_rot(i,3))
end


% for epoch = epoches2process+1
%     sparsePts(epoch).ptCloud    = pointCloud(sparsePts(epoch).pointsXYZ'); %, "Color", ptCloudCols');
%     pcwrite(sparsePts(epoch).ptCloud, ['ptcloud/sparsepts_t', num2str(epoch-1), '.ply'])
% end

%% 
% % ipotesi che anche camera 2 sia fissa  
% for epoch = epoches2process+1  
%     cameras(2).R{epoch} = cameras(2).R{1}; 
%     cameras(2).t{epoch} = cameras(2).t{1}; 
%     cameras(2).P{epoch} = cameras(2).P{1};
%     cameras(2).X0{epoch} = cameras(2).X0{1};        
%         
%     sparsePts(epoch).pointsXYZ  = triang_lin_batch( {cameras(1).P{epoch}, cameras(2).P{epoch}}, {features(1).pUnd{epoch}, features(2).pUnd{epoch}} );
%     sparsePts(epoch).labels     = features(1).label{epoch};
%     for i = 1:size(sparsePts(epoch).pointsXYZ,2)
%         sparsePts(epoch).pointsXYZ(:, i) = triang_nonlin(sparsePts(epoch).pointsXYZ(:,i), ...
%              {cameras(1).P{epoch}, cameras(2).P{epoch}}, {features(1).pUnd{epoch}(:,i), features(2).pUnd{epoch}(:,i)});
%     end
% 
%     sparsePts(epoch).ptCloud    = pointCloud(sparsePts(epoch).pointsXYZ'); 
% end
% 
i = 1; j = epoch;
sparsePts(i).ptCloud.Color = uint8(repmat([255 0 0],sparsePts(i).ptCloud.Count,1));
sparsePts(j).ptCloud.Color = uint8(repmat([0 255 0],sparsePts(j).ptCloud.Count,1));
figure('Name', sprintf('Point cloud at epoch t%i', epoch-1)); hold on;
pcshow(sparsePts(i).ptCloud, 'VerticalAxis','Y', 'VerticalAxisDir','Down', 'MarkerSize',80)
pcshow(sparsePts(j).ptCloud, 'VerticalAxis','Y', 'VerticalAxisDir','Down', 'MarkerSize',80)
xlabel('x'), ylabel('y'), zlabel('z')   
% 
% i = 5; j = 20;
pcwrite(sparsePts(i).ptCloud, ['ptcloud/sparsepts_t', num2str(i-1), '.ply'])
pcwrite(sparsePts(j).ptCloud, ['ptcloud/sparsepts_t', num2str(j-1), '.ply'])


%%  Save workspace
fprintf('Saving workspace to file...')
save('Results')
fprintf('Done.\n')


%% Write results to file

for epoch = epoches2process+1
    for jj = 1:2
        filename = fullfile('outdir', ['imm_cam', num2str(jj), '_t', num2str(epoch-1), '.txt']);
        fid = fopen(filename, 'w');    
        for i=1:length(features(1).p{epoch})
            fprintf(fid, '%i %5.0f %5.0f\n', ...
                   10000+i, features(jj).p{epoch}(1,i), features(jj).p{epoch}(2,i));
        end
        fclose(fid);
    end

    for jj = 1:2
        filename = fullfile('outdir', ['imm_cam', num2str(jj), '_t', num2str(epoch-1), '_undistorted.txt']);
        fid = fopen(filename, 'w');    
        for i=1:length(features(1).p{epoch})
            fprintf(fid, '%i %5.0f %5.0f\n', ...
                   10000+i, features(jj).pUnd{epoch}(1,i), features(jj).pUnd{epoch}(2,i));
        end
        fclose(fid);
    end

    for jj = 1:2
        filename = fullfile('outdir', ['app_world', '_t', num2str(epoch-1), '.txt']);
        fid = fopen(filename, 'w');    
        for i=1:length(features(1).p{epoch})
            fprintf(fid, '%i %09.4f %09.4f %09.4f\n', ...
                   10000+i, sparsePts(epoch).pointsXYZ(1,i), ...
                   sparsePts(epoch).pointsXYZ(2,i),  sparsePts(epoch).pointsXYZ(3,i));
        end
        fclose(fid);
    end

    filename = fullfile('outdir', ['camOri', '_t', num2str(epoch-1), '.txt']);
    fid = fopen(filename, 'w');    
    fprintf(fid, 'camId X0[m] Y0[m] Z0[m] omega[deg] phi[deg] kappa[deg]\n');
    for jj = 1:2
    fprintf(fid, '%i %.4f %.4f %.4f %.6f %.6f %.6f \n', ...
               jj, cameras(jj).X0{epoch},...
               rad2deg(ieul(cameras(jj).R{epoch})) );
    end
    fclose(fid);



    rms_cam_mov(i) = rms(cameras(2).X0{i} - cameras(2).X0{1});
    diff_cam_rot(i,:) =  rad2deg(ieul(cameras(2).R{i})) -  rad2deg(ieul(cameras(2).R{1}));
    fprintf('Epoch %i: \t%.3f \t\t%+.4f \t\t%+.4f \t\t%+.4f  \n', ...
        i-1, rms_cam_mov(i), diff_cam_rot(i,1), diff_cam_rot(i,2), diff_cam_rot(i,3))
end


%%
% 
% % Test pydegensac
% epoch = 1;
% % mkpts0 = features(2).p{epoch}';
% % mkpts1 = features(1).p{epoch}';
% mkpts0 = (pointsA);
% mkpts1 = (pointsB);
% px_tr = 3;
% [F, mask] = runPyDegensac(mkpts0, mkpts1, px_tr, 'matches');
% % im = cell(1, numCams);    
% % for jj=1:numCams 
% %     im{jj} = readimage(images(jj).imds, epoch);
% % end   
% % figure('Name', sprintf('Matches at epoch t%i', epoch-1));
% % showMatchedFeatures(im{1}, im{2}, mkpts0, mkpts1, 'montage','PlotOptions',{'ro','go','y:'});          
% 
% 
% epoch = 1;
% mkpts0 = features(2).p{epoch}';
% mkpts1 = features(1).p{epoch}';
% [F, inliersIdx] = estimateFundamentalMatrix(mkpts0, mkpts1,'Method','MSAC', 'DistanceThreshold', 2, 'NumTrials', 1e5);
% fprintf('Inliers: %i (%.0f%%)\n',sum(inliersIdx),sum(inliersIdx)/length(inliersIdx)*100)
% for jj=1:numCams 
%     im{jj} = readimage(images(jj).imds, epoch);
% end  
% figure('Name', sprintf('Matches at epoch t%i', epoch-1));
% showMatchedFeatures(im{1}, im{2}, mkpts0(inliersIdx,:), mkpts1(inliersIdx,:), 'montage','PlotOptions',{'ro','go','y:'});          
% 
% % 
% epoch = 1;
% np = py.importlib.import_module('numpy');
% pydegensac = py.importlib.import_module('pydegensac');
% mkpts0 = py.numpy.array(features(1).p{epoch}');
% mkpts1 = py.numpy.array(features(2).p{epoch}');
% % mkpts0 = py.numpy.array(pointsA);
% % mkpts1 = py.numpy.array(pointsB);
% out = pydegensac.findFundamentalMatrix(mkpts0, mkpts1, 2);
% F = out(1);
% mask = out(2);
% F = double(np.float32(F));
% inliers = double(np.float32(mask));
% fprintf('Inliers: %i (%.0f%%)\n',sum(inliers),sum(inliers)/length(inliers)*100)


% 
% % mconf = mconf[inlMask]
% % mkpts0 = mkpts0[inlMask]
% % mkpts1 = mkpts1[inlMask]
% % scores0 = scores0[inlMask]
% % scores1 = scores1[inlMask]
% % descriptors0 = descriptors0[:,inlMask]
% % descriptors1 = descriptors1[:,inlMask]

