%This function estimtates the centers of nPnts points that lie along a line
%defined by the vector vectDirIn. The first point is pointIn
%thres is  threshold that binarize the input image.

function centPnts = estCentHalfLine(pointIn, vectDirIn, nPnts, inpIMg, thres)
    % Initialize visualization flag and set to 0 for no visualization.
    figOn = 0;

    % Initialize the array to hold the center points and set the first point.
    centPnts = zeros(nPnts, 2);
    centPnts(1, :) = pointIn;
    pointIni = centPnts(1, :);  % Initial point [row, col], not x, y.
    vectDir = vectDirIn;  % Direction vector [row, col], not x, y.
    rN = 9;  % Radius of neighborhood in pixels.
    scale = 5;  % Scaling factor for resizing images.

    % Normalize input image to have max value of 1.
    inpIMg = inpIMg / max(inpIMg(:));

    if figOn == 1
        % Visualization code (only executed if figOn is set to 1).
        x = 1:size(inpIMg, 1);
        [mX, mY] = meshgrid(x, x);
        maskCirc = double(((mX - pointIni(2)).^2 + (mY - pointIni(1)).^2) < rN^2);
        testImag = max(inpIMg, maskCirc);
        testImag = double(imbinarize(testImag, thres));
        figure(59); imagesc(testImag); colormap('hot');
    else
        % Binarize input image based on threshold.
        testImag = double(imbinarize(inpIMg, thres));
    end

    % Loop through each point to estimate center points along the line.
    for i = 2:nPnts
        period = norm(vectDir);  % Calculate the distance (period) between points.
        midPoint = pointIni + vectDir;  % Calculate midpoint for current segment.

        % Calculate indices for cropping, ensuring they stay within image bounds.
        iniIndF = max(fix(midPoint(1)) - fix(period), 1);
        finIndF = min(fix(midPoint(1)) + fix(period), size(inpIMg, 1));
        if mod(finIndF - iniIndF, 2) == 1
            finIndF = finIndF - 1;
            error('edge');
        end

        iniIndC = max(fix(midPoint(2)) - fix(period), 1);
        finIndC = min(fix(midPoint(2)) + fix(period), size(inpIMg, 2));
        if mod(finIndC - iniIndC, 2) == 1
            finIndC = finIndC - 1;
            error('edge');
        end
        % Crop and resize the section of interest.
        croppSect = testImag(iniIndF:finIndF, iniIndC:finIndC);
        croppSect = croppSect / max(croppSect(:));
        initSizCropS = size(croppSect);
        croppSect = imresize(croppSect, scale, 'box');
        sizeCrop = size(croppSect);

        % Generate grid for circular mask application.
        x = linspace(1, initSizCropS(2), sizeCrop(2));
        y = linspace(1, initSizCropS(1), sizeCrop(1));
        dx = x(2) - x(1);
        dy = y(2) - y(1);
        [mX, mY] = meshgrid(x, y);
        maskCirc = double(((mX - (initSizCropS(2) + 1) / 2).^2 + (mY - (initSizCropS(1) + 1) / 2).^2) < rN^2);

        % Convolve mask with cropped section and find the point of highest intensity.
        maxC = conv2(maskCirc, croppSect, 'same');
        [sortMaxC, indM1] = sort(maxC(:), 'descend');
        indM1 = indM1(sortMaxC == sortMaxC(1));
        [maxIndF, maxIndC] = ind2sub(sizeCrop, indM1);
        maxIndF = mean(maxIndF);
        maxIndC = mean(maxIndC);

        % Update center point and direction for next iteration.
        centPnts(i, :) = fix(midPoint) + ([0, maxIndC] - [0, (sizeCrop(2) + 1) / 2]) * dx + ([maxIndF, 0] - [(sizeCrop(1) + 1) / 2, 0]) * dy;
        pointIni = centPnts(i, :);
        vectDir = centPnts(i, :) - centPnts(i - 1, :);
    end

    % If visualization is enabled, plot the estimated center points.
    if figOn == 1
        hold on; plot(centPnts(:, 2), centPnts(:, 1), 'or');
    end
end

