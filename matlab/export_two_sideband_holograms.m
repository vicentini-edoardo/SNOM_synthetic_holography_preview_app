function summary = export_two_sideband_holograms(folderPath)
%EXPORT_TWO_SIDEBAND_HOLOGRAMS Export SNOM holograms as per-harmonic MAT files.
%   SUMMARY = EXPORT_TWO_SIDEBAND_HOLOGRAMS(FOLDERPATH) loads the forward (O)
%   and reverse (R-O) harmonic stacks found in FOLDERPATH, applies the same
%   two-sideband processing workflow used by the Python preview app, and writes
%   one self-contained MAT file per detected harmonic into:
%       <folderPath>/matlab-two-sideband-export

    arguments
        folderPath (1, :) char
    end

    folderPath = char(string(folderPath));
    if ~isfolder(folderPath)
        error('export_two_sideband_holograms:FolderNotFound', ...
            'Folder does not exist: %s', folderPath);
    end

    folderPath = char(java.io.File(folderPath).getCanonicalPath());
    [~, imageName] = fileparts(folderPath);
    outputFolder = fullfile(folderPath, 'matlab-two-sideband-export');
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    padFact = 1;
    alphaValue = 0.3;
    processingMode = 'two_sideband';
    referenceHarmonic = 2;
    passageConfigs = {
        'forward', 'O';
        'reverse', 'R-O';
    };

    summary = repmat(empty_summary(), 0, 1);

    for idx = 1:size(passageConfigs, 1)
        passageName = passageConfigs{idx, 1};
        modeName = passageConfigs{idx, 2};
        passageData = load_passage_stack(folderPath, imageName, modeName);
        if isempty(passageData.available_harmonics)
            continue;
        end

        if ~any(passageData.available_harmonics == referenceHarmonic)
            error('export_two_sideband_holograms:MissingReferenceHarmonic', ...
                ['Passage "%s" is missing harmonic %d. ', ...
                 'Two-sideband parameter detection requires harmonic %d.'], ...
                passageName, referenceHarmonic, referenceHarmonic);
        end

        processed = process_stack(passageData.raw_stack, ...
            passageData.available_harmonics, padFact, alphaValue, referenceHarmonic);

        filesWritten = {};
        for harmonicIndex = reshape(passageData.available_harmonics, 1, [])
            rawHologram = processed.raw_stack(:, :, harmonicIndex + 1);
            processedHologram = processed.processed_stack(:, :, harmonicIndex + 1);
            rawAmplitude = abs(rawHologram);
            rawPhase = angle(rawHologram);
            processedAmplitude = abs(processedHologram);
            processedPhase = processed_phase(processedHologram);
            fileName = sprintf('%s_h%d_two_sideband.mat', passageName, harmonicIndex);
            outputPath = fullfile(outputFolder, fileName);

            passage = passageName; %#ok<NASGU>
            harmonic_index = harmonicIndex; %#ok<NASGU>
            processing_mode = processingMode; %#ok<NASGU>
            image_name = imageName; %#ok<NASGU>
            folder_path = folderPath; %#ok<NASGU>
            raw_hologram = rawHologram; %#ok<NASGU>
            processed_hologram = processedHologram; %#ok<NASGU>
            raw_amplitude = rawAmplitude; %#ok<NASGU>
            raw_phase = rawPhase; %#ok<NASGU>
            processed_amplitude = processedAmplitude; %#ok<NASGU>
            processed_phase = processedPhase; %#ok<NASGU>
            carrier_row = processed.carrier_row; %#ok<NASGU>
            filter_width_y = processed.filter_width_y; %#ok<NASGU>
            fft_center_row = processed.fft_center_row; %#ok<NASGU>
            mirror_row = processed.mirror_row_by_harmonic(harmonicIndex + 1); %#ok<NASGU>
            rotation_angle_rad = processed.rotation_angle_rad_by_harmonic(harmonicIndex + 1); %#ok<NASGU>
            rotation_angle_deg = processed.rotation_angle_deg_by_harmonic(harmonicIndex + 1); %#ok<NASGU>
            pad_fact = padFact; %#ok<NASGU>
            alpha = alphaValue; %#ok<NASGU>

            save(outputPath, ...
                'image_name', 'folder_path', 'passage', 'harmonic_index', ...
                'processing_mode', 'raw_hologram', 'processed_hologram', ...
                'raw_amplitude', 'raw_phase', 'processed_amplitude', ...
                'processed_phase', 'carrier_row', 'filter_width_y', ...
                'fft_center_row', 'mirror_row', 'rotation_angle_rad', ...
                'rotation_angle_deg', 'pad_fact', 'alpha');
            filesWritten{end + 1, 1} = outputPath; %#ok<AGROW>
        end

        passageSummary = empty_summary();
        passageSummary.passage = passageName;
        passageSummary.mode_name = modeName;
        passageSummary.image_name = imageName;
        passageSummary.folder_path = folderPath;
        passageSummary.output_folder = outputFolder;
        passageSummary.processing_mode = processingMode;
        passageSummary.reference_harmonic = referenceHarmonic;
        passageSummary.detected_harmonics = passageData.available_harmonics;
        passageSummary.skipped_harmonics = setdiff(0:5, passageData.available_harmonics);
        passageSummary.files_written = filesWritten;
        passageSummary.carrier_row = processed.carrier_row;
        passageSummary.filter_width_y = processed.filter_width_y;
        passageSummary.fft_center_row = processed.fft_center_row;
        passageSummary.pad_fact = padFact;
        passageSummary.alpha = alphaValue;
        summary(end + 1, 1) = passageSummary; %#ok<AGROW>
    end
end


function summary = empty_summary()
    summary = struct( ...
        'passage', '', ...
        'mode_name', '', ...
        'image_name', '', ...
        'folder_path', '', ...
        'output_folder', '', ...
        'processing_mode', 'two_sideband', ...
        'reference_harmonic', 2, ...
        'detected_harmonics', [], ...
        'skipped_harmonics', [], ...
        'files_written', {{}}, ...
        'carrier_row', NaN, ...
        'filter_width_y', NaN, ...
        'fft_center_row', NaN, ...
        'pad_fact', 1, ...
        'alpha', 0.3);
end


function passageData = load_passage_stack(folderPath, imageName, modeName)
    rawStack = [];
    availableHarmonics = [];
    firstShape = [];

    for harmonicIndex = 0:5
        [ampPath, phasePath] = mode_file_paths(folderPath, imageName, modeName, harmonicIndex);
        ampExists = isfile(ampPath);
        phaseExists = isfile(phasePath);
        if xor(ampExists, phaseExists)
            error('export_two_sideband_holograms:IncompleteHarmonicPair', ...
                ['Passage "%s" harmonic %d is incomplete. ', ...
                 'Expected both amplitude and phase GSF files.'], ...
                modeName, harmonicIndex);
        end
        if ~ampExists
            continue;
        end

        amplitude = read_gsf(ampPath);
        phase = read_gsf(phasePath);
        complexImage = amplitude .* exp(1i .* phase);

        if isempty(firstShape)
            firstShape = size(complexImage);
            rawStack = complex(nan(firstShape(1), firstShape(2), 6), ...
                               nan(firstShape(1), firstShape(2), 6));
        elseif ~isequal(size(complexImage), firstShape)
            error('export_two_sideband_holograms:ShapeMismatch', ...
                'Passage "%s" harmonic %d has shape [%d %d], expected [%d %d].', ...
                modeName, harmonicIndex, size(complexImage, 1), size(complexImage, 2), ...
                firstShape(1), firstShape(2));
        end

        rawStack(:, :, harmonicIndex + 1) = complexImage;
        availableHarmonics(end + 1) = harmonicIndex; %#ok<AGROW>
    end

    passageData = struct( ...
        'raw_stack', rawStack, ...
        'available_harmonics', availableHarmonics);
end


function [ampPath, phasePath] = mode_file_paths(folderPath, imageName, modeName, harmonicIndex)
    ampPath = fullfile(folderPath, sprintf('%s %s%dA raw.gsf', imageName, modeName, harmonicIndex));
    phasePath = fullfile(folderPath, sprintf('%s %s%dP raw.gsf', imageName, modeName, harmonicIndex));
end


function data = read_gsf(filePath)
    fileId = fopen(filePath, 'rb');
    if fileId == -1
        error('export_two_sideband_holograms:ReadError', 'Unable to open GSF file: %s', filePath);
    end
    cleanupObj = onCleanup(@() fclose(fileId)); %#ok<NASGU>

    bytes = fread(fileId, Inf, '*uint8');
    content = native2unicode(bytes.', 'latin1');
    scalingPos = strfind(content, 'erScaling=1');
    xResPos = strfind(content, 'XRes=');
    yResPos = strfind(content, 'YRes=');
    yResIncompletePos = strfind(content, 'YResIncomplete=');

    if isempty(scalingPos) || isempty(xResPos) || isempty(yResPos) || isempty(yResIncompletePos)
        error('export_two_sideband_holograms:ReadError', ...
            'GSF header is missing expected fields: %s', filePath);
    end

    payloadOffset = scalingPos(1) + 10;
    payloadOffset = ceil(double(payloadOffset) / 4) * 4;

    resXText = strtrim(content((xResPos(1) + 5):(yResPos(1) - 1)));
    resYText = strtrim(content((yResPos(1) + 5):(yResIncompletePos(1) - 1)));
    resX = str2double(resXText);
    resY = str2double(resYText);
    if isnan(resX) || isnan(resY)
        error('export_two_sideband_holograms:ReadError', ...
            'GSF header contains invalid dimensions in: %s', filePath);
    end

    fseek(fileId, payloadOffset, 'bof');
    data = fread(fileId, resX * resY, '*single');
    if numel(data) ~= resX * resY
        error('export_two_sideband_holograms:ReadError', ...
            'GSF payload size does not match dimensions in: %s', filePath);
    end
    data = reshape(data, [resX, resY]).';
end


function processed = process_stack(rawStack, availableHarmonics, padFact, alpha, referenceHarmonic)
    referenceImage = rawStack(:, :, referenceHarmonic + 1);
    [referenceProcessed, carrierRow, filterWidthY, diagnostics] = ...
        open_hologram_2d(referenceImage, padFact, alpha, [], []);

    processedStack = complex(nan(size(rawStack)), nan(size(rawStack)));
    processedStack(:, :, referenceHarmonic + 1) = referenceProcessed;

    rotationAngleRadByHarmonic = nan(1, size(rawStack, 3));
    rotationAngleDegByHarmonic = nan(1, size(rawStack, 3));
    mirrorRowByHarmonic = nan(1, size(rawStack, 3));
    rotationAngleRadByHarmonic(referenceHarmonic + 1) = diagnostics.rotation_angle_rad;
    rotationAngleDegByHarmonic(referenceHarmonic + 1) = diagnostics.rotation_angle_deg;
    mirrorRowByHarmonic(referenceHarmonic + 1) = diagnostics.mirror_row;

    for harmonicIndex = reshape(availableHarmonics, 1, [])
        if harmonicIndex == referenceHarmonic
            continue;
        end
        [processedImage, ~, ~, perHarmonicDiagnostics] = ...
            open_hologram_2d(rawStack(:, :, harmonicIndex + 1), ...
                padFact, alpha, carrierRow, filterWidthY);
        processedStack(:, :, harmonicIndex + 1) = processedImage;
        rotationAngleRadByHarmonic(harmonicIndex + 1) = perHarmonicDiagnostics.rotation_angle_rad;
        rotationAngleDegByHarmonic(harmonicIndex + 1) = perHarmonicDiagnostics.rotation_angle_deg;
        mirrorRowByHarmonic(harmonicIndex + 1) = perHarmonicDiagnostics.mirror_row;
    end

    processed = struct( ...
        'raw_stack', rawStack, ...
        'processed_stack', processedStack, ...
        'carrier_row', carrierRow, ...
        'filter_width_y', filterWidthY, ...
        'fft_center_row', floor((size(rawStack, 1) * padFact) / 2), ...
        'rotation_angle_rad_by_harmonic', rotationAngleRadByHarmonic, ...
        'rotation_angle_deg_by_harmonic', rotationAngleDegByHarmonic, ...
        'mirror_row_by_harmonic', mirrorRowByHarmonic);
end


function [imageComplexOut, carrierRow, filterWidthY, diagnostics] = ...
    open_hologram_2d(imageComplexIn2D, padFact, alpha, carrierRow, filterWidthY)

    [rotatedComplex, rotationAngle] = rotate_to_real_axis(imageComplexIn2D);
    signalIn = real(rotatedComplex);

    nY = size(signalIn, 1);
    nX = size(signalIn, 2);
    nY2 = nY * padFact;
    nX2 = nX;

    signalPad = zeros(nY2, nX2);
    startY = floor((nY2 - nY) / 2) + 1;
    startX = floor((nX2 - nX) / 2) + 1;
    signalPad(startY:(startY + nY - 1), startX:(startX + nX - 1)) = signalIn;

    signalFt = fftshift(fft2(signalPad));
    magSignalFt = abs(signalFt);
    verticalProfile = build_vertical_profile(magSignalFt);

    if isempty(carrierRow)
        carrierRow = round(find_vertical_carrier(verticalProfile));
    end
    if isempty(filterWidthY) || filterWidthY <= 0
        filterWidthY = estimate_filter_width(verticalProfile, carrierRow);
    end
    [carrierRow, filterWidthY] = validate_vertical_filter(nY2, carrierRow, filterWidthY);

    mirrorRow = nY2 - carrierRow;
    if mirrorRow >= nY2
        mirrorRow = nY2 - 1;
    end

    sidebandFilterPos = build_sideband_filter(nY2, carrierRow, filterWidthY, alpha);
    sidebandFilterNeg = build_sideband_filter(nY2, mirrorRow, filterWidthY, alpha);
    filteredPos = bsxfun(@times, signalFt, sidebandFilterPos);
    filteredNeg = bsxfun(@times, signalFt, sidebandFilterNeg);
    shiftedPos = shift_band_to_center(filteredPos, carrierRow);
    shiftedNeg = shift_band_to_center(filteredNeg, mirrorRow);

    fieldPosFull = ifft2(ifftshift(shiftedPos));
    fieldNegFull = ifft2(ifftshift(shiftedNeg));
    filteredFullRotated = 0.5 .* (conj(fieldPosFull) + fieldNegFull);
    filteredFull = filteredFullRotated .* exp(1i .* rotationAngle);

    imageComplexOut = filteredFull(startY:(startY + nY - 1), startX:(startX + nX - 1));
    diagnostics = struct( ...
        'rotation_angle_rad', rotationAngle, ...
        'rotation_angle_deg', rad2deg(rotationAngle), ...
        'mirror_row', mirrorRow);
end


function [rotated, angleValue] = rotate_to_real_axis(imageComplexIn2D)
    angleValue = data_angle(real(imageComplexIn2D), imag(imageComplexIn2D));
    rotated = imageComplexIn2D .* exp(-1i .* angleValue);
end


function angleValue = data_angle(xValues, yValues)
    xx = xValues(:);
    yy = yValues(:);
    validMask = isfinite(xx) & isfinite(yy);
    xx = xx(validMask);
    yy = yy(validMask);
    if numel(xx) < 2
        error('export_two_sideband_holograms:AngleEstimationFailed', ...
            'Need at least two finite samples to estimate the complex-plane angle.');
    end
    if min(xx) == max(xx)
        angleValue = pi / 2;
        return;
    end
    if min(yy) == max(yy)
        angleValue = 0;
        return;
    end

    coeffHor = polyfit(xx, yy, 1);
    horFit = polyval(coeffHor, xx);
    stderrHor = fit_stderr(yy, horFit);

    coeffVer = polyfit(yy, xx, 1);
    verFit = polyval(coeffVer, yy);
    stderrVer = fit_stderr(xx, verFit);

    if stderrHor < stderrVer
        angleValue = atan(coeffHor(1));
    else
        angleValue = atan(1 ./ coeffVer(1));
    end
end


function stderrValue = fit_stderr(observed, fitted)
    residual = observed - fitted;
    dof = max(numel(observed) - 2, 1);
    stderrValue = sqrt(sum(abs(residual) .^ 2) ./ dof);
end


function verticalProfile = build_vertical_profile(magnitudeFt)
    profile = log1p(mean(magnitudeFt, 2));
    kernelSize = max(5, min(21, floor(numel(profile) / 32)));
    if mod(kernelSize, 2) == 0
        kernelSize = kernelSize + 1;
    end
    kernel = ones(kernelSize, 1) ./ kernelSize;
    verticalProfile = conv(profile, kernel, 'same');
end


function peakIdx = find_vertical_carrier(profile)
    centerIdx = floor(numel(profile) / 2);
    exclusionHalfWidth = max(8, floor(numel(profile) / 32));
    edgeMargin = 2;
    searchStop = centerIdx - exclusionHalfWidth;
    if searchStop <= edgeMargin + 1
        error('export_two_sideband_holograms:CarrierDetectionFailed', ...
            'Unable to search for a carrier away from the zero order.');
    end

    segmentStart = edgeMargin + 1;
    segmentStop = searchStop - edgeMargin;
    searchSegment = profile(segmentStart:segmentStop);
    minPeakDistance = max(4, floor(numel(profile) / 64));
    minPeakProminence = max(std(profile) * 0.25, 1e-6);

    [peakValues, peakLocs, ~, peakProminences] = findpeaks( ...
        searchSegment, ...
        'MinPeakDistance', minPeakDistance, ...
        'MinPeakProminence', minPeakProminence);

    if ~isempty(peakLocs)
        candidateRows = peakLocs + segmentStart - 1;
        [~, bestIdx] = max(peakProminences .* peakValues);
        bestPeak = candidateRows(bestIdx);
        peakIdx = refine_peak_subpixel(profile, bestPeak - 1);
        return;
    end

    searchProfile = profile;
    searchProfile(1:edgeMargin) = -inf;
    searchProfile(searchStop:end) = -inf;
    [peakValue, idx] = max(searchProfile);
    if ~isfinite(peakValue)
        error('export_two_sideband_holograms:CarrierDetectionFailed', ...
            'Unable to find a valid vertical carrier away from the zero order.');
    end
    peakIdx = refine_peak_subpixel(profile, idx - 1);
end


function refinedIndex = refine_peak_subpixel(profile, peakIdxZeroBased)
    peakIndexOneBased = peakIdxZeroBased + 1;
    if peakIndexOneBased <= 1 || peakIndexOneBased >= numel(profile)
        refinedIndex = double(peakIdxZeroBased);
        return;
    end

    left = profile(peakIndexOneBased - 1);
    center = profile(peakIndexOneBased);
    right = profile(peakIndexOneBased + 1);
    denominator = left - 2 .* center + right;
    if abs(denominator) < 1e-12
        refinedIndex = double(peakIdxZeroBased);
        return;
    end

    offset = 0.5 .* (left - right) ./ denominator;
    offset = min(max(offset, -0.5), 0.5);
    refinedIndex = double(peakIdxZeroBased) + offset;
end


function width = estimate_filter_width(profile, carrierRow)
    centerIdx = floor(numel(profile) / 2);
    distanceToZeroOrder = abs(carrierRow - centerIdx);
    distanceToEdge = min(carrierRow, numel(profile) - carrierRow - 1);
    width = floor(min(distanceToZeroOrder, distanceToEdge));
    if width < 2
        error('export_two_sideband_holograms:FilterWidthFailed', ...
            'Detected carrier is too close to the zero order to build a stable filter.');
    end
end


function [carrierRow, width] = validate_vertical_filter(lengthY, carrierRow, filterWidthY)
    centerIdx = floor(lengthY / 2);
    exclusionHalfWidth = max(8, floor(lengthY / 32));
    if carrierRow >= centerIdx - exclusionHalfWidth
        error('export_two_sideband_holograms:InvalidCarrierRow', ...
            'Carrier center must stay above the zero-order exclusion band.');
    end
    if carrierRow < 1 || carrierRow >= lengthY - 1
        error('export_two_sideband_holograms:InvalidCarrierRow', ...
            'Carrier center must stay inside the Fourier-space image bounds.');
    end

    width = normalize_filter_width(filterWidthY, lengthY);
    [~, stopIdx] = band_bounds(carrierRow, width, lengthY);
    if stopIdx >= centerIdx - exclusionHalfWidth + 1
        error('export_two_sideband_holograms:InvalidFilterWidth', ...
            'Filter width reaches the zero order; reduce the width or move the center upward.');
    end
end


function width = normalize_filter_width(width, lengthY)
    width = max(2, min(round(width), lengthY));
    if mod(width, 2) == 1
        width = width + 1;
    end
    width = min(width, lengthY);
end


function [startIdx, stopIdx] = band_bounds(centerRow, width, lengthY)
    width = normalize_filter_width(width, lengthY);
    startIdx = centerRow - floor(width / 2);
    stopIdx = startIdx + width;
    if startIdx < 0
        stopIdx = stopIdx - startIdx;
        startIdx = 0;
    end
    if stopIdx > lengthY
        startIdx = startIdx - (stopIdx - lengthY);
        stopIdx = lengthY;
    end
    startIdx = max(0, startIdx);
    stopIdx = min(lengthY, stopIdx);
    if stopIdx - startIdx < 2
        error('export_two_sideband_holograms:InvalidFilterWidth', ...
            'Vertical filter band is too narrow after clamping.');
    end
end


function filterWindow = build_sideband_filter(lengthY, centerRow, filterWidthY, alpha)
    [startIdx, stopIdx] = band_bounds(centerRow, filterWidthY, lengthY);
    filterWindow = zeros(lengthY, 1);
    filterWindow((startIdx + 1):stopIdx) = tukey_window(stopIdx - startIdx, alpha);
end


function shifted = shift_band_to_center(filteredFt, sourceCenterRow)
    lengthY = size(filteredFt, 1);
    targetCenterRow = floor(lengthY / 2);
    shiftValue = targetCenterRow - sourceCenterRow;
    shifted = zeros(size(filteredFt));

    srcStart = max(0, -shiftValue);
    srcStop = min(lengthY, lengthY - shiftValue);
    dstStart = max(0, shiftValue);
    dstStop = min(lengthY, lengthY + shiftValue);

    shifted((dstStart + 1):dstStop, :) = filteredFt((srcStart + 1):srcStop, :);
end


function windowValues = tukey_window(width, alpha)
    if width <= 0
        windowValues = zeros(0, 1);
        return;
    end
    if alpha <= 0
        windowValues = ones(width, 1);
        return;
    end
    if alpha >= 1
        windowValues = hann_window(width);
        return;
    end

    x = linspace(0, 1, width).';
    windowValues = ones(width, 1);
    firstRegion = x < alpha / 2;
    thirdRegion = x >= (1 - alpha / 2);

    windowValues(firstRegion) = 0.5 .* ...
        (1 + cos((2 .* pi ./ alpha) .* (x(firstRegion) - alpha / 2)));
    windowValues(thirdRegion) = 0.5 .* ...
        (1 + cos((2 .* pi ./ alpha) .* (x(thirdRegion) - 1 + alpha / 2)));
end


function windowValues = hann_window(width)
    if width == 1
        windowValues = 1;
        return;
    end
    n = (0:(width - 1)).';
    windowValues = 0.5 .* (1 - cos((2 .* pi .* n) ./ (width - 1)));
end


function phaseImage = processed_phase(image)
    phaseImage = unwrap(angle(image), [], 1);
    phaseImage = correct_baseline_slope(phaseImage);
end


function corrected = correct_baseline_slope(zValues)
    [rows, cols] = size(zValues);
    [xVals, yVals] = meshgrid(0:(cols - 1), 0:(rows - 1));
    edgeY = max(1, floor(rows / 20));
    edgeX = max(1, floor(cols / 20));
    fitMask = true(rows, cols);
    if rows > 2 * edgeY && cols > 2 * edgeX
        fitMask(1:edgeY, :) = false;
        fitMask((rows - edgeY + 1):rows, :) = false;
        fitMask(:, 1:edgeX) = false;
        fitMask(:, (cols - edgeX + 1):cols) = false;
    end
    fitMask = fitMask & isfinite(zValues);
    if nnz(fitMask) < 3
        fitMask = isfinite(zValues);
    end

    matrixA = [xVals(fitMask), yVals(fitMask), ones(nnz(fitMask), 1)];
    matrixB = zValues(fitMask);
    coeffs = matrixA \ matrixB;
    zFit = coeffs(1) .* xVals + coeffs(2) .* yVals + coeffs(3);
    corrected = zValues - zFit;
end
