function DeepLearningImageClassificationExample

%outputFolder = fullfile(tempdir, 'M');
rootFolder = fullfile('UCMerced/Images/');
categories = {'agricultural','airplane','baseballdiamond','beach', 'buildings','chaparral',...
    'denseresidential', 'forest','freeway','golfcourse','harbor','intersection','mediumresidential',...
    'mobilehomepark','overpass', 'parkinglot','river','runway','sparseresidential',...
    'storagetanks' , 'tenniscourt'
    };

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

net = alexnet();
net.Layers
net.Layers(1)
% Inspect the last layer
net.Layers(end)

% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames)
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
function Iout = readAndPreprocessImage(filename)

        I = imread(filename);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for the CNN.
        Iout = imresize(I, [227 227]);

        % Note that the aspect ratio is not preserved. In Caltech 101, the
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
end

[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
    % Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')

featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'KNN', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32,'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

mean(diag(confMat))

newImage = fullfile(rootFolder, 'river', 'river20.tif');


% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);

% Extract image features using the CNN
imageFeatures = activations(net, img, featureLayer, 'OutputAs', 'columns');
% Make a prediction using the classifier
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

end

