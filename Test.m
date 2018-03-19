load PA3Models.mat;
load PA3SampleCases.mat;
image1 = Part1SampleImagesInput;

factors = ComputeSingletonFactors(image1, imageModel);
factors = ComputeTripletFactors(image1,tripletList,26);
factors = ComputePairwiseFactors(image1,pairwiseModel,26);
factors = RunInference(BuildOCRNetwork(images,imageModel,[],[]));
result = ScoreModel(images,imageModel,[],[]);
result = ScoreModel(image1,imageModel,pairwiseModel,[]);

images = Part2SampleImagesInput;
factors = ComputePairwiseFactors(images, pairwiseModel, imageModel.K);
factors = SortAllFactors(factors);
out = SerializeFactorsFgGrading(factors);