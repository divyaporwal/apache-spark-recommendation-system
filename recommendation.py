from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

data = sc.textFile("/FileStore/tables/ratings.dat")
ratings = data.map(lambda l: l.split('::')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
train_RDD, test_RDD = ratings.randomSplit([.6, .4], seed=1234)
test_for_prediction_RDD = test_RDD.map(lambda x: (x[0], x[1]))
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
best_rank = -1
min_error = float('inf')
best_iteration = -1
numIterations = 10
for rank in ranks:
  model = ALS.train(train_RDD, rank, numIterations,lambda_=.1)
  testdata = test_for_prediction_RDD
  predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
  ratingsAndPreds = test_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
  MSE = ratingsAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
  
  if MSE < min_error:
    min_error = MSE
    best_rank = rank
	
  print("Mean Squared Error for rank "+str(rank)+ " is : "+ str(MSE/4))
  errors[err] = MSE
  err += 1

print("The best model was trained with rank = " + str(best_rank))
