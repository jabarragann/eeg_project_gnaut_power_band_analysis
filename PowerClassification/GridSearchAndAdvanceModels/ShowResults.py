import pickle



f = open('./gridResult.pickle','rb')

grid_result = pickle.load(f)

print("Best: %f using %s" % (grid_result[ 'best_score_'], grid_result['best_params']))
means = grid_result['cv_results']['mean_test_score']
stds = grid_result['cv_results']['std_test_score']
params = grid_result['cv_results']['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
