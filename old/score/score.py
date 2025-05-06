import numpy as np

def score(model_dir):
    print("This score function will evaluate the trained model on some test data that was not seen during training. \
          The evaluation metrics will then be stored but for now I will just store ones to see if it works.")
    # in reality load model from model dir and use it on eval data, then store the prediction result in the model's dir
    res = np.array([1,1,1,1,1,1,1,1])
    np.savetxt(model_dir + "/predict_result.csv", res, delimiter=",")


