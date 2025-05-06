# SHuBERT needs feature extractor, thus we train Amit's VQ-VAE first, which we can just reuse his code for
# SHuBERT had 4 "streams", which I could kinda get by using instead 3 codebooks (face,body,hands) 

# Receives [Batch, Length, Keypoints, Dim]
# Sends it through Amit's VQ-VAE encoder to get Quantized Features 
# Generate pseudo-labels by applying kmeans to data, either using sklearn with data in numpy form or using kmeans_pytorch https://github.com/subhadarship/kmeans_pytorch (can use gpu)
# Mask features randomly (easiest and many papers have said this works well for computer vision)
# Send masked features through transformer encoder with linear projection at end to same dimension as cluster
# Compare predicted vs true cluster

# Alternatively, I could use wavLM's CNN encoder like structure, but just have 3 variantes (one each for face, body, hands) and apply k-means
# on the features during training to get the pseudo-labels, and the transformer encoder should be able to stay the same then IMO
# as the wavLM was made for pseudo-label prediction
# the CNN is followed by linear projection to 768 dim and the transformer outputs X feature vectors with 768 dim, so I can just compare
# check if the predicted feature would fit to the same cluster for all masked features. 