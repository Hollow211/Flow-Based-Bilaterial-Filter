import Functions

# Variables
sigma = 1.5 # How much blur do we apply to get gradient for the strcture tensor
sigmaC = 2.3 # How much do we blur the structure tensor
sigmaD = 6
sigmaR = 0.0425
QuantizOutput = False
K = 21 # For color Quantization
PixelOutput = False
pixelSize = 128

image_path = "/content/lena.png"

Functions.Inference(image_path, pixelSize, K, sigmaD, sigmaR, QuantizOutput, PixelOutput)





