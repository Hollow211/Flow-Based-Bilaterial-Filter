# Flow-Based-Bilaterial-Filter
 Flow Based Bilaterial Filter

## Implementation of the Flow-Based Bilaterial Filter.
Bilaterial filter is one of the simplest filters and has a wide range of uses like denoising, tone management, enhancements. <br>
But the most important feature of the Bilaterial filter is its ability to smooth images while also perserving edges or discontinuities, this allows it to be one of the best filters when it comes to image abstraction in image processing term or image features reduction in deep/machine learning term. <br><br>

While taking performance and the ability to produce quality images into account the Bilterial filter scores low as it's not separable (at least not without leaving many artifacts), also a Bileterial filter that convolves over all of the image will mostly just blur it instead of smoothing and keeping the discontinuities. <br>

That's why we have to take local orientation of the image (Structure tensor) into account, we simply filter along  along the local gauge coordinates of the image, instead of along the coordinate axes. In case of an edge, filtering along the gradient direction results in strengthening of the edge, while filtering along the tangent direction results in a smoothing operation along the edge, avoiding artifacts and producing more coherent region boundaries.
