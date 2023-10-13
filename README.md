# Flow-Based-Bilaterial-Filter
 Flow Based Bilaterial Filter

## Implementation of the Orientation-Aligned Bilaterial Filter.
Bilaterial filter is one of the simplest filters and has a wide range of uses like denoising, tone management, enhancements. <br>
But the most important feature of the Bilaterial filter is its ability to smooth images while also perserving edges or discontinuities, this allows it to be one of the best filters when it comes to image abstraction in image processing term or image features reduction in deep/machine learning term. <br><br>

While taking performance and the ability to produce quality images into account the Bilterial filter scores low as it's not separable (at least not without leaving many artifacts), also a Bileterial filter that convolves over all of the image will mostly just blur it instead of smoothing and keeping the discontinuities. <br>

That's why we have to take local orientation of the image (Structure tensor) into account, we simply filter along  along the local gauge coordinates of the image, instead of along the coordinate axes. In case of an edge, filtering along the gradient direction results in strengthening of the edge, while filtering along the tangent direction results in a smoothing operation along the edge, avoiding artifacts and producing more coherent region boundaries.

<table>
  <tr>
    <th>Original</th>
    <th>F-Bilaterial</th>
  </tr>
  <tr>
    <td><img src="/results/Cat0/og.png"></td>
    <td><img src="/results/Cat0/flow.png"></td>
  </tr>
</table>

<table>
  <tr>
    <th>F-Bilaterial SigmaR=0.0525</th>
    <th>F-Bilaterial SigmaR=0.0625 2 Iterations</th>
  </tr>
  <tr>
    <td><img src="/results/Cat0/flow2.png"></td>
    <td><img src="/results/Cat0/flow3.png"></td>
  </tr>
</table>

## Color quantization.
To give the image more "abstraction" we can quantize the image to have a limited palette. It works really well with the Bilaterial filter smoothing and give the image better abstracted look.<br><br>
<img src="/results/Cat0/quant.png"><br><br>
## Pixelization
Since we now have a limited palette and a very abstracted image, we can apply pixelization filter on the result to get a nicely pixelated image of different resolutions.<br>
Without the two previous steps, pixelization would look really bad and it will have alot of colors that hurt the eyes.
<table>
  <tr>
    <th>F-Bilaterial</th>
    <th>F-Bilaterial SigmaR=0.0625 2 Iterations</th>
  </tr>
  <tr>
    <td><img src="/results/Cat0/pixel.png"></td>
    <td><img src="/results/Cat0/pixe3.png"></td>
  </tr>
</table>

# Results/Samples
NOTE: Please refer to the full resolution images if you want to see the smoothing as I didn't apply too much smoothing in my examples.
## 1) Flower
<table>
  <tr>
    <th>Original</th>
    <th>F-Bilaterial</th>
  </tr>
  <tr>
    <td><img src="/results/Flower/og.png"></td>
    <td><img src="/results/Flower/flow.png"></td>
  </tr>
</table>
<table>
  <tr>
    <th>Quantization</th>
    <th>Pixelated</th>
  </tr>
  <tr>
    <td><img src="/results/Flower/quant.png"></td>
    <td><img src="/results/Flower/pixel.png"></td>
  </tr>
</table>

## 2) Silly cat on pc
<table>
  <tr>
    <th>Original</th>
    <th>F-Bilaterial</th>
  </tr>
  <tr>
    <td><img src="/results/Cat/og.png"></td>
    <td><img src="/results/Cat/flow.png"></td>
  </tr>
</table>
<table>
  <tr>
    <th>Quantization</th>
    <th>Pixelated</th>
  </tr>
  <tr>
    <td><img src="/results/Cat/quant.png"></td>
    <td><img src="/results/Cat/pixel.png"></td>
  </tr>
</table>

## 3) lena
<table>
  <tr>
    <th>Original</th>
    <th>F-Bilaterial</th>
  </tr>
  <tr>
    <td><img src="/results/lena/og.png"></td>
    <td><img src="/results/lena/flow.png"></td>
  </tr>
</table>
<table>
  <tr>
    <th>Quantization</th>
    <th>Pixelated</th>
  </tr>
  <tr>
    <td><img src="/results/lena/quant.png"></td>
    <td><img src="/results/lena/pixel.png"></td>
  </tr>
</table>

## 4) Dog
<table>
  <tr>
    <th>Original</th>
    <th>F-Bilaterial</th>
  </tr>
  <tr>
    <td><img src="/results/Dog/og.png"></td>
    <td><img src="/results/Dog/flow.png"></td>
  </tr>
</table>
<table>
  <tr>
    <th>Quantization</th>
    <th>Pixelated</th>
  </tr>
  <tr>
    <td><img src="/results/Dog/quant.png"></td>
    <td><img src="/results/Dog/pixel.png"></td>
  </tr>
</table>
