### Custom architecture for semantic segmentation of Modified Dataset. 
Forground mask of MNIST images are segmented using total sum of squares(like otsu). These images with corresponding masks are concatenated in size of 2x2 along with masks, resulting in input size of 56x56 with its corresponding mask. Pixels in mask are replaced by the value of corresponding class they belong to. Eg. Mask belonging to class 9, constains either background pixels or forground pixels viz 10 and 9 respectively.

Semantic segmentation is performed using the UNET inspired architecture.
#### Results
<img src="../main/CV/Assignment_3/Q4.png">
[Report](../master/CV/Assignment_3/Report.pdf)
