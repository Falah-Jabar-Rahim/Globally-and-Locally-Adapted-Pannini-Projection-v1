# Introduction


Python implementation of locally adapted Pannini projection (LAPP). LAPP is a content-aware sphere to plane projection proposed in [1] for viewport rendering of omnidirectional images; it is based on Pannini projection proposed in [2]. This projection was developed to improve the performance of globally adapted Pannini projection (GAPP) proposed in [3]. GAPP is globally adapted to the viewport content (i.e., the projection parameters d and vc have the same values for the whole viewport), therefore the geometric distortions such as stretching and/or bending may be still visible for some image regions and structures. In LAPP the projection parameters are allowed to vary locally, and thus geometric distortions are further reduced, especially for the foreground objects. LAPP is used the similar procedure as in [4]. Refer to [3] and [1] for more details on GAPP and LAPP respectively.

[]!(https://github.com/jwtyar/Locally-Adapted-Pannini-Projection/blob/main/Results.bmp)




# References
[1] F. Jabar, J. Ascenso, and M.P. Queluz, “Globally and Locally Optimized Pannini Projection for Viewport Rendering of 360° Images”, Submitted to J. Vis. Commun. Image     Represent., Oct. 2022 \
[2] T. K.Sharpless, B. Postle, and D. M.German, “Pannini : A New Projection for Rendering Wide Angle Perspective Images,” in Proc. of the 6th Int. Conf. on Computational     Aesthetics in Graphics, Visualization and Imaging, London, United Kingdom, Jul. 2010.\
 [3] F. Jabar, J. Ascenso, and M.P. Queluz “Object-Based Geometric Distortion Metric for Viewport Rendering of 360⸰ Images”, IEEE Access, vol. 10, pp. 13827-13843, Jan.        2022. \
[4] Y. Shih, W. Lai, and C. Liang, “Distortion-Free Wide-Angle Portraits on Camera Phones,” ACM Trans. Graph., vol. 38, no. 4, pp. 1–12, Jul. 2019.
