# Microscopy Imaging Illumination and Contrast Evaluation

![MIICE logo](https://raw.githubusercontent.com/wayscience/MIICE/main/media/logo/MIICE_logo_v1.png?raw=true)

> Don’t let your images squeak by untested! 🐭🧀

It is nearly impossible to acquire "perfect" images across large high-throughput screens.
Technical artifacts from the microscope can impact all of the images (like uneven illumination) or some of the images (out-of-focus, over-saturation, or poor contrast).
Most modern microscopes can attempt to correct image quality issues, but these correction are not foolproof and they are rarely available to analysts.

We introduce MIICE, a Python package to evaluate microscopy images for quality issues like uneven illumination and poor contrast prior to perform image analysis.

With MIICE, you can test unmodified or corrected microscopy images to test the quality of the illumination and contrast.
The results can be used to determine next steps, including:

1. Decision to perform illumination correction on a microscopy dataset that was said to be already corrected by the microscope.
1. Confirm optimal parameters or refine for illumination correction on a dataset.
1. Flag and/or filter out low contrast image sets in a dataset.

## Installation

```shell
pip install "git+https://https://github.com/WayScience/MIICE"
```

## Contributing, Development, and Testing

Please see our [contributing](./CONTRIBUTING.md) documentation for more details on contributions, development, and testing.
