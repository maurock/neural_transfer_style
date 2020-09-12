# Neural Style Transfer using Deep Learning
Transfer painting styles to chosen images.

This project reproduces the paper "A Neural Algorithm of Artistic Style" by Gatys et al. (https://arxiv.org/abs/1508.06576).

## Run
To run the neural style transfer script:
- place the base image (the image to modify) in `reference_images\base_image`
- place the style image (the image from which the style is transferred) in `reference_images\style_image`
- run the script `main.py` in the main directory

After each iteration, an image is generated.

Run example:
```python
python main.py --base_image=jumping_me.jpg --style_image=starry_night.jpg --iterations=20
```

Arguments description:

- --base_image - Type string, default jumping_me.jpg, the image to modify
- --style_image - Type string, default starry_night.jpg, the image from which the style is transferred
- --iterations - Type int, default 20, the number of iterations
