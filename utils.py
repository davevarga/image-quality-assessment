import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot(images, layout=None, fig_size=None):
    """
    Plots a list of images in a grid layout. Works with both colored and grayscale images.
    :param images: A list of images (as NumPy arrays) that you want to plot.
    :param layout:An optional tuple specifying the number of rows and columns.
        If not provided, the function automatically determines the layout based
        on the number of images.
    :param fig_size: Specifies the proportions of each image.
    """
    num_images = len(images)

    # If layout is not provided, determine best fitting layout
    if layout is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = layout

    # Initialize figure size if not provided
    if fig_size is None:
        fig_size = (cols * 3, rows * 3)

    # Create a figure with the specified layout
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each image in the grid
    for i, image in enumerate(images):
        if len(image.shape) == 2:  # Grayscale image
            axes[i].imshow(image, cmap='gray')
        else:  # Colored image
            axes[i].imshow(image)
        axes[i].axis('off')  # Hide the axes

    # Turn off any remaining empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def gradients(image):
    """
    Calculate the first-order gradients of an image using Sobel filters.
    :param image: Input image, can be colored or grayscale (NumPy array).
    :return grad_x, rad_y: Gradient along the x-axis and y-axis.

    """
    # If the image is colored (3 channels), convert to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Image is already grayscale

    # Calculate gradients using Sobel filters
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Gradient along x-axis
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Gradient along y-axis

    return grad_x, grad_y


def first_norm(image):
    """
    Calculate the first norm of the image using Sobel filter
    :param image: Color or grayscale image.
    :return: The first norm of the image.
    """
    grad_x, grad_y = gradients(image)
    grad_1 = grad_x + grad_y
    return grad_1 * 255 /grad_1.max()


def second_norm(image):
    """
    Calculate the second norm of the image using Sobel filter
    :param image: Color or grayscale image.
    :return: Second norm of the image.
    """
    grad_x, grad_y = gradients(image)
    grad_2 = np.sqrt(grad_x.astype(np.uint16) ** 2 + grad_y.astype(np.uint16) ** 2)
    return grad_2 * 255 /grad_2.max()

