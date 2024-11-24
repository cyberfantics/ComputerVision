import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

class Repeatable:
    def __init__(self, resize_width=None, resize_height=None, base_path='data/images/'):
        '''Initializes the Repeatable object with optional resizing parameters.'''
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.base_path = base_path

    def showImage(self, img, name="Image", show_axis=False, save_path=None):
        '''Displays a single image using matplotlib with a given title, 
        optionally saving it and toggling axis visibility.

        Args:
            img (ndarray): The image to be displayed (BGR format).
            name (str): The title of the image to be displayed.
            show_axis (bool): If True, displays the axis on the image.
            save_path (str): If provided, saves the image to the specified path.
        '''
        # Resize the image if the resize dimensions are specified
        if self.resize_width and self.resize_height:
            img = cv2.resize(img, (self.resize_width, self.resize_height))

        # Display the image using Matplotlib
        plt.imshow(img)
        plt.axis('on' if show_axis else 'off')
        plt.title(name)

        # Save the image if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    def show_image_with_color(self, img, name="Image", show_axis=False, save_path=None):
        '''Displays an image using matplotlib with a given title, 
        optionally saving it and toggling axis visibility.

        Args:
            img (ndarray): The image to be displayed (BGR format).
            name (str): The title of the image to be displayed.
            show_axis (bool): If True, displays the axis on the image.
            save_path (str): If provided, saves the image to the specified path.
        '''
        # Resize the image if the resize dimensions are specified
        if self.resize_width and self.resize_height:
            img = cv2.resize(img, (self.resize_width, self.resize_height))

        # Convert BGR to RGB for correct display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.imshow(img_rgb)
        plt.axis('on' if show_axis else 'off')
        plt.title(name)

        # Save the image if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    def show_multiple_images(self, images, titles=None, cols=3, show_axis=False, save_path=None, resize=True):
        '''
        Displays multiple images using Matplotlib subplots.

        Args:
            images (list of ndarray): List of images to display (BGR format).
            titles (list of str, optional): List of titles for each image.
            cols (int, optional): Number of columns in the subplot grid. Defaults to 2.
            show_axis (bool, optional): If True, displays axes for all images. Defaults to False.
            save_path (str, optional): If provided, saves the figure to the specified path.
        '''
        if not isinstance(images, (list, tuple)):
            raise TypeError("Images should be provided as a list or tuple of ndarray images.")

        num_images = len(images)
        if titles and len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")

        # Calculate the number of rows needed
        rows = math.ceil(num_images / cols)

        # Create a matplotlib figure
        plt.figure(figsize=(5 * cols, 5 * rows))

        for idx, img in enumerate(images):
            # Resize the image if the resize dimensions are specified
            if self.resize_width and self.resize_height and resize:
                print('Resizing Image')
                img = cv2.resize(img, (self.resize_width, self.resize_height))

            # Convert BGR to RGB for correct display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Add a subplot for each image
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img_rgb)
            ax.axis('on' if show_axis else 'off')
            if titles:
                ax.set_title(titles[idx])

        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    def shape_of_images(self, *args):
        '''Checks and prints the shapes of multiple images.

        Args:
            *args: A variable number of image arrays (ndarrays).
        '''
        for idx, img in enumerate(args, start=1):
            try:
                print(f'Image {idx} shape: {img.shape}')
            except:
                print(f'Image {idx} is not a valid NumPy array.')

    def writeText(self, img, text, color=(120, 255, 80), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=3):
        """
        Adds centered text to an image with customizable font, color, and size.
    
        Parameters:
            img (numpy.ndarray): The image on which the text will be placed.
            text (str): The text string to be added to the image.
            color (tuple): The color of the text in BGR format (default is light green).
            font (int): The font to be used (default is cv2.FONT_HERSHEY_SIMPLEX).
            font_scale (float): The scale of the font (default is 1).
            thickness (int): The thickness of the text (default is 3).
    
        Returns:
            numpy.ndarray: The image with the text drawn on it.
        """
        
        # Get the image dimensions
        h, w = img.shape[:2]
        
        # Calculate the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate the center position
        center_x = (w - text_width) // 2
        center_y = (h + text_height) // 2  # Add height to vertical center so the text is not cut off
        
        # Put the text on the image
        img = cv2.putText(img, text, (center_x, center_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
        # Return the image with the text
        return img

    # Function to load image
    def load_image(self, img='bin_Salman.jpg', flag=1):
        """
        Loads an image from the predefined directory.

        Parameters:
        img (str): The name of the image file to load. Default is 'bin_Salman.jpg'.

        Returns:
        img (numpy.ndarray): The loaded image as a NumPy array, or None if the image is not found.
        
        This function constructs the full image path by combining the base path
        with the provided image file name and loads the image using OpenCV.
        If the base path changes in the future, it needs to be updated in only
        one place (the 'base_path' class variable).
        """
        # Construct the full image path using the base path
        img_path = self.base_path + img
        
        # Load and return the image
        return cv2.imread(img_path, flag)


    # Show Images Through OpenCV
    def showImagesThroughOpenCV(images, titles, delay=0, window_size=None, window_position=None):
        """
        Displays a list of images in separate OpenCV windows with specified titles.
        
        Args:
        images (list of numpy arrays): List of images to be displayed. Each image is a numpy array representing an image.
        titles (list of str): List of titles for each window that will display the images.
        delay (int, optional): Delay in milliseconds to display each image. Default is 0 (wait indefinitely for key press).
        window_size (tuple, optional): Tuple specifying the width and height to resize each window (width, height). 
                                    If None, windows will be displayed in their original size. Default is None.
        window_position (tuple, optional): Tuple specifying the (x, y) position to place the window. Default is None.
        
        Raises:
        ValueError: If the number of images does not match the number of titles.
        FileNotFoundError: If an image fails to load or is None.
        """
        # Check if both lists are of the same length
        if len(images) != len(titles):
            raise ValueError("The number of images and titles must be the same.")

        # Check if either list is empty
        if not images or not titles:
            raise ValueError("Images or titles list is empty.")

        # Iterate through the images and titles
        for idx, (image, title) in enumerate(zip(images, titles)):
            # Check if the image is valid
            if image is None:
                raise FileNotFoundError(f"Image at index {idx} failed to load or is None.")

            # Set window properties if provided
            if window_size:
                cv2.resizeWindow(title, window_size[0], window_size[1])
            if window_position:
                cv2.moveWindow(title, window_position[0], window_position[1] + idx * 50)  # Slight vertical offset for each window

            # Show the image
            cv2.imshow(title, image)
            
            # Wait for a specified delay or until a key is pressed
            key = cv2.waitKey(delay)
            if key == ord('q'):  # If 'q' is pressed, break the loop and close all windows
                print("Closing windows...")
                break

        # Close all windows after the key press or delay
        cv2.destroyAllWindows()
 