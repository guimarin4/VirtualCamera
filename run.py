# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)
        
        

        # Example of keyboard is pressed
        # If you want to use this method then consider implementing a counter
        # that ignores for example the next five keyboard press events to
        # "prevent" double clicks due to high fps rates
        if keyboard.is_pressed('h') :
            print('h pressed')
            

        ###
        ### Histogram overlay example (without data)
        ###
        
        # Load the histogram values
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the overlay
        sequence = plot_overlay_to_image(sequence, fig)
        
        ###
        ### END Histogram overlay example
        ###

        
        # Display text example
        display_text_arr = ["Test", "abc"]
        sequence = plot_strings_to_image(sequence, display_text_arr)

        
        # Make sure to yield your processed image
        yield sequence



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            # vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()