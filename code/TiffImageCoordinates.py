import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image_with_coordinates(filename):
    image = mpimg.imread(filename)
    plt.imshow(image)

    def on_mouse_click(event):
        x, y = event.xdata, event.ydata
        print("Coordinates: (", x, ", ", y, ")")

    plt.connect("button_press_event", on_mouse_click)
    plt.show()

if __name__ == "__main__":
    filename = "Images/test2.tif"
    show_image_with_coordinates(filename)
