from interface import Ui_MainWindow
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QLabel,
    QWidget,
    QVBoxLayout,
    QApplication,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import os
from detection import detection
from segmentation import segmentation

class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        # self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.load_folder)
        self.pushButton_2.clicked.connect(self.previous_image)
        self.pushButton_3.clicked.connect(self.next_image)
        self.pushButton_4.clicked.connect(self.run_detection)
        self.pushButton_5.clicked.connect(self.run_segmentation)

        self.image_list = []
        self.current_image_index = 0
        self.image_folder_path = None
        # Position the main window to the left side of the screen
        self.move_to_left()

    def move_to_left(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        screen_height = screen_geometry.height()
        screen_width = screen_geometry.width()
        window_width = self.geometry().width()
        window_height = self.geometry().height()
        x_position = int(screen_width / 4 - window_width / 2)
        y_position = int((screen_height - window_height) / 2)
        self.setGeometry(x_position, y_position, window_width, window_height)

    def load_folder(self):
        self.image_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.image_folder_path:
            # Filter out only common image file types
            image_extensions = (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".gif",
                ".tiff",
                ".tif",
            )  # Add or remove extensions as needed
            self.image_list = [
                f
                for f in os.listdir(self.image_folder_path)
                if f.lower().endswith(image_extensions)
                and os.path.isfile(os.path.join(self.image_folder_path, f))
            ]
            self.current_image_index = 0
            if self.image_list:
                self.display_image()
                self.pushButton.setEnabled(False)
            else:
                self.label.setText("No images found in the selected folder.")

    def display_image(self):
        if self.image_list:
            image_path = os.path.join(
                self.image_folder_path, self.image_list[self.current_image_index]
            )
            self.label.setText(
                f"Current Image: {self.image_list[self.current_image_index]}"
            )
            pixmap = QPixmap(image_path)

            # Create a new widget to display the image
            self.image_window = QWidget()
            self.image_window.setWindowTitle("Original")  # Set the window title here

            # Calculate the aspect ratio of the image
            aspect_ratio = pixmap.width() / pixmap.height()
            # Set a fixed width or adjust as needed
            window_width = 600
            window_height = int(window_width / aspect_ratio)

            # Resize the window based on the aspect ratio
            self.image_window.resize(window_width, window_height)

            # Set up a layout and add a label to it
            layout = QVBoxLayout()
            image_label = QLabel()
            image_label.setPixmap(
                pixmap.scaled(
                    window_width, window_height, Qt.AspectRatioMode.KeepAspectRatio
                )
            )  # Scale the pixmap to fit the window

            layout.addWidget(image_label)

            # Set the layout for the widget and show it
            self.image_window.setLayout(layout)
            self.image_window.show()

    def previous_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_image()
    
    def run_detection(self):
        if self.image_list:
            current_image_path = os.path.join(self.image_folder_path, self.image_list[self.current_image_index])
            overall_Precision, overall_Recall, overall_Accuracy, overall_IoU = detection(current_image_path)
            print(f"Overall Metrics: Precision: {overall_Precision}, Recall: {overall_Recall}, Accuracy: {overall_Accuracy}, IoU: {overall_IoU}")
            # Update labels with new values
            self.label_2.setText(f"IoU: {round(overall_IoU, 4)}")
            self.label_3.setText(f"Accuracy: {round(overall_Accuracy, 4)}")
            self.label_4.setText(f"Precision: {round(overall_Precision, 4)}")
            self.label_5.setText(f"Recall: {round(overall_Recall, 4)}")
    
    def run_segmentation(self):
        if self.image_list:
            current_image_path = os.path.join(self.image_folder_path, self.image_list[self.current_image_index])
            predicted_class_label, dice_score = segmentation(current_image_path)
            self.label_6.setText(f"Dice Coefficient: {round(dice_score, 4)} \nClassification: {predicted_class_label}")

# The entry point of the application
if __name__ == "__main__":
    import sys

    # Initialize and run the application
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
