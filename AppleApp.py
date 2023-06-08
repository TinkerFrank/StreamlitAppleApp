import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("apple_vit_classifier.pt", map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5125,0.4667,0.4110],
                         std= [0.2621,0.2501,0.2453]),
])


def main():
    st.title("Apple Classification via Webcam")

    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)
    

    # Create an empty placeholder for the image
    image_placeholder = st.empty()

    # Create an empty placeholder for the predicted label
    label_placeholder = st.empty()

    # Continuously read frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()

        # Check if the frame was successfully read
        if not ret:
            st.error("Failed to read a frame from the camera.")
            break

        # Convert the OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a PIL image
        pil_image = Image.fromarray(rgb_frame)

        # Apply transformations to the image
        transformed_image = transform(pil_image)

        # Add a batch dimension to the image
        input_image = transformed_image.unsqueeze(0).to(device)

        # Pass the image through the model
        with torch.no_grad():
            outputs = model(input_image)

        # Perform classification
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = "Normal Apple" if predicted_idx == 1 else "Not Normal Apple"

         # Display the frame and predicted label
        image_placeholder.image(frame, channels="BGR", use_column_width=True)
        label_placeholder.text("Predicted Label: {}".format(predicted_label))

    # Release the VideoCapture object and close the Streamlit app
    video_capture.release()


if __name__ == "__main__":
    main()

#streamlit run streamlitapp.py