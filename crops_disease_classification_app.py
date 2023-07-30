import streamlit as st
from PIL import Image

# Create a radio button group with three options
selected_option = st.radio("Select an option:", ('Corn', 'Potato', 'Wheat', 'Rice'))
# Display the selected option
st.write(f"You selected: {selected_option}")

file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Perform inference on a single image
def classify_image(image, model_ft, class_labels):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model_ft(image)
        _, predicted_class = torch.max(output, 1)
        
        prob = torch.nn.functional.softmax(output, dim = 1)[0] * 100
        _, indices = torch.sort(output, descending = True)
    return [(class_labels[idx], prob[idx].item()) for idx in indices[0][:3]]
        
        # class_idx = predicted_class.item()
        # class_label = class_labels[class_idx]
    # return class_label
 
#Options
if selected_option == "Corn":
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)

    path = 'F:/Semester _8/FYP2/corn/corn_resnet18_best.pt'
    device = torch.device('cpu')
    model_ft.load_state_dict(torch.load(path, map_location=device))
    model_ft.to(device)
    model_ft.eval()

    # Define the class labels
    class_labels = ['Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Leaf_Blight', 'Invalid']

elif selected_option == "Wheat":
     model_ft = models.resnet18(pretrained=False)
     num_ftrs = model_ft.fc.in_features
     model_ft.fc = nn.Linear(num_ftrs, 4)

     path = 'F:/Semester _8/FYP2/wheat/wheat_resnet18_best.pt'
     device = torch.device('cpu')
     model_ft.load_state_dict(torch.load(path, map_location=device))
     model_ft.to(device)
     model_ft.eval()
     class_labels = ['Invalid', 'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust']

elif selected_option == "Potato":
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    
    path = 'F:/Semester _8/FYP2/potato/potato_resnet18_best.pt'
    device = torch.device('cpu')
    model_ft.load_state_dict(torch.load(path, map_location=device))
    model_ft.to(device)
    model_ft.eval()
    # Define the class labels
    class_labels = ['Invalid', 'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight']
    
elif selected_option == "Rice":
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 5.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 5)
    
    # model_ft = model_ft.to(device)
    path = 'F:/Semester _8/FYP2/rice/rice_resnet50_best.pt'
    device = torch.device('cpu')
    model_ft.load_state_dict(torch.load(path, map_location=device))
    model_ft.to(device)
    model_ft.eval() 
# Define the class labels
    class_labels = ['Invalid', 'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Hispa', 'Rice___Leaf_Blast']

   
if file_up is not None:
    # Display the uploaded image    
    image = Image.open(file_up)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_classes = classify_image(image, model_ft, class_labels)
    # print out the top 3 prediction labels with scores
   
    for i in predicted_classes:
        st.write("Prediction:", i[0], ",   Score: ", i[1])
