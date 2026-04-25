from ast import alias
from concurrent.futures import process
from django.shortcuts import render
from PIL import Image
import datetime

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Scalable_Deep_Learning_for_Categorization_of_Satellite_Images

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
 


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            try:
                user = form.save(commit=False)
                user.status = 'activated'  # Ensure status is set to activated
                user.save()
                print(f"User registered successfully: {user.loginid}")
                messages.success(request, 'You have been successfully registered')
                return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
            except Exception as e:
                print(f"Registration error: {str(e)}")
                messages.error(request, f'Registration error: {str(e)}')
        else:
            print("Form validation failed")
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        
        if not loginid or not pswd:
            messages.error(request, 'Please enter both login ID and password')
            return render(request, 'UserLogin.html')
        
        try:
            # Get user and validate credentials
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            
            # Check if account is activated
            if user.status != "activated":
                messages.error(request, 'Your account is not activated')
                return render(request, 'UserLogin.html')
            
            # Set up session for satellite image validation
            request.session['id'] = user.id
            request.session['loggeduser'] = user.name
            request.session['loginid'] = loginid
            request.session['email'] = user.email
            
            print(f"User {user.name} logged in successfully")
            
            # Redirect to satellite image classification interface
            context = {
                'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
                'image_requirements': {
                    'min_resolution': '100x100',
                    'color_modes': ['RGB', 'Grayscale'],
                    'aspect_ratio': '0.5 to 2.0'
                }
            }
            return render(request, 'users/UserHomePage.html', context)
            
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid login credentials')
        except Exception as e:
            print(f"Login error: {str(e)}")
            messages.error(request, 'An error occurred during login')
    
    return render(request, 'UserLogin.html')


def UserHome(request):
    # Check if user is logged in
    if 'loggeduser' not in request.session:
        messages.error(request, 'Please login to access this page')
        return render(request, 'UserLogin.html')
        
    # Pass satellite image classification context
    context = {
        'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
        'image_requirements': {
            'min_resolution': '100x100',
            'color_modes': ['RGB', 'Grayscale'],
            'aspect_ratio': '0.5 to 2.0'
        }
    }
    return render(request, 'users/UserHomePage.html', context)



# def DatasetView(request):
#     path = settings.MEDIA_ROOT + "//" + 'data.csv'
#     df = pd.read_csv(path, nrows=100)
#     df = df.to_html
#     return render(request, 'users/viewdataset.html', {'data': df})


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

 

def ml(request):
    
    # Create an empty dataframe
    data=pd.DataFrame()
    data = pd.DataFrame(columns=['image_path', 'label'])

    labels = {
        os.path.join(settings.MEDIA_ROOT, 'data', 'cloudy'): 'Cloudy',
        os.path.join(settings.MEDIA_ROOT, 'data', 'desert'): 'Desert',
        os.path.join(settings.MEDIA_ROOT, 'data', 'green_area'): 'Green_Area',
        os.path.join(settings.MEDIA_ROOT, 'data', 'water'): 'Water',
    }

    for folder in labels:
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            label = labels[folder]
            new_data = pd.DataFrame({'image_path': image_path, 'label': label}, index=[0])
            data = pd.concat([data, new_data])

    # Save the data to a CSV file
    csv_path = settings.MEDIA_ROOT + '//'  + 'image_dataset1.csv'      
    data.to_csv(csv_path, index=False)
      
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

    df =pd.read_csv(settings.MEDIA_ROOT + '//'  + 'image_dataset1.csv')      

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Pre-process the data
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       rotation_range=45,
                                       vertical_flip=True,
                                       fill_mode='nearest')


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        x_col="image_path",
                                                        y_col="label",
                                                        target_size=(255, 255),
                                                        batch_size=42,
                                                        class_mode="categorical")

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      x_col="image_path",
                                                      y_col="label",
                                                      target_size=(255, 255),
                                                      batch_size=42,
                                                      class_mode="categorical") 
    
    # Build a deep learning model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), input_shape=(253, 253, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    path = settings.MEDIA_ROOT + '//' + 'Model.h5'
    model = load_model(path)
    num_samples = test_df.shape[0]
    score = model.evaluate(test_generator, steps=num_samples // 42 + 1)

    # Get predictions
    y_pred = model.predict(test_generator, steps=num_samples // 42 + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calculate and print classification report
    report = classification_report(true_classes, y_pred_classes, target_names=class_labels)
    print("Classification Report:\n", report)

    # Calculate and plot confusion matrix
    cm = confusion_matrix(true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png'))
    plt.close()

    # Render the results to an HTML template
    return render(
        request,
        'users/ml.html',
        {
            'accuracy': score[1],  # Assuming accuracy is at index 1 in score list
            'classification_report': report,
            'confusion_matrix_image': '/media/confusion_matrix.png',  # Replace with your path
        }
    )


def is_likely_satellite_image(image):
    try:
        # Check image dimensions (satellite images typically have specific resolutions)
        img = Image.open(image)
        width, height = img.size
        
        # Check if image has appropriate channels (RGB or multispectral)
        if img.mode not in ['RGB', 'L', 'RGBA']:  # Added RGBA support
            print(f"Invalid image mode: {img.mode}")
            return False
            
        # More lenient minimum resolution
        if width < 50 or height < 50:  # Reduced from 100 to 50
            print(f"Image too small: {width}x{height}")
            return False
            
        # More lenient aspect ratio check
        aspect_ratio = width / height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Widened the acceptable range
            print(f"Invalid aspect ratio: {aspect_ratio}")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating image: {str(e)}")
        return False

def is_image_in_dataset(image):
    try:
        # Convert uploaded image to numpy array for comparison
        img = Image.open(image)
        img = img.convert('RGB')  # Convert to RGB mode for consistency
        img = img.resize((255, 255))  # Resize to match our model's input size
        img_array = np.array(img)

        # Dataset directories
        dataset_dirs = [
            os.path.join(settings.MEDIA_ROOT, 'data', 'cloudy'),
            os.path.join(settings.MEDIA_ROOT, 'data', 'desert'),
            os.path.join(settings.MEDIA_ROOT, 'data', 'green_area'),
            os.path.join(settings.MEDIA_ROOT, 'data', 'water')
        ]

        # Check each directory in the dataset
        for dir_path in dataset_dirs:
            if not os.path.exists(dir_path):
                print(f"Dataset directory not found: {dir_path}")
                continue

            for img_file in os.listdir(dir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        dataset_img_path = os.path.join(dir_path, img_file)
                        dataset_img = Image.open(dataset_img_path)
                        dataset_img = dataset_img.convert('RGB')  # Convert to RGB mode
                        dataset_img = dataset_img.resize((255, 255))
                        dataset_array = np.array(dataset_img)
                        
                        # Compare images with some tolerance for minor differences
                        difference = np.mean(np.abs(img_array - dataset_array))
                        if difference < 10:  # Allow small differences
                            return True
                    except Exception as e:
                        print(f"Error processing dataset image {img_file}: {str(e)}")
                        continue

        print("No matching image found in dataset")
        return False
    except Exception as e:
        print(f"Error checking dataset: {str(e)}")
        return False

def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        
        try:
            # First check if it's a valid satellite image (255x255x3)
            if not is_likely_satellite_image(uploaded_image):
                messages.error(request, 'Please upload a valid satellite image with dimensions 255x255 pixels in RGB format.')
                return render(request, 'users/predictForm.html', {
                    'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
                    'requirements': {
                        'dimensions': '255x255 pixels',
                        'format': 'RGB',
                        'categories': ['Cloudy', 'Desert', 'Green Area', 'Water']
                    }
                })
            
            # Then check if the image is from our dataset
            if not is_image_in_dataset(uploaded_image):
                messages.error(request, 'The uploaded image must be from our training dataset. Please select an image from one of our dataset categories.')
                return render(request, 'users/predictForm.html', {
                    'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
                    'requirements': {
                        'dimensions': '255x255 pixels',
                        'format': 'RGB',
                        'categories': ['Cloudy', 'Desert', 'Green Area', 'Water']
                    }
                })
            
            # Load the trained model
            model_path = os.path.join(settings.MEDIA_ROOT, 'Model.h5')
            model = load_model(model_path)
            
            class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
            
            # Save the uploaded image in a structured way
            fs = FileSystemStorage()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{uploaded_image.name}"
            filename = fs.save(f"uploads/{safe_filename}", uploaded_image)
            uploaded_file_url = fs.url(filename)
            
            # Load and preprocess the image for prediction
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            img = load_img(image_path, target_size=(255, 255))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions[0])
            predicted_label = class_names[class_index]
            
            # Calculate confidence scores
            confidence_scores = {
                class_name: float(score) * 100 
                for class_name, score in zip(class_names, predictions[0])
            }
            sorted_confidence = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
            
            # Get model architecture summary
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            
            # Model information
            model_info = {
                'input_shape': model.input_shape,
                'num_layers': len(model.layers),
                'total_params': model.count_params(),
            }

            print(f"Image '{safe_filename}' classified as '{predicted_label}' with {sorted_confidence[predicted_label]:.2f}% confidence")

            context = {
                'predicted_image': uploaded_file_url,
                'predicted_label': predicted_label,
                'confidence_scores': sorted_confidence,
                'model_info': model_info,
                'model_summary': '\n'.join(model_summary),
                'preprocessing_steps': [
                    'Resize image to 255x255 pixels',
                    'Convert to RGB format',
                    'Normalize pixel values to range [0,1]',
                    'Validate against dataset images',
                    'Apply data augmentation preprocessing'
                ]
            }
            
            return render(request, 'users/prediction.html', context)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            messages.error(request, 'An error occurred while processing the image. Please try again.')
            return render(request, 'users/predictForm.html', {
                'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
                'requirements': {
                    'dimensions': '255x255 pixels',
                    'format': 'RGB',
                    'categories': ['Cloudy', 'Desert', 'Green Area', 'Water']
                }
            })

    # GET request - show the upload form
    return render(request, 'users/predictForm.html', {
        'categories': ['Cloudy', 'Desert', 'Green_Area', 'Water'],
        'requirements': {
            'dimensions': '255x255 pixels',
            'format': 'RGB',
            'categories': ['Cloudy', 'Desert', 'Green Area', 'Water']
        }
    })