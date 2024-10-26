AR T-Shirt Overlay Using Flask, OpenCV, and MediaPipe<br>

This project is a Flask-based web application that allows users to try on virtual t-shirts using augmented reality (AR) through a webcam feed. The t-shirt models are overlaid on the user’s body in real-time using OpenCV and MediaPipe.<br>

Features<br>
Real-time t-shirt overlay on webcam feed using OpenCV and MediaPipe.<br>
Dynamic selection of t-shirt styles.<br>
HTML, CSS, and Bootstrap used for UI design.<br>
Jinja2 for rendering dynamic templates.<br>
Tech Stack<br>
Backend: Flask<br>
Frontend: HTML, CSS, Bootstrap, Jinja2<br>
Computer Vision: OpenCV, MediaPipe<br>
Project Structure<br>
graphql
<br>
Copy code<br>
├── app.py              # Main Flask application<br>
├── templates/          # HTML templates with Jinja2<br>
│   ├── home.html       # Home page for webcam stream<br>
│   └── collection.html # Page for t-shirt selection<br>
├── static/             # Static files (t-shirt images, CSS)<br>
│   ├── css/            # Custom CSS files<br>
│   ├── js/             # JavaScript (if required)<br>
│   └── <tshirt_images>.png<br>
├── README.md           # Documentation<br>
└── requirements.txt    # Python dependencies<br>
Installation and Setup<br>
Prerequisites<br>
Python 3.x installed<br>
pip package manager<br>
Clone the Repository<br>
bash<br>
Copy code<br>
git clone <repository-url><br>
cd <repository-folder><br>
Install Dependencies<br>
Create a virtual environment (optional):<br>

bash<br>
Copy code<br>
python -m venv venv<br>
source venv/bin/activate  # On Windows: venv\Scripts\activate<br>
Install required Python packages:<br>

bash<br>
Copy code<br>
pip install -r requirements.txt<br>
Contents of requirements.txt:<br>

makefile<br>
Copy code<br>
Flask==2.1.0<br>
opencv-python==4.8.0.74<br>
mediapipe==0.10.1<br>
numpy==1.24.4<br>
Usage<br>
Run the Application:<br>

bash<br>
Copy code<br>
python app.py<br>
Access the Web Application:<br>
Open your browser and visit http://127.0.0.1:5000/.<br>

Select T-Shirts:<br>

Go to the Collection page.<br>
Choose a t-shirt from the list.<br>
The webcam feed will overlay the selected t-shirt onto your body.<br>
HTML and CSS Structure<br>
HTML Templates:<br>
The home.html and collection.html pages are built using Bootstrap for responsive design and Jinja2 for dynamic content rendering.<br>

CSS:<br>
Custom styling is added via CSS files inside the static/css folder. Bootstrap components (like buttons and cards) are used to enhance the UI.<br>


Contributing
Feel free to fork the repository and submit pull requests to enhance the project!<br>

License<br>
This project is licensed under the MIT License.<br>

Troubleshooting<br>
Webcam Not Detected:<br>
Ensure your webcam is enabled and accessible by the browser.<br>

T-Shirt Image Not Found Error:<br>
Verify that the t-shirt images are correctly placed in the static/ folder.<br>
