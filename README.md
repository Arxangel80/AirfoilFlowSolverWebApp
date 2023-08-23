# AirfoilFlowSolverWebApp
This project is a simple web application that provides a user-friendly interface for simulating the aerodynamic behavior of 2D airfoils. Users can input airfoil coordinates, set flow parameters, and visualize aerodynamic characteristics through generated graphs. The application is built using Flask, Python, JavaScript, jQuery, CSS, and HTML. The flow simulation within the web application is based on the vortex-source panel method, which discretizes the airfoil surface into panels and models the flow using a combination of vortex and source elements. While more sophisticated methods exist, this approach strikes a balance between accuracy and simplicity, making it suitable for initial analysis and learning purposes. Users can gain insights into aerodynamics by observing lift, drag, and other coefficients computed from the panel strengths, helping them understand the fundamentals of flow simulation and its applications in 2D airfoil analysis.

## How to run
```
pip install -r requirements.txt
python app.py
```

## Acknowledgments
This project was inspired by the learning material from the AeroPython repository. A big thank you to the Barba Group for providing such great educational resources.
