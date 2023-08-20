from flask import Flask, flash, request, redirect, url_for, render_template, session, make_response
from matplotlib.figure import Figure
import numpy as np
import re
import csv
from io import StringIO, BytesIO
import base64
import AirfoilCalculation
from flask_session import Session


app = Flask(__name__, template_folder='templates')
ALLOWED_EXTENSIONS = {'txt', 'dat'}
app.config['MAX_CONTENT_LENGTH'] = 10000
app.config['SECRET_KEY'] = "Very secret key"

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


def plot_airfoil(x: float, y: float, airfoil_name: str):
    """

    Generate a airfoil geometry shape and saves the graph using a memory buffer.

    """

    plot_name = "Geometry of a " + airfoil_name + " Afirfoil"
    width = 8
    fig = Figure(figsize=(width, width))
    ax = fig.subplots()
    ax.set_title(loc="center", label=plot_name)
    ax.plot(x, y, color="black", marker='o', linestyle='-', linewidth=2)
    ax.grid()
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.axis('scaled')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.2)
    buf = BytesIO()
    # writes plot to BytesIO class
    fig.savefig(buf, format='jpg', bbox_inches="tight")
    plotimg = base64.b64encode(buf.getbuffer()).decode(
        "ascii")  # decodes the stored plot into base64 format
    return plotimg


def plot_cp_surfaces(panels:np.ndarray, airfoil_name:str):
    """

    Generates a graph of pressure distribution on the surfaces of the airfoil

    """
    plot_name = "Pressure distribution on surfaces " + airfoil_name
    width = 8
    fig = Figure(figsize=(width, width*0.35), dpi=100)
    ax = fig.subplots()
    ax.set_title(loc="center", label=plot_name)
    ax.grid()
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$C_p$', fontsize=16)
    ax.plot([panel.xc for panel in panels if panel.loc == 'upper'],
            [panel.cp for panel in panels if panel.loc == 'upper'],
            label='upper surface',
            color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    ax.plot([panel.xc for panel in panels if panel.loc == 'lower'],
            [panel.cp for panel in panels if panel.loc == 'lower'],
            label='lower surface',
            color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    ax.legend(loc='best', prop={'size': 16})
    buf = BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches="tight")
    plotimg = base64.b64encode(buf.getbuffer()).decode("ascii")
    return plotimg


def plot_velocity_streamplot(panels:np.ndarray, freestream:AirfoilCalculation.Freestream, plot_name:str, u:np.ndarray, v:np.ndarray, x_mesh:np.ndarray, y_mesh:np.ndarray):
    """

    Generates a streamplot graph around the airfoil

    """

    width = 8
    fig = Figure(figsize=(width, width*0.35))
    ax = fig.subplots()
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.streamplot(x_mesh, y_mesh, u, v,
                  density=1, linewidth=1, arrowsize=1, arrowstyle='->')
    ax.fill([panel.xc for panel in panels],
            [panel.yc for panel in panels],
            color='k', linestyle='solid', linewidth=2, zorder=2)
    # ax.axis('scaled')
    ax.set_xlim(min(panel.xa for panel in panels)-1,
                max(panel.xa for panel in panels)+1)
    ax.set_ylim(min(panel.ya for panel in panels)-0.3,
                max(panel.ya for panel in panels)+0.3)
    ax.set_title(
        f'Streamlines around a {plot_name} airfoil (AoA = {np.round(np.degrees(freestream.alpha))})', fontsize=16)
    buf = BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches="tight")
    plotimg = base64.b64encode(buf.getbuffer()).decode("ascii")
    return plotimg


def plot_pressure_field(panels:np.ndarray, freestream:AirfoilCalculation.Freestream, plot_name:str, u:np.ndarray, v:np.ndarray, x_mesh:np.ndarray, y_mesh:np.ndarray):
    """

    Generates pressure distribution graph around airfoil

    """

    cp = 1.0 - (u**2 + v**2) / freestream.u_inf**2

    width = 8
    fig = Figure(figsize=(width, width*0.35))
    ax = fig.subplots()
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    contf = ax.contourf(x_mesh, y_mesh, cp,
                        levels=np.linspace(-2.0, 1.0, 100), extend='both')
    cbar = fig.colorbar(contf,
                        orientation='horizontal',
                        shrink=0.5, pad=0.1,
                        ticks=[-2.0, -1.0, 0.0, 1.0])
    cbar.set_label('$C_p$', fontsize=16)
    ax.fill([panel.xc for panel in panels],
            [panel.yc for panel in panels],
            color='k', linestyle='solid', linewidth=2, zorder=2)
    ax.axis('scaled')
    ax.set_xlim(min(panel.xa for panel in panels)-1,
                max(panel.xa for panel in panels)+1)
    ax.set_ylim(min(panel.ya for panel in panels)-0.3,
                max(panel.ya for panel in panels)+0.3)
    ax.set_title('Contour of pressure field around ' + plot_name, fontsize=16)
    buf = BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches="tight")
    plotimg = base64.b64encode(buf.getbuffer()).decode("ascii")
    return plotimg


def allowed_file(file_name: str):
    """

    Checks if the file format is correct

    """

    if not "." in file_name:
        return False
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_is_input_values_float(data: str):
    """

    Checks whether the data contains only integers or floating-point numbers, the characters "+" "-", "E" for engineering format and the new line character "/n".

    """

    data = data.splitlines()[1:]
    data = '\n'.join(data)
    return bool(re.match(r"^[0-9.\n -E+]+$", data))


def validate_form_data(data: str, max_value: float, form_name: str):
    """

    Checks that the data given in the forms is correct. Checks that the data provided contains only numbers or floating point numbers and that the data is not greater than the maximum value.

    """

    if (data is None):
        flash(f'Please fill in {form_name} form')
        return redirect(request.url)
    if not bool(re.match(r"^[0-9.]+$", data)):
        flash(f"Data in {form_name} form must be a possitive number")
        return redirect(request.url)
    data = float(data.replace(",", "."))
    if data > max_value:
        flash(f"Data in {form_name} form must not be greater than {max_value}")
        return redirect(request.url)
    return data


@ app.route('/', methods=['GET', 'POST'])
def index():
    """ Airfoil sent as a file """
    if request.method == 'POST' and 'SubmitAirfoilFile' in request.form:
        # Is the file selected
        airfoil_file = request.files['AirfoilData']
        if airfoil_file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Checks if the format is correct, and writes the file to RAM
        if airfoil_file and allowed_file(airfoil_file.filename):
            airfoil_file = airfoil_file.stream
            airfoil_file.seek(0)

        # Checks if the file has correct data
        if not check_is_input_values_float(airfoil_file.read().decode("utf-8")):
            flash('Please send data in selig format')
            return redirect(request.url)

        airfoil_file.seek(0)
        # Parses the file into variables
        try:
            x, y = np.loadtxt(airfoil_file, dtype=float,
                              unpack=True, skiprows=1)
        except:
            flash('Something went wrong, please try until my program work')
            return redirect(request.url)

        # Saves to cookies
        session["CoordinatesX"] = x.tolist()
        session["CoordinatesY"] = y.tolist()

        # File name used as profile name
        airfoil_name = airfoil_file.filename.rsplit('.', 1)[0]
        session["AirfoilName"] = airfoil_name
        return render_template('index.html', plotimg=plot_airfoil(x, y, airfoil_name), show_airfoil=True)

    ''' Profil wysyłany w formie tekstu '''
    if request.method == 'POST' and 'SubmitAirfoilText' in request.form:
        # Parses the sent text into the variables
        submitted_airfoil_data = request.form['TextCoords']

        if submitted_airfoil_data == '':
            flash('No data provided')
            return redirect(request.url)

        if not check_is_input_values_float(submitted_airfoil_data):
            flash('Please provide data in selig format')
            return redirect(request.url)

        try:
            x, y = np.loadtxt(StringIO(submitted_airfoil_data), dtype=float,
                              unpack=True, skiprows=1)
        except:
            flash('Something went wrong, please try until my program work')
            return redirect(request.url)

        # Assigns the first line of text to a variable as a profile name
        airfoil_name = submitted_airfoil_data.split(
            "\n")[0].lower().replace(" ", "").rstrip()

        session["CoordinatesX"] = x.tolist()
        session["CoordinatesY"] = y.tolist()
        session["AirfoilName"] = airfoil_name
        return render_template('index.html', plotimg=plot_airfoil(x, y, airfoil_name), show_airfoil=True)

    '''The "calculate" button was pressed'''
    if request.method == 'POST' and 'Calc' in request.form:
        if session.get("AirfoilName"):
            pan_num = int(validate_form_data(
                request.form['PanNum'], 300, 'Number of panels'))
            AoA = validate_form_data(
                request.form['AoA'], 30, "Angle of attack")
            freestream_velocity = validate_form_data(
                request.form['freestream_velocity'], 100, "Freestream velocity")
            freestream_density = validate_form_data(
                request.form['freestream_density'], 2000, "Freestream Density")
            try:
                # Parses coordinates from cookies into variables
                x, y = np.array(session["CoordinatesX"]), np.array(
                    session["CoordinatesY"])

                # Calculates parameters on panels
                panels, gamma, freestream = AirfoilCalculation.compute_airloil(
                    x, y, pan_num, freestream_velocity, AoA, freestream_density)
            except:
                flash('Something went wrong, please try until my program work')
                return redirect(request.url)

            # Calculates aerodynamic parameters of the profile
            accuracy, cl, cm, cd, chord, u, v, x_mesh, y_mesh, L, D = AirfoilCalculation.compute_results(
                panels, freestream, gamma)

            # Saves parameters to list and to cookies
            results_data = [[0 for j in range(pan_num)] for i in range(9)]
            for i, panel in enumerate(panels):
                results_data[0][i] = panel.xc
                results_data[1][i] = panel.yc
                results_data[2][i] = panel.cp
                results_data[3][i] = panel.cl
                results_data[4][i] = panel.cm
                results_data[5][i] = panel.cd
                results_data[6][i] = panel.cn
                results_data[7][i] = panel.ct
                results_data[8][i] = panel.loc
            results_data.append([cl, cm, cd])
            session.pop('results_data', None)
            session["results_data"] = results_data
            """
            0       1       2          3           4          5             6           7           8
            xc      yc      cp         cl          cm         cd         CN(CFX)      CA(CFY)    location
            """
            # Returns parameters and plots
            return render_template('index.html', plotimg=plot_airfoil(x, y, session["AirfoilName"]), plot_cp=plot_cp_surfaces(panels, session["AirfoilName"]), velocity_streamplot=plot_velocity_streamplot(panels, freestream, session["AirfoilName"], u, v, x_mesh, y_mesh), pressure_field=plot_pressure_field(panels, freestream, session["AirfoilName"], u, v, x_mesh, y_mesh), show_airfoilscarousel=True, chord=round(chord, 2), Cl=round(cl, 3), accuracy=round(accuracy, 6), Cm=round(cm, 5), Cd=round(cd, 5), L=round(L, 3), D=round(D, 5), LtoD=round(cl/cd, 4))
        else:
            flash('Upload your airfoil first')
            return redirect(request.url)

    '''The button "Download results" was pressed'''
    if request.method == 'POST' and 'DownloadResults' in request.form:
        if session.get("results_data"):
            # Transcribes from cookies to variable data
            csv_data = session["results_data"]
            str_io = StringIO()
            writer = csv.writer(str_io)
            # List of headings
            labels = ["xc", "yc", "cp", "cl",  "cm",
                      "cd", "CN", "CT", "location", "Total cl = ", "Total cm = ", "Total cd = ", ]

            # Writes the profile name to 1 line
            writer.writerow([(session["AirfoilName"])])
            # Saves the data and header to a CSV file with an accuracy of 3 clipping places, up to the 3rd line.
            # (xc, yc, cp)
            for i, row in enumerate(csv_data[:3]):
                row = [round(value, 3) for value in row]
                writer.writerow([labels[i]] + row)

            # Saves the data and header to a CSV file with an accuracy of 3 clipping places, from 4 to 7 lines
            # (cl, cm, cd CN CT)
            for i, row in enumerate(csv_data[3:8]):
                row = [round(value, 6) for value in row]
                writer.writerow([labels[i+3]] + row)

            # Saves the panel position and header to a CSV file until the last line
            for row in csv_data[8:9]:
                writer.writerow([labels[8]] + row)

            # Adds the resulting cl, cd, cm
            for i in range(9, 11+1):
                writer.writerow([
                    labels[i] + str(round(csv_data[9:][0][i-9], 6))])

            output = str_io.getvalue()
            # Creates a Flask response with content and appropriate headers. Name from cookies.
            response = make_response(output)
            response.headers[
                "Content-Disposition"] = f'attachment; filename={session["AirfoilName"]}.csv'
            response.headers["Content-type"] = "text/csv"

            return response
        else:
            flash('Calculate your airfoil first')
            return redirect(request.url)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
