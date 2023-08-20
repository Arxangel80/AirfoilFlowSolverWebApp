import os
import math
import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt


def openFile(filepath: str):
    """
    Opens a text file and parses to x, y variables.

    filepath: string
        Path to the naca file containing the profile coordinates.

    Returns
    -------
    x: 1D floating point number matrix
        Matrix of x-coordinates defining the geometry of the profile.
    y: 1D floating-point number matrix
        Matrix of y-coordinates defining the geometry of the profile.
    """

    np.set_printoptions(suppress=True)
    with open(filepath, 'r') as file_name:
        xz, yz = np.loadtxt(file_name, dtype=float, unpack=True, skiprows=1)
    return xz, yz


class Panel:
    """
    Contains information related to a panel.

    """

    def __init__(self, xa: float, ya: float, xb: float, yb: float):
        """
        Initializes the panel.

        Sets the end-points and calculates the center-point, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is located on the upper or lower surface of the geometry.
        Initializes the source-strength, tangential velocity, and pressure coefficient
        of the panel to zero.

        Parameters
        ---------_
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # panel center
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length

        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)

        # panel location with respect to the x-axis
        if self.beta <= np.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

        self.sigma = 0.0  # source strength
        self.vt = 0.0  # Tangential velocity
        self.cp = 0.0  # Pressure coefficient

        self.cn = 0.0  # Pressure normal vector
        self.ct = 0.0  # Pressure tangencial vector

        self.cl = 0.0  # Lift coefficient
        self.cd = 0.0  # Drag coefficient
        self.cm = 0.0  # Moment coefficient


def define_panels(x: np.ndarray, y: np.ndarray, N: int = 40) -> np.ndarray:
    """
    Discretizes the geometry into panels using 'cosine' method.

    Parameters
    ----------
    x: 1D array of floats
        x-coordinate of the points defining the geometry.
    y: 1D array of floats
        y-coordinate of the points defining the geometry.
    N: integer, optional
        Number of panels;
        default: 40.

    Returns
    -------
    panels: 1D np array of Panel objects.
        The list of panels.
    """

    R = (x.max() - x.min()) / 2.0  # circle radius
    x_center = (x.max() + x.min()) / 2.0  # x-coordinate of circle center

    # vector of floating point values
    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)
    # x-coordinates of the circle described on the profile
    x_circle = x_center + R * np.cos(theta)

    x_ends = np.copy(x_circle)  # x-coordinate of the end points of the panels
    # y-coordinate of the end points of the panels
    y_ends = np.empty_like(x_ends)

    # extend coordinates to consider closed surface
    x, y = np.append(x, x[0]), np.append(y, y[0])

    # calculates the y-coordinate of the endpoints by projecting the described circle  to the profile
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            # Checks if the point x_ends[i] lies between the next and previous point
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        # Creates the equation of a linear function where a - directional coefficient, b - free expression
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    # Saves the parameters of the class Panel
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels


class Freestream:
    """
    Freestream conditions.

    """

    def __init__(self, u_inf: float = 1.0, alpha: float = 0.0, rho: float = 1.225):
        """
        Sets speed, angle of attack (given in degrees and converted to radians) and free flow density.

        Parameters
        ----------
        u_inf: float, optional
            Freestream speed.
            default: 1.0.
        alpha: float, optional
            Angle of attack in degrees.
            default 0.0.
        rho: float, optional
            Freestream density.
            default 1.225.
        """

        self.u_inf = u_inf
        self.alpha = np.radians(alpha)  # converts radians into degrees
        self.rho = rho


def integral(x: float, y: float, panel: Panel, dxdk: float, dydk: float) -> float:
    """
    Evaluates the contribution from a panel at a given point.

    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Panel whose contribution is evaluated.
    dxdk: float
        Value of the derivative of x in a certain direction.
    dydk: float
        Value of the derivative of y in a certain direction.

    Returns
    -------
    Contribution from the panel at a given point (x, y).
    """
    def integrand(s):
        Term_1 = x - (panel.xa - np.sin(panel.beta)*s)
        Term_2 = y - (panel.ya + np.cos(panel.beta)*s)
        return (Term_1 * dxdk + Term_2 * dydk)/(Term_1**2 + Term_2**2)
    return integrate.quad(integrand, 0.0, panel.length)[0]


def source_contribution_normal(panels: np.ndarray) -> np.ndarray:
    """
    Builds the source contribution matrix for the normal velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.

    Returns
    -------
    A: 2D np array of floats
        Source contribution matrix.

    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # fill the main diagonal of the matrix with the number 0.5, for estimating the share of own source per calculated panel
    np.fill_diagonal(A, 0.5)
    # contribution of sources of other panels per calculated panel
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(panel_i.xc, panel_i.yc,
                                                 panel_j,
                                                 np.cos(panel_i.beta),
                                                 np.sin(panel_i.beta))
    return A


def vortex_contribution_normal(panels: np.ndarray) -> np.ndarray:
    """
    Builds the vortex contribution matrix for the normal velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.

    Returns
    -------
    A: 2D np array of floats
        Vortex contribution matrix.
    """

    # vortex contribution on a panel from itself
    A = np.zeros((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / np.pi * integral(panel_i.xc, panel_i.yc,
                                                  panel_j,
                                                  np.sin(panel_i.beta),
                                                  -np.cos(panel_i.beta))
    return A


def kutta_condition(A_source: np.ndarray, B_vortex: np.ndarray) -> np.ndarray:
    """
    Builds the Kutta condition array.

    Parameters
    ----------
    A_source: 2D np array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D np array of floats
        Vortex contribution matrix for the normal velocity.

    Returns
    -------
    b: 1D np array of floats
        The left-hand side of the Kutta-condition equation.
    """

    b = np.empty(A_source.shape[0] + 1, dtype=float)
    # matrix of the source contribution to tangential velocity is the same as the matrix of the vortex contribution to normal velocity
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    # vortex contribution matrix at tangential velocity is the opposite of the source contribution matrix at normal velocity
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    return b


def build_singularity_matrix(A_source: np.ndarray, B_vortex: np.ndarray) -> np.ndarray:
    """
    Builds the left-hand side matrix of the system arising from source and vortex contributions.

    Parameters
    ----------
    A_source: 2D np array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D np array of floats
        Vortex contribution matrix for the normal velocity.

    Returns
    -------
    A:  2D np array of floats
        Matrix of the linear system.
    """
    A = np.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)

    # source contribution matrix
    A[:-1, :-1] = A_source

    # vortex contribution array
    A[:-1, -1] = np.sum(B_vortex, axis=1)

    # Matrix taking into account Kutta's condition
    A[-1, :] = kutta_condition(A_source, B_vortex)
    return A


def build_freestream_rhs(panels: np.ndarray, freestream: Freestream) -> np.ndarray:
    """
    Builds the matrix of the right-hand side of equation (b) resulting from the free-flow contribution.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.

    Returns
    -------
    b: 1D np array of floats
        Freestream contribution on each panel and on the Kutta condition.
    """
    b = np.empty(panels.size + 1, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.alpha - panel.beta)
    # contribution of free flow near the edge of the wing (Kutta condition)
    b[-1] = -freestream.u_inf * (np.sin(freestream.alpha - panels[0].beta) +
                                 np.sin(freestream.alpha - panels[-1].beta))
    return b


def compute_tangential_velocity(panels: np.ndarray, freestream: Freestream, delta_gamma: float, A_source: np.ndarray, B_vortex: np.ndarray):
    """
    Computes the tangential surface velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    gamma: float
        Circulation density.
    A_source: 2D np array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D np array of floats
        Vortex contribution matrix for the normal velocity.
    """

    A = np.empty((panels.size, panels.size + 1), dtype=float)
    # matrix of the source contribution to tangential velocity is the same as the matrix of the vortex contribution to normal velocity
    A[:, :-1] = B_vortex
    # vortex influence matrix for tangential velocity is the opposite of the source influence matrix for normal velocity
    A[:, -1] = -np.sum(A_source, axis=1)
    # freestream contribution
    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta
                                   for panel in panels])

    strengths = np.append([panel.sigma for panel in panels], delta_gamma)

    tangential_velocities = np.dot(A, strengths) + b

    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]


def compute_pressure_coefficient(panels: np.ndarray, freestream: Freestream):
    """
    Computes the surface pressure coefficients.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2


def get_velocity_field(panels: np.ndarray, freestream: Freestream, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the velocity field on a given 2D mesh.

    Parameters
    ---------
    panels: 1D array of Panel objects
        The source panels.
    freestream: Freestream object
        The freestream conditions.
    X: 2D Numpy array of floats
        x-coordinates of the mesh points.
    Y: 2D Numpy array of floats
        y-coordinate of the mesh points.

    Returns
    -------
    u: 2D Numpy array of floats
        x-component of the velocity vector field.
    v: 2D Numpy array of floats
        y-component of the velocity vector field.
    """
    # freestream contribution
    u = freestream.u_inf * \
        math.cos(freestream.alpha) * np.ones_like(X, dtype=float)
    v = freestream.u_inf * \
        math.sin(freestream.alpha) * np.ones_like(X, dtype=float)
    # adding contribution of each sources (superposition rule)
    vec_intregral = np.vectorize(integral)
    for panel in panels:
        u += panel.sigma / (2.0 * math.pi) * \
            vec_intregral(X, Y, panel, 1.0, 0.0)
        v += panel.sigma / (2.0 * math.pi) * \
            vec_intregral(X, Y, panel, 0.0, 1.0)

    return u, v


def compute_airloil(x: np.ndarray, y: np.ndarray, N: int = 40, u_inf: float = 1, alpha: int = 0, density: float = 1.225):
    """
    Calculates all necessary calculation parameters.

    Parameters
    ---------
    x: 1D numpy array of floating point numbers
        The x-coordinates of the grid.
    y: 1D numpy array of floating point numbers
        The y-coordinates of the grid.
    N: integer
        Number of panels.
    u_inf: floating point number
        Free flow speed.
    alpha: floating point number
        Airfoil angle of attack.
    density: floating point number
        Flow density.

    Returns
    -------
    panels: 1D numpy array of objects of class Panel
        List of panel objects.
    delta_gamma: floating point number
        Circulation density.
    freestream: Objekt klasy Freestream
        Freestream parameters.

    """
    panels = define_panels(x, y, N)
    freestream = Freestream(u_inf, alpha, density)

    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)

    A = build_singularity_matrix(A_source, B_vortex)
    b = build_freestream_rhs(panels, freestream)

    strengths = np.linalg.solve(A, b)
    for i, panel in enumerate(panels):
        panel.sigma = strengths[i]

    # Circulation density, considered as the delta of the circulation
    delta_gamma = strengths[-1]

    compute_tangential_velocity(
        panels, freestream, delta_gamma, A_source, B_vortex)
    compute_pressure_coefficient(panels, freestream)

    return panels, delta_gamma, freestream


def compute_results(panels: np.ndarray, freestream: Freestream, delta_gamma: float):
    """

    Calculates profile parameters.

    Parameters
    ---------
    panels: 1D numpy array of objects of class Panel
        List of panel objects.
    freestream: Freestream object
        The freestream conditions.
    delta_gamma: floating point number
        Circulation density.

    """

    accuracy = sum([panel.sigma * panel.length for panel in panels])

    chord = abs(max(panel.xa for panel in panels) -
                min(panel.xa for panel in panels))

    for panel in panels:
        panel.cn = (-panel.cp*panel.length*math.sin(panel.beta))
        panel.ct = (-panel.cp*panel.length*math.cos(panel.beta))
        panel.cl = (delta_gamma*panel.length) / \
            (0.5 * freestream.u_inf * chord)
        panel.cd = (panel.cn*math.sin(freestream.alpha)) - \
            (panel.ct*math.cos(freestream.alpha))
        panel.cm = panel.cp*(panel.xc-0.25)*panel.length*np.cos(panel.beta)

    cl, cd, cm = sum([panel.cl for panel in panels]), sum([panel.cd for panel in panels]), sum(
        [panel.cm for panel in panels])

    # Lift and drag force on a 1m wing
    L = 0.5*freestream.rho*freestream.u_inf**2*chord*cl
    D = 0.5*freestream.rho*freestream.u_inf**2*chord*cd

    nx, ny = 20, 20  # number of points in the x and y directions
    x_start, x_end = chord-2, chord+1
    y_start, y_end = min(panel.ya for panel in panels) - \
        0.3, max(panel.ya for panel in panels)+0.3
    x_mesh, y_mesh = np.meshgrid(np.linspace(x_start, x_end, nx),
                                 np.linspace(y_start, y_end, ny))
    u, v = get_velocity_field(panels, freestream, x_mesh, y_mesh)
    return accuracy, cl, cm, cd, chord, u, v, x_mesh, y_mesh, L, D


def plot_results(panels: np.ndarray, u: np.ndarray, v: np.ndarray, x_mesh: np.ndarray, y_mesh: np.ndarray):
    """
    Creates and draw graphs.

    Parameters
    ---------
    panels: 1D numpy array of objects of class Panel
        List of panel objects.
    u: 2D Numpy array of floats
        x-component of the velocity vector field.
    v: 2D Numpy array of floats
        y-component of the velocity vector field.
    x_mesh: 2D Numpy array of floats
        Mesh grid of vectors.
    y_mesh: 2D Numpy array of floats
        Mesh grid of vectors.
    """

    # Panels chart on airfoil
    width = 10
    plt.figure(figsize=(width, width))
    plt.grid()
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.plot(x, y, color='k', linestyle='-', linewidth=2)
    plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
             np.append([panel.ya for panel in panels], panels[0].ya),
             linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    plt.axis('scaled')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 0.1)

    # Generates a plot of the pressure distribution around the airfoil
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$C_p$', fontsize=16)
    plt.plot([panel.xc for panel in panels if panel.loc == 'upper'],
             [panel.cp for panel in panels if panel.loc == 'upper'],
             label='upper surface',
             color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    plt.plot([panel.xc for panel in panels if panel.loc == 'lower'],
             [panel.cp for panel in panels if panel.loc == 'lower'],
             label='lower surface',
             color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    plt.legend(loc='best', prop={'size': 16})
    plt.xlim(-0.1, 1.1)
    plt.ylim(1.0, -2.0)
    plt.title('Number of panels: {}'.format(panels.size), fontsize=16)

    # Generates a plot of flow lines around an airfoil
    width = 10
    plt.figure(figsize=(width, width))
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.streamplot(x_mesh, y_mesh, u, v,
                   density=1, linewidth=1, arrowsize=1, arrowstyle='->')
    plt.fill([panel.xc for panel in panels],
             [panel.yc for panel in panels],
             color='k', linestyle='solid', linewidth=2, zorder=2)
    plt.axis('scaled')
    plt.xlim(min(panel.xa for panel in panels).any()-1,
             max(panel.xa for panel in panels).any()+1)
    plt.ylim(min(panel.ya for panel in panels).any()-0.3,
             min(panel.ya for panel in panels).any()+0.3)
    plt.title(f'Streamlines around a NACA 0012 airfoil (AoA = {freestream.alpha}^o$)',
              fontsize=16)


if __name__ == "__main__":
    """ Running the program in the console """

    filepath = input('Enter path to naca.dat file:')
    N = int(input('Enter the number of panels:'))
    alpha = float(input('Enter angle of attack:'))
    u_inf = float(input('Enter flow velocity:'))
    density = float(input('Enter flow density:'))
    x, y = openFile(filepath)
    width = 10

    plt.figure(figsize=(width, width))
    plt.grid()
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.plot(x, y, color='k', marker='o', linestyle='-', linewidth=2)
    plt.axis('scaled')

    panels, delta_gamma, freestream = compute_airloil(
        x, y, N=N, u_inf=u_inf, alpha=alpha, density=density)
    accuracy, cl, cm, cd, chord, u, v, x_mesh, y_mesh, L, D = compute_results(
        panels, freestream, delta_gamma)

    print('Dokładność = : {:0.6f}'.format(accuracy))
    print('Współczynnik siły nośnej: CL = {:0.3f}'.format(cl))
    print("Cięciwa profilu c = {:0.2f}".format(chord))
    print("Cd = ", cd)
    print("Cm = ", cm)
    print("Sila nosna L =", L)
    plot_results(panels, u, v, x_mesh, y_mesh)
