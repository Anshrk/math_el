# Import necessary libraries
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set up page configuration
st.set_page_config(page_title="ODE System Solver", layout="wide")

# Navigation between pages
page = st.sidebar.selectbox("Select a page:", ["Collective Graph", "Individual Graphs"])

if page == "Collective Graph":
    # Page 1: Code Explanation
    def system(eta, y, params):
        f, f_prime, f_double_prime, F, F_prime, theta, theta_prime, theta_p, theta_p_prime = y
        l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T, S = params
        
        # Define the derivatives based on the provided equations
        d_f = f_prime
        d_f_prime = f_double_prime
        d_f_double_prime = -f * f_double_prime + (f_prime)**2 - l * (F_prime - f_prime) + S * f_prime - Q * np.exp(-A * eta)
        
        d_F = F_prime
        d_F_prime = (F**2 - beta * (F_prime - f_prime)) / F if F != 0 else 0  # Avoid division by zero
        
        d_theta = theta_prime
        d_theta_prime = (-Pr * (f * theta_prime - 2 * f_prime * theta) 
                        - Pr * l * beta_T * (theta_p - theta) 
                        - l * beta * Pr * Ec * (F_prime - f_prime)**2) / (1 + 4 * Rd / 3)
        
        d_theta_p = theta_p_prime
        d_theta_p_prime = (2 * F_prime * theta_p - beta_T * (theta_p - theta)) / F if F != 0 else 0  # Avoid division by zero
        
        return [d_f, d_f_prime, d_f_double_prime, d_F, d_F_prime, d_theta, d_theta_prime, d_theta_p, d_theta_p_prime]

    # Define initial conditions for the boundary conditions
    def boundary_conditions(params):
        l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T = params

        f0 = 0            # f(0) = 0
        f_prime0 = 1      # f'(0) = 1
        theta0 = 1        
        theta_prime0 = -Bi * (1 - theta0)  # θ'(0) = -Bi * (1 - θ(0))
        
        f_double_prime0 = 0
        F0 = 1e-6  
        F_prime0 = 0
        theta_p0 = 0
        theta_p_prime0 = 0

        return [f0, f_prime0, f_double_prime0, F0, F_prime0, theta0, theta_prime0, theta_p0, theta_p_prime0]

    # Set up Streamlit app
    st.title("ODE System Solver and Plotter")

    # Sidebar inputs for parameters
    st.sidebar.header("Set Parameters")
    l = st.sidebar.slider("l", 0.0, 2.0, 0.6)
    beta = st.sidebar.slider("beta", 0.0, 2.0, 0.5)
    Q = st.sidebar.slider("Q", 0.0, 2.0, 0.1)
    A = st.sidebar.slider("A", 0.0, 5.0, 2.0)
    Rd = st.sidebar.slider("Rd", 0.0, 5.0, 3.0)
    Pr = st.sidebar.slider("Pr", 0.0, 1.0, 0.1)
    Bi = st.sidebar.slider("Bi", 0.0, 5.0, 1.5)
    Ec = st.sidebar.slider("Ec", 0.0, 1.0, 0.5)
    beta_T = st.sidebar.slider("beta_T", 0.0, 1.0, 0.5)
    S = st.sidebar.slider("S", 0.0, 5.0, 2.0)

    params = (l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T, S)

    # Integration range for eta
    eta_span = (0, 5)

    # Initial conditions
    y0 = boundary_conditions(params[:-1])

    # Solve the ODE system
    solution = solve_ivp(system, eta_span, y0, args=(params,), method='RK45', dense_output=True, rtol=1e-6, atol=1e-9)

    # Extract eta and the solutions
    eta_vals = solution.t
    f_prime_vals = solution.y[1]
    F_prime_vals = solution.y[4]
    theta_vals = solution.y[5]
    theta_p_vals = solution.y[7]
    f_vals = -solution.y[0]
    F_vals = -solution.y[3]

    # Plot the solutions
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot with negative eta values for mirroring
    ax.plot(-eta_vals, f_prime_vals/100000, label="f'(η)", linestyle='-', color='blue')
    ax.plot(-eta_vals, F_prime_vals/100000, label="F'(η)", linestyle='-', color='orange')
    ax.plot(-eta_vals, -theta_vals/100000, label="Q'(η)", linestyle='-', color='green')
    ax.plot(-eta_vals, theta_p_vals/100000, label="Q_p'(η)", linestyle='-', color='red')
    ax.plot(-eta_vals, -f_vals/100000, label='f(η)', linestyle='-', color='black')
    ax.plot(-eta_vals, -F_vals/100000, label='F(η)', linestyle='-', color='yellow')
    ax.plot(-eta_vals, -theta_vals/100000, label='θ(η)', linestyle='-', color='pink')
    ax.plot(-eta_vals, theta_p_vals/100000, label='θ_p(η)', linestyle='-', color='purple')

    # Labels and title
    ax.set_xlabel('η')
    ax.set_ylabel('Values')
    ax.set_title("Plot of f'(η), F'(η), Q'(η), and Q_p'(η)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

elif page == "Individual Graphs":
    # Define the system of first-order ODEs
    def system(eta, y, params):
        f, f_prime, f_double_prime, F, F_prime, theta, theta_prime, theta_p, theta_p_prime = y
        l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T, S = params
        
        # Define the derivatives based on the provided equations
        d_f = f_prime
        d_f_prime = f_double_prime
        d_f_double_prime = -f * f_double_prime + (f_prime)**2 - l * (F_prime - f_prime) + S * f_prime - Q * np.exp(-A * eta)
        
        d_F = F_prime
        d_F_prime = (F**2 - beta * (F_prime - f_prime)) / F if F != 0 else 0  # Avoid division by zero
        
        d_theta = theta_prime
        d_theta_prime = (-Pr * (f * theta_prime - 2 * f_prime * theta) 
                        - Pr * l * beta_T * (theta_p - theta) 
                        - l * beta * Pr * Ec * (F_prime - f_prime)**2) / (1 + 4 * Rd / 3)
        
        d_theta_p = theta_p_prime
        d_theta_p_prime = (2 * F_prime * theta_p - beta_T * (theta_p - theta)) / F if F != 0 else 0  # Avoid division by zero
        
        return [d_f, d_f_prime, d_f_double_prime, d_F, d_F_prime, d_theta, d_theta_prime, d_theta_p, d_theta_p_prime]

    # Define initial conditions for the boundary conditions
    def boundary_conditions(params):
        l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T = params

        f0 = 0            # f(0) = 0
        f_prime0 = 1      # f'(0) = 1
        theta0 = 1        
        theta_prime0 = -Bi * (1 - theta0)  # θ'(0) = -Bi * (1 - θ(0))
        
        f_double_prime0 = 0
        F0 = 1e-6  
        F_prime0 = 0
        theta_p0 = 0
        theta_p_prime0 = 0

        return [f0, f_prime0, f_double_prime0, F0, F_prime0, theta0, theta_prime0, theta_p0, theta_p_prime0]

    # Set up Streamlit app
    st.title("ODE System Solver and Plotter")

    # Sidebar inputs for parameters
    st.sidebar.header("Set Parameters")
    l = st.sidebar.slider("l", 0.0, 2.0, 0.6)
    beta = st.sidebar.slider("beta", 0.0, 2.0, 0.5)
    Q = st.sidebar.slider("Q", 0.0, 2.0, 0.1)
    A = st.sidebar.slider("A", 0.0, 5.0, 2.0)
    Rd = st.sidebar.slider("Rd", 0.0, 5.0, 3.0)
    Pr = st.sidebar.slider("Pr", 0.0, 1.0, 0.1)
    Bi = st.sidebar.slider("Bi", 0.0, 5.0, 1.5)
    Ec = st.sidebar.slider("Ec", 0.0, 1.0, 0.5)
    beta_T = st.sidebar.slider("beta_T", 0.0, 1.0, 0.5)
    S = st.sidebar.slider("S", 0.0, 5.0, 2.0)

    params = (l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T, S)

    # Integration range for eta
    eta_span = (0, 5)

    # Initial conditions
    y0 = boundary_conditions(params[:-1])

    # Solve the ODE system
    solution = solve_ivp(system, eta_span, y0, args=(params,), method='RK45', dense_output=True, rtol=1e-6, atol=1e-9)

    # Extract eta and the solutions
    eta_vals = solution.t
    f_prime_vals = solution.y[1]
    F_prime_vals = solution.y[4]
    theta_vals = solution.y[5]
    theta_p_vals = solution.y[7]
    f_vals = -solution.y[0]
    F_vals = -solution.y[3]

    # Plot the solutions
    fig, ax = plt.subplots(figsize=(14, 8))
    # Plot with negative eta values for mirroring
    ax.plot(-eta_vals, f_prime_vals/100000, label="f'(η)", linestyle='-', color='blue')

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(-eta_vals, F_prime_vals/100000, label="F'(η)", linestyle='-', color='orange')

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.plot(-eta_vals, -theta_vals/100000, label="Q'(η)", linestyle='-', color='green')

    fig3, ax3 = plt.subplots(figsize=(14, 8))
    ax3.plot(-eta_vals, theta_p_vals/100000, label="Q_p'(η)", linestyle='-', color='red')

    fig4, ax4 = plt.subplots(figsize=(14, 8))
    ax4.plot(-eta_vals, -f_vals/100000, label='f(η)', linestyle='-', color='black')

    fig5, ax5 = plt.subplots(figsize=(14, 8))
    ax5.plot(-eta_vals, -F_vals/100000, label='F(η)', linestyle='-', color='yellow')

    fig6, ax6 = plt.subplots(figsize=(14, 8))
    ax6.plot(-eta_vals, -theta_vals/100000, label='θ(η)', linestyle='-', color='pink')

    fig7, ax7 = plt.subplots(figsize=(14, 8))
    ax7.plot(-eta_vals, theta_p_vals/100000, label='θ_p(η)', linestyle='-', color='purple')


    # Labels and title
    ax.set_xlabel('η')
    ax.set_ylabel('Values')
    ax.set_title("Plot of f'(η)")
    ax.grid(True)
    ax.legend()

    # Labels and title
    ax1.set_xlabel('η')
    ax1.set_ylabel('Values')
    ax1.set_title("Plot of F'(η)")
    ax1.grid(True)
    ax1.legend()
    # Labels and title
    ax2.set_xlabel('η')
    ax2.set_ylabel('Values')
    ax2.set_title("Plot of Q'(η)")
    ax2.grid(True)
    ax2.legend()
    # Labels and title
    ax3.set_xlabel('η')
    ax3.set_ylabel('Values')
    ax3.set_title("Plot of Q_p'(η)")
    ax3.grid(True)
    ax3.legend()
    # Labels and title
    ax4.set_xlabel('η')
    ax4.set_ylabel('Values')
    ax4.set_title("Plot of f(η)")
    ax4.grid(True)
    ax4.legend()
    # Labels and title
    ax5.set_xlabel('η')
    ax5.set_ylabel('Values')
    ax5.set_title("Plot of F(η)")
    ax5.grid(True)
    ax5.legend()
    # Labels and title
    ax6.set_xlabel('η')
    ax6.set_ylabel('Values')
    ax6.set_title("Plot of θ(η)")
    ax6.grid(True)
    ax6.legend()
    # Labels and title
    ax7.set_xlabel('η')
    ax7.set_ylabel('Values')
    ax7.set_title("Plot of θ_p(η)")
    ax7.grid(True)
    ax7.legend()

    st.pyplot(fig)
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)
    st.pyplot(fig7)


import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

