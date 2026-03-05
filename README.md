# QCES Climate Data: Data Assimilation

This repository contains the lecture and practical material for the **Data Assimilation** part of the QCES Climate Data course. The course introduces Bayesian methods, Kalman filters, and ensemble techniques using a single and double pendulums as "toy problems" to illustrate
 complex climate science concepts.


## Running in Google Colab

The easiest way to run these notebooks is via Google Colab. 


| Notebook | Run in Colab |
| :--- | :--- |
| **Lecture 1** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/QCES_data_assimilation/blob/main/lectures/lecture1.ipynb) |
| **Lecture 2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/QCES_data_assimilation/blob/main/lectures/lecture2.ipynb) |
| **Practical 1** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/QCES_data_assimilation/blob/main/practicals/practical1.ipynb) |
| **Practical 2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/QCES_data_assimilation/blob/main/practicals/practical2.ipynb) |



> **How to save your work:** > When you open these links, you are in a "playground" mode. To save your own edits and solutions, go to **File > Save a copy in Drive**. This will create a copy in your personal Google Drive that you can edit and save indefinitely.

--- 

## Local Installation

If you prefer to work locally, this project uses **Poetry** for dependency management.

### 1. Install Poetry
If you do not have Poetry installed, follow the [official documentation](https://python-poetry.org/docs/#installation).

### 2. Clone and Setup
Clone the repository and install the environment:
```bash
git clone https://github.com/da380/QCES_data_assimilation.git
cd data_assimilation
poetry install
poetry run jupyter notebook
```

---


## Assessment: Practical Report

This module is assessed via a single written report based on the work you complete in Practical 1 and Practical 2. The report should summarise the key concepts of data assimilation you have explored, supported by the results and visualizations generated from your code.

### **Report Guidelines**
* **Format:** The report should be written in the style of a technical report or short journal article. It should be accessible to someone with general knowledge of climate science and mathematics but who is not familiar with this specific project.
* **Length:** Approximately 4-5 pages (single-spaced), including figures.
* **Content:** Your report does *not* need to be a step-by-step account of the practicals. Instead, focus on synthesising your findings to tell a coherent story about the methods, why they are needed, what there strengths and weaknesses are, and what challenges remain.

### **Key Topics to Address**
Your report should cover the following core areas:

1.  **The full Bayesian Approach (Single Pendulum):**
    * Explain the "Forecast-Analysis" cycle.
    * **Sensitivity to the Prior:** Discuss and demonstrate how the choice of the initial prior distribution affects the assimilation results. Does the solution converge to the truth regardless of the initial guess given enough observations?
    * **Reanalysis:** Show how using future data improves the estimate of past states.

2.  **The Ensemble Kalman Filter (Double Pendulum):**
    * Explain why ensemble methods are necessary for non-linear, high-dimensional systems (chaos).
    * **Forecast Skill:** Analyze the "predictability horizon" of the double pendulum. How long does your forecast remain reliable after the last observation? How does this depend on the specific initial conditions (e.g., low energy vs. high energy/chaotic regimes)?    


### **Assessment Criteria**
* **Clarity of Presentation:** Quality of writing and effectiveness of figures.
* **Quality of Analysis:** Is the interpretation of the results sound? Are the conclusions supported by the numerical results?
* **Conceptual Understanding:** Does the report demonstrate a solid grasp of Bayesian inference, chaos, and the differences between grid-based and ensemble methods?
