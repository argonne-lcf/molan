# Melting Point Insights and Prediction from a Diversified Machine Learning Strategy

![](media/banner.png)





Intro statement

## How do I run this?
There are two options:
* **Locally**

This requires a standard scientific Python 3 environment with rdkit and tensorflow+pytorch and a cloned github.
A simple way of getting that is installing [Anaconda](https://www.anaconda.com/distribution/#download-section).

First to clone the github and then replicate a new anaconda environment using the **environment.yml** file:
```
git clone https://github.com/beangoben/melting_points_ml
cd melting_points_ml
conda env create -f environment.yml
```

* **Remotely** via Google Colab

Visit [google colab](https://colab.research.google.com/) (requires a gdrive account) and open a colab notebook via github:

![](media/colab_menu.png)

## What's inside?

* **data**
  - 47K: Folder of json, each containing information for one molecule.
  - \*csv: csv files.
* **notebooks**: Jupyter notebooks (run these!)
  - Exploratory_Data_Analysis.ipynb
  - semisupervised_VAE.ipynb
  - Graph_Neural_Networks.ipynb
  - Gaussian_Processes.ipynb
* **code**: Repo specific modules for training and creating the models.
* **results**: Figures and weights for models.
* **media**: Assorted images.
