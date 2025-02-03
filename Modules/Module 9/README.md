
# **PINN for Seismic Wave Propagation**

## **Overview**
This submission presents a **Physics-Informed Neural Network (PINN)** to model **seismic wave propagation** in a planetary body. The model is trained to predict the scalar potential **Φ** of P-Waves and S-Waves based on spatial and temporal inputs.

This implementation is part of a competition submission, and all necessary files are provided for model evaluation and visualization.

---

## **Requirements**
Ensure the following dependencies are installed before running the notebooks:

```bash
pip install torch numpy pandas matplotlib opencv-python
```

Alternatively, the notebooks can be run in **Google Colab**, where all dependencies can be installed as required.

---

## **Files Overview**
### **Notebooks**
- 📜 **`astromodule9_P_Wave.ipynb`** – Contains the trained PINN for **P-Wave propagation**.
- 📜 **`astromodule9_S_Wave.ipynb`** – Contains the trained PINN for **S-Wave propagation**.

### **Pre-trained Models**
- 🔍 **`phimodelp.pth`** – Pre-trained model for **P-Wave prediction**.
- 🔍 **`phimodels2.pth`** – Pre-trained model for **S-Wave prediction**.

### **Data**
- 𝄜 **`model.csv`** - Dataset for our model
- Source - Use Model_1.txt from LSL_Models from https://dataverse.ipgp.fr/dataset.xhtml?persistentId=doi:10.18715/IPGP.2023.llxn7e6d
          and converted to csv
---

## **Running the Evaluation**
To evaluate the trained models and visualize results, follow these steps:

### **Option 1: Using the Pre-trained Models**
If you want to use the pre-trained models for visualization and analysis **(without re-training)**:

1. **Open the respective notebook** in Jupyter Notebook or Google Colab.
2. **Load the trained model by running the following cell**:
   ```python
   import torch
   phimodel = torch.load("phimodelp.pth")  # Load P-Wave Model
   # OR
   phimodel = torch.load("phimodels2.pth")  # Load S-Wave Model
   phimodel.eval()
   ```
3. **Run only the required visualization cells**:
   - To generate a **wave propagation video**, run the section that iterates through time steps and generates frames.
   - To plot **shadow zones**, execute:
     ```python
     plt.plot(t)
     plt.show()
     ```

This allows you to **directly use the trained models** without running the full notebook.

---

### **Option 2: Training a New Model**
If you wish to **train the model from scratch**, run the entire notebook:

1. **Run all preprocessing and data preparation cells**.
2. **Train the model** by executing:
   ```python
   for epoch in range(900):
       phimodel.zero_grad()
       loss_eq = eqloss(x, y, t, vpall)  # Compute physics loss
       loss_boundary = boundaryloss(xo, yo, to, vpo)  # Compute boundary loss
       loss = loss_eq + loss_boundary
       loss.backward()
       optimizer.step()
   ```
3. **Save the newly trained model**:
   ```python
   torch.save(phimodel, "new_model.pth")
   ```

This process ensures reproducibility and allows independent verification of results.

---

## **Model Physics and Source Function**
The model is governed by the **wave equation**:

```
Φ_tt - Vp^2 (Φ_xx + Φ_yy) = S
```

where **S(x,y,t)** represents the **epicenter source function**:

```
S(x, y, t) = A * exp(-α (t - t0)^2) * exp(-β ((x-x0)^2 + (y-y0)^2))
```

This ensures that the wave propagation follows **physical constraints** as defined in the competition.

---




## **Final Notes**
- This notebook and model are structured for **competition evaluation**.
- The provided pre-trained models ensure quick verification of results.
- Retraining should only be done if explicitly required as per the competition guidelines.

For any queries related to the evaluation process, refer to the **notebooks** and their respective comments.

