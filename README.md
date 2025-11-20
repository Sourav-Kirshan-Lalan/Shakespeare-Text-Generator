# 🧙‍♂️ Shakespeare Text Generator with LSTM

Generate Shakespeare-style text using an LSTM model trained on historical literary data. This web app allows users to enter a seed text and control the creativity and length of the generated output.

## 🚀 Live Demo
[Deployed on Azure](https://shakespeare-eec4fkfha2bhf6am.southeastasia-01.azurewebsites.net/)

---

## 💡 Features

*  **LSTM model** trained on Shakespearean data
*  **Adjustable creativity** via temperature parameter
*  **User-defined seed text** input
*  **Configurable output length**
*  **Web interface** powered by Flask
*  **Cloud deployment** on Azure Container Instances

---

## Technologies Used

* **Python **
* **Flask** – Web framework
* **TensorFlow / Keras** – Deep learning
* **HTML + CSS** – Frontend
* **Docker** – Containerization
* **Azure Container Instances** – Cloud hosting

---

## How It Works

The LSTM model predicts the next character based on a sequence of previous characters.

### You can control:
* **seed** → starting text
* **temperature** → randomness (higher = more creative)
* **length** → number of generated characters

---

## ✍️ Example Output

**Seed:**  
`to be or not to be that is the`

**Generated:**  
```
to be or not to be that is the night when virtue walks in fair disguise and prince of denmark speaks again beneath the blood of broken skies
```

---

## 🐳 Deployment on Azure

This application is containerized using Docker and deployed on **Azure Container Instances** for scalable and reliable hosting.

### Deployment Steps:
1. Build the Docker image
2. Push to Azure Container Registry (ACR)
3. Deploy to Azure Container Instances
4. Access via public IP or custom domain

---

