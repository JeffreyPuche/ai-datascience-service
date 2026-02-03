# ⚡ DATA SCIENCE SERVICES ⚡
<!-- Title Banner with Neon Style Badges -->
<div align="center">

![DevOps](https://img.shields.io/badge/FOCUS-DEVOPS-00ffff?style=for-the-badge&logo=azure-devops&logoColor=black)
![MLOps](https://img.shields.io/badge/LIFECYCLE-MLOPS-ff00ff?style=for-the-badge&logo=dvc&logoColor=white)
![Python](https://img.shields.io/badge/CODE-PYTHON-ffe100?style=for-the-badge&logo=python&logoColor=black)
![Docker](https://img.shields.io/badge/SHIP-DOCKER-0099ff?style=for-the-badge&logo=docker&logoColor=white)

<br>

**Una implementación de referencia para la Ingeniería de Datos y Machine Learning en el Mundo Real.**

[Explorar Proyectos](#-proyectos-python) • [Arquitectura](#-arquitectura-del-repositorio) • [Tecnologías](#-stack-tecnológico)

---
</div>

## 🔮 La Visión
> *"La diferencia entre un notebook y un producto es la ingeniería."*

Este repositorio no es solo una colección de scripts; es una **demostración viva** de cómo estructurar proyectos de Ciencia de Datos siguiendo los más altos estándares de la industria. Aquí rompemos la barrera entre el análisis exploratorio y el software de producción.

El objetivo es mostrar el **Ciclo Completo de Desarrollo (CI/CD)**, integrando prácticas de **MLOps** para garantizar que los modelos no solo funcionen en una máquina local, sino que escalen y sirvan valor en el mundo real.

---

## 🧬 Arquitectura del Repositorio

La estructura ha sido diseñada modularmente para separar responsabilidades (Data, Research, Infraestructura, Code).

```mermaid
graph TD;
    Root[data_science_services] --> Data(datasets 🗄️);
    Root --> Images(container-images 🐳);
    Root --> Analysis(notebooks-analysis 🔬);
    Root --> Code(python-projects 🐍);
    
    Data --> DVC[DVC Files -> S3/DagsHub];
    Images --> Prod[Production Images];
    Analysis --> PDF[Reportes & PDFs];
    Code --> APIs[FastAPI / Training Pipelines];
```

### 📂 Desglose de Directorios

#### 1. `datasets/` 🗄️
**"La Fuente de la Verdad."**
Aquí no encontrarás gigabytes de CSVs crudos. Este directorio actúa como un índice inteligente.
*   **Gestión con DVC (Data Version Control):** Almacenamos archivos `.dvc` (metadatos) que apuntan a nuestro almacenamiento remoto (S3, DagsHub, Azure Blob).
*   **Descarga Eficiente:** Permite al equipo descargar solo la versión exacta de los datos necesaria para reproducir un experimento específico.

#### 2. `container-images/` 🐳
**"Listos para el Despegue."**
Contiene las definiciones de infraestructura inmutable.
*   Aquí residen los `Dockerfiles` base y configuraciones optimizadas para entornos de producción.
*   Garantiza que "funciona en mi máquina" signifique "funciona en producción".

#### 3. `notebooks-analysis/` 🔬
**"El Laboratorio de Ideas."**
El espacio para la creatividad y la exploración estadística.
*   Contiene **Jupyter Notebooks** para EDA (Exploratory Data Analysis) y prototipado rápido.
*   Incluye versiones en **PDF** de los análisis para facilitar la lectura y divulgación de insights a stakeholders no técnicos.

#### 4. `python-projects/` 🐍
**"El Motor de Producción."**
Donde el código se vuelve profesional. Aquí residen las aplicaciones estructuradas.
*   **Modularidad:** Código fuente organizado en paquetes, separado de la lógica de notebooks.
*   **Microservicios:** APIs (ej. FastAPI), pipelines de entrenamiento y clientes de consumo.
*   **Calidad:** Testing, Linting y Type Checking configurados.

---

## 🚀 Proyectos Destacados

<div align="center">

| Proyecto | Descripción | Estado |
| :--- | :--- | :---: |
| **Credit Score AI** | **[Completado]** Evaluación de riesgo crediticio E2E. Incluye preprocesamiento robusto, entrenamiento de modelos, API con FastAPI y un cliente web interactivo. <br> 📺 **[Vídeo 1: Explicación y Demo](https://youtu.be/S5j4cSOEyik)** <br> 🚀 **[Vídeo 2: Despliegue del Servicio](https://youtu.be/V2LokJd68bU)** <br> 🔗 **[Ir al proyecto Credit Score AI](python-projects/credit-score/README.md)** | ![Active](https://img.shields.io/badge/Status-Active-brightgreen) |
| **Energy Imports** | *Work in progress*. Análisis y predicción de importaciones de energía. Se desplegará prontamente. | ![Pending](https://img.shields.io/badge/Status-Pending-orange) |
| **Retail Sales** | *Work in progress*. Optimización y pronóstico de ventas para retail. Se desplegará prontamente. | ![Pending](https://img.shields.io/badge/Status-Pending-orange) |
| **X-ray Diagnosis** | *Work in progress*. Clasificación de imágenes médicas mediante Deep Learning. Se desplegará prontamente. | ![Pending](https://img.shields.io/badge/Status-Pending-orange) |
| **API Consumption** | *Work in progress*. Módulo especializado en la integración y consumo eficiente de APIs externas. Se desplegará prontamente. | ![Pending](https://img.shields. :---: |
| **Project 2** | *[En Desarrollo]* | ![Pending](https://img.shields.io/badge/Status-Pending-orange) |

</div>

---

## 🛠 Stack Tecnológico

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,docker,git,githubactions,fastapi,sklearn,pandas,dvc" />
</div>

- **Lenguaje:** Python 3.10+
- **Control de Versiones:** Git & DVC
- **Contenedores:** Docker & Docker Compose
- **Orquestación:** GitHub Actions (CI/CD)
- **Frameworks ML:** Scikit-Learn, TensorFlow/PyTorch
- **API:** FastAPI

---

<div align="center">
<sub>Hecho con ❤️ para la comunidad de Data Science.</sub>
</div>
